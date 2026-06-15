import pathlib
import cv2
import torch
import numpy as np
import threading
import time
from tkinter import Tk, Button, Label, messagebox, StringVar
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Fix PosixPath issue on Windows
pathlib.PosixPath = pathlib.WindowsPath

# ---------- Configuration ----------
YOLO_MODEL_PATH = "best.pt"            # replace if needed
EMOTION_MODEL_PATH = "emotion_model.h5"  # replace if needed
WEBCAM_INDEX = 0
YOLO_RUN_EVERY_N_FRAMES = 2  # run YOLO every N frames to save CPU
WARNING_COOLDOWN_SEC = 4     # seconds to ignore repeated immediate detections
CHEATING_DETECTIONS_TO_END = 2  # number of separate cheating detections before terminating exam
EMOTION_WINDOW_SIZE = 8      # smooth emotion over last N faces
# suspicious keywords in YOLO labels (lowercase substrings)
SUSPICIOUS_KEYWORDS = ["copy", "mobile", "paper_exchange"]

# ---------- Load models ----------
# Load YOLOv5 custom model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"YOLOv5 loaded on {device}")

# Load Keras emotion model
print("Loading emotion model...")
emotion_model = load_model(EMOTION_MODEL_PATH)
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']  # adapt if your classes differ
print("Emotion model loaded")

# ---------- Video capture ----------
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# ---------- App state ----------
running_flag = False
frame_lock = threading.Lock()
frames_processed = 0
last_warning_time = 0.0
cheating_detections = 0
detected_items_history = []  # list of (timestamp, label)
emotion_window = []

# Tkinter GUI
root = Tk()
root.title("Integrated YOLOv5 + Emotion Detection - Exam Monitor")

video_label = Label(root)
video_label.pack()

status_var = StringVar()
status_var.set("Frames: 0 | Cheating events: 0 | Cheating %: 0.00")
status_label = Label(root, textvariable=status_var)
status_label.pack(pady=6)

# Helper: check if a YOLO label is suspicious
def is_label_suspicious(label_text: str) -> bool:
    txt = label_text.lower()
    for kw in SUSPICIOUS_KEYWORDS:
        if kw in txt:
            return True
    return False

# Helper: show Tkinter warning safely from worker thread
def show_warning_dialog(title, message):
    # must call messagebox in main thread
    def _show():
        messagebox.showwarning(title, message)
    root.after(0, _show)

def show_info_dialog(title, message):
    def _show():
        messagebox.showinfo(title, message)
    root.after(0, _show)

# Main detection loop run in separate thread
def detection_loop():
    global running_flag, frames_processed, last_warning_time, cheating_detections
    frame_count = 0

    while running_flag:
        ret, frame = cap.read()
        if not ret:
            print("Warning: failed to read frame")
            break

        frame = cv2.flip(frame, 1)  # mirror for usability
        display_frame = frame.copy()
        frame_count += 1
        frames_processed += 1
        

        # ---------- Face + emotion detection (every frame) ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48,48))

        # Run emotion prediction on each face found
        face_emotion_labels = []
        for (x, y, w, h) in faces:
            try:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48,48))
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = emotion_model.predict(roi, verbose=0)
                emotion_label = EMOTIONS[np.argmax(preds[0])]
                face_emotion_labels.append((x, y, w, h, emotion_label))
            except Exception as e:
                # if something goes wrong with face crop/pred, skip
                print("Emotion predict error:", e)
                continue

        # Smooth emotion across recent frames to reduce flicker
        if face_emotion_labels:
            # take first face's emotion for windowing (simple)
            _, _, _, _, first_emotion = face_emotion_labels[0]
            emotion_window.append(first_emotion)
            if len(emotion_window) > EMOTION_WINDOW_SIZE:
                emotion_window.pop(0)
            # most frequent
            try:
                smooth_emotion = max(set(emotion_window), key=emotion_window.count)
            except ValueError:
                smooth_emotion = first_emotion
        else:
            smooth_emotion = None

        # ---------- YOLO object detection (every N frames) ----------
        detected_labels = []
        if frame_count % YOLO_RUN_EVERY_N_FRAMES == 0:
            # YOLO expects RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img_rgb, size=640)  # you can change size
            detections = results.xyxy[0]  # x1,y1,x2,y2,conf,class
            # draw boxes
            for *box, conf, cls in detections.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                cls = int(cls)
                label_name = results.names[cls] if hasattr(results, "names") else str(cls)
                label_text = f"{label_name} {conf:.2f}"
                detected_labels.append(label_name)

                # Draw bounding boxes and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---------- Draw face boxes and emotion ----------
        for (x, y, w, h, emotion_label) in face_emotion_labels:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 128, 0), 2)
            cv2.putText(display_frame, f"{emotion_label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
        if smooth_emotion:
            cv2.putText(display_frame, f"Emotion: {smooth_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)

        # ---------- Cheating detection logic ----------
        current_time = time.time()
        suspicious_found = False
        suspicious_labels = []

        for lbl in detected_labels:
            if is_label_suspicious(lbl):
                suspicious_found = True
                suspicious_labels.append(lbl)

        # If suspicious object(s) found, handle warnings and counts
        if suspicious_found:
            # If recent warning happened within cooldown, ignore as repeat
            if current_time - last_warning_time > WARNING_COOLDOWN_SEC:
                last_warning_time = current_time
                detected_items_history.append((current_time, suspicious_labels))
                # If this is the first cheating detection (cheating_detections==0) -> warn
                if cheating_detections == 0:
                    cheating_detections += 1
                    # Show overlay text and dialog
                    cv2.putText(display_frame, "WARNING: Suspicious object detected!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    print("Warning: suspicious object(s):", suspicious_labels)
                    show_warning_dialog("Warning", f"Suspicious object detected: {', '.join(suspicious_labels)}.\nThis is the first warning.")
                else:
                    # second (or further) independent detection -> increment and may end exam
                    cheating_detections += 1
                    print("Cheating event registered. Count:", cheating_detections)
                    show_warning_dialog("Cheating Detected", f"Suspicious object detected again: {', '.join(suspicious_labels)}.\nCheating events: {cheating_detections}")

                    # If threshold reached -> terminate exam (stop capture and present dialog)
                    if cheating_detections >= CHEATING_DETECTIONS_TO_END:
                        # draw final overlay and end
                        cv2.putText(display_frame, "EXAM TERMINATED FOR CHEATING", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 4)
                        # show info dialog and stop
                        show_info_dialog("Exam Terminated", "Exam terminated due to repeated cheating detections.")
                        # Set running_flag False to end loop
                        running_flag = False

        # ---------- Compute cheating percentage and update status ----------
        try:
            cheating_percent = (cheating_detections / frames_processed) * 100.0 if frames_processed > 0 else 0.0
        except Exception:
            cheating_percent = 0.0

        status_text = f"Frames: {frames_processed} | Cheating events: {cheating_detections} | Cheating %: {cheating_percent:.2f}"
        status_var.set(status_text)

        # ---------- Convert for Tkinter and display ----------
        img_rgb_disp = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_disp)
        img_tk = ImageTk.PhotoImage(img_pil)

        # update on main thread
        def update_label():
            video_label.config(image=img_tk)
            video_label.image = img_tk
        root.after(0, update_label)

        # short sleep to keep CPU reasonable (fine tune as needed)
        time.sleep(0.02)

    # cleanup when loop ends
    with frame_lock:
        if cap.isOpened():
            cap.release()
    print("Detection loop stopped")

# Control functions for buttons
def start_detection():
    global running_flag, detection_thread, frames_processed, cheating_detections, detected_items_history, emotion_window, last_warning_time
    if running_flag:
        return
    # reset counters if you want to start fresh each time (optional)
    frames_processed = 0
    cheating_detections = 0
    detected_items_history = []
    emotion_window = []
    last_warning_time = 0.0

    running_flag = True
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    print("Detection started")

def stop_detection():
    global running_flag
    if not running_flag:
        return
    running_flag = False
    # detection loop will release cap and exit gracefully
    print("Stopping detection...")

# Buttons
start_btn = Button(root, text="Start Detection", command=start_detection, bg="green", fg="white")
start_btn.pack(pady=6)

stop_btn = Button(root, text="Stop Detection", command=stop_detection, bg="red", fg="white")
stop_btn.pack(pady=6)

# Ensure the camera is released when user closes the window
def on_closing():
    global running_flag
    if running_flag:
        if messagebox.askokcancel("Quit", "Detection is running. Do you want to stop and quit?"):
            running_flag = False
            root.destroy()
    else:
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
