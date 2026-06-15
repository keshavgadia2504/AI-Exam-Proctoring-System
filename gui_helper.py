import pathlib
import cv2
import torch
import numpy as np
import threading
import time
from tkinter import Tk, Button, Label, messagebox, StringVar, Toplevel, Entry, Frame, OptionMenu
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from datetime import datetime, timedelta
import pyttsx3
import sqlite3
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Fix PosixPath issue for Windows
pathlib.PosixPath = pathlib.WindowsPath

# ---------- Configuration ----------
YOLO_MODEL_PATH = "best.pt"
EMOTION_MODEL_PATH = "emotion_model.h5"
WEBCAM_INDEX = 0
YOLO_RUN_EVERY_N_FRAMES = 2
WARNING_COOLDOWN_SEC = 4
EMOTION_WINDOW_SIZE = 8
CONFIDENCE_THRESHOLD = 0.55  # noise removal threshold
SUSPICIOUS_KEYWORDS = ["copy", "mobile", "paper_exchange"]
CHEATING_LIMIT = 3

# Email Configuration
SENDER_EMAIL = "ajayduraisamy@gmail.com"
SENDER_PASSWORD = "qznyhbqpasuzvutv"
RECIPIENT_EMAIL = "jeevar122002@gmail.com"

# ---------- Initialize voice engine ----------
engine = pyttsx3.init()
engine.setProperty('rate', 165)

# ---------- Email Function ----------
def send_termination_email(student_name, reason):
    """Send email notification when exam is terminated"""
    try:
        subject = f"Exam Termination Alert - {student_name}"
        body = f"""
        Exam Proctor System Alert
        
        Student: {student_name}
        Status: EXAM TERMINATED
        Reason: {reason}
        Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        This is an automated notification from the AI Exam Proctor System.
        """
        
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
        server.quit()
        
        print(f"Termination email sent for {student_name}")
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

# ---------- Database setup ----------
def setup_database():
    conn = sqlite3.connect("exam_logs.db")
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        # Create new table with student_name column
        cursor.execute("""
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                student_name TEXT,
                event_type TEXT,
                message TEXT
            )
        """)
        print("New database table created with student_name column")
    else:
        # Check if student_name column exists
        cursor.execute("PRAGMA table_info(logs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'student_name' not in columns:
            # Add student_name column to existing table
            cursor.execute("ALTER TABLE logs ADD COLUMN student_name TEXT")
            print("Added student_name column to existing table")
    
    conn.commit()
    conn.close()

# Initialize database
setup_database()

def log_event(student_name, event_type, message):
    conn = sqlite3.connect("exam_logs.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO logs (timestamp, student_name, event_type, message) VALUES (?, ?, ?, ?)",
                   (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), student_name, event_type, message))
    conn.commit()
    conn.close()

# ---------- Load Models ----------
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"YOLOv5 loaded on {device}")

print("Loading Emotion model...")
emotion_model = load_model(EMOTION_MODEL_PATH)
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']
print("Emotion model loaded")

# ---------- App State ----------
running_flag = False
frames_processed = 0
last_warning_time = 0.0
cheating_detections = 0
emotion_window = []
exam_end_time = None
current_student_name = ""
camera_active = False
window_violation_count = 0
camera_start_time = None
window_control_active = False  # New flag for window control activation

# ---------- GUI ----------
root = Tk()
root.title("AI Exam Proctor System")
root.geometry("800x700")
root.configure(bg="#0B132B")

# Custom styling
BUTTON_STYLE = {
    "font": ("Arial", 12, "bold"),
    "width": 20,
    "pady": 8
}

HEADER_STYLE = {
    "font": ("Arial", 22, "bold"),
    "bg": "#0B132B",
    "fg": "#F8F32B",
    "pady": 10
}

LABEL_STYLE = {
    "font": ("Arial", 12),
    "bg": "#0B132B",
    "fg": "white"
}

# Header
header = Label(root, text="AI EXAM MONITORING SYSTEM", **HEADER_STYLE)
header.pack()

# Video display
video_frame = Frame(root, bg="#0B132B", relief="ridge", bd=2)
video_frame.pack(pady=10)
video_label = Label(video_frame, bg="#0B132B")
video_label.pack(padx=5, pady=5)

# Status display
status_frame = Frame(root, bg="#1C2541", relief="sunken", bd=1)
status_frame.pack(pady=10, padx=20, fill="x")

status_var = StringVar()
status_var.set("Frames: 0 | Warnings: 0 | Status: Ready")
status_label = Label(status_frame, textvariable=status_var, font=("Arial", 11, "bold"),
                     fg="#6FFFE9", bg="#1C2541", pady=8)
status_label.pack()

# Button frame
button_frame = Frame(root, bg="#0B132B")
button_frame.pack(pady=15)

# ---------- Helper Functions ----------
def speak(text):
    threading.Thread(target=lambda: engine.say(text) or engine.runAndWait(), daemon=True).start()

def is_label_suspicious(label):
    label = label.lower()
    return any(kw in label for kw in SUSPICIOUS_KEYWORDS)

def show_dialog(title, msg, type="info"):
    if type == "info": messagebox.showinfo(title, msg)
    elif type == "warn": messagebox.showwarning(title, msg)
    elif type == "error": messagebox.showerror(title, msg)

def convert_12h_to_24h(time_str, period):
    """Convert 12-hour format to 24-hour format"""
    try:
        # Parse the time string (e.g., "2:30")
        if ':' in time_str:
            hour, minute = map(int, time_str.split(':'))
        else:
            hour = int(time_str)
            minute = 0
            
        # Convert to 24-hour format
        if period.upper() == "PM" and hour != 12:
            hour += 12
        elif period.upper() == "AM" and hour == 12:
            hour = 0
            
        return hour, minute
    except:
        raise ValueError("Invalid time format")

def terminate_exam(reason):
    """Handle exam termination with email notification"""
    global running_flag, window_violation_count, cheating_detections
    
    running_flag = False
    
    # Log the termination
    log_event(current_student_name, "Termination", f"Exam terminated: {reason}")
    
    # Send email notification
    email_sent = send_termination_email(current_student_name, reason)
    
    # Speak and show message
    speak("Exam terminated due to violation. Administrator has been notified.")
    show_dialog("Exam Terminated", 
                f"Exam terminated for: {reason}\n\n"
                f"Email notification: {'Sent' if email_sent else 'Failed'}", 
                "error")
    
    # Reset counters
    window_violation_count = 0
    cheating_detections = 0
    
    # Open admin panel
    root.after(1000, open_admin_panel)

# ---------- Fullscreen Warning System ----------
def show_fullscreen_warning(reason="window minimized"):
    """Show fullscreen warning when camera/minimize detected."""
    global window_violation_count
    
    warn_win = Toplevel(root)
    warn_win.attributes("-fullscreen", True)
    warn_win.configure(bg="black")
    
    # Make it a top-level window that grabs all focus
    warn_win.transient(root)
    warn_win.grab_set()
    warn_win.focus_force()

    Label(warn_win, text="⚠️ WARNING: UNAUTHORIZED ACTION DETECTED ⚠️",
          font=("Arial", 36, "bold"), fg="red", bg="black").pack(pady=100)
    Label(warn_win, text=f"Reason: {reason}",
          font=("Arial", 24, "bold"), fg="white", bg="black").pack(pady=20)
    Label(warn_win, text="Please return to your exam immediately!",
          font=("Arial", 24, "bold"), fg="white", bg="black").pack(pady=30)
    
    violation_text = f"Violation {window_violation_count + 1} of 3 - Next violation will terminate exam!"
    Label(warn_win, text=violation_text,
          font=("Arial", 20, "bold"), fg="yellow", bg="black").pack(pady=20)

    def close_warning():
        nonlocal warn_win
        warn_win.destroy()
        speak("Warning acknowledged. Continue your exam carefully.")
        log_event(current_student_name, "Warning", f"Student {reason} - violation {window_violation_count}")
        
        # Reset focus to main window
        root.focus_force()
        root.lift()
        root.attributes('-topmost', True)
        root.after(1000, lambda: root.attributes('-topmost', False))

    Button(warn_win, text="I'm Back - Continue Exam", command=close_warning,
           font=("Arial", 18, "bold"), bg="green", fg="white", width=25, pady=15).pack(pady=80)

    speak(f"Warning! {reason} detected. Return to exam immediately!")

    # Force the warning window to stay on top
    def keep_on_top():
        if warn_win.winfo_exists():
            warn_win.lift()
            warn_win.attributes('-topmost', True)
            warn_win.after(1000, keep_on_top)
    
    keep_on_top()

# ---------- Detection Logic ----------
def detection_loop():
    global running_flag, frames_processed, last_warning_time, cheating_detections, exam_end_time, camera_active, camera_start_time, window_control_active
    
    # Set camera start time
    camera_start_time = time.time()
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        show_dialog("Camera Error", "Could not access webcam", "error")
        running_flag = False
        camera_active = False
        return

    camera_active = True
    log_event(current_student_name, "Exam", "Exam started with camera monitoring")

    while running_flag and camera_active:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed")
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        frames_processed += 1

        # Activate window control after 5 seconds
        if not window_control_active and (time.time() - camera_start_time) > 5:
            window_control_active = True
            print("Window control activated after 5 seconds")

        # Time check
        if exam_end_time and datetime.now() >= exam_end_time:
            show_dialog("Exam Ended", "Exam time over!", "info")
            speak("Exam time completed.")
            running_flag = False
            log_event(current_student_name, "Exam", "Exam ended by time")
            break

        # Face detection + emotion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(48, 48))
        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_model.predict(roi, verbose=0)
            emotion = EMOTIONS[np.argmax(preds[0])]
            emotion_window.append(emotion)
            if len(emotion_window) > EMOTION_WINDOW_SIZE:
                emotion_window.pop(0)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
            cv2.putText(display_frame, emotion, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # YOLO detection
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size=640)
        detections = results.xyxy[0].cpu().numpy()
        detected_labels = []
        for *box, conf, cls in detections:
            if conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, box)
            label_name = results.names[int(cls)]
            detected_labels.append(label_name)
            # Draw thick red border
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(display_frame, f"{label_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Suspicious logic
        suspicious_labels = [lbl for lbl in detected_labels if is_label_suspicious(lbl)]
        current_time = time.time()

        if suspicious_labels and (current_time - last_warning_time > WARNING_COOLDOWN_SEC):
            last_warning_time = current_time
            cheating_detections += 1
            event_msg = f"Suspicious: {', '.join(suspicious_labels)}"
            log_event(current_student_name, "Warning", event_msg)

            if cheating_detections == 1:
                speak("Warning detected! Please focus on your screen.")
                show_dialog("Warning", "Suspicious object detected. Please focus.", "warn")

            elif cheating_detections == 2:
                speak("Final warning! Any further cheating will terminate the exam.")
                show_dialog("Final Warning", "Final Warning! Stop cheating.", "warn")

            elif cheating_detections >= CHEATING_LIMIT:
                terminate_exam(f"Repeated cheating detected: {', '.join(suspicious_labels)}")
                break

        # Status Update
        status_var.set(f"Student: {current_student_name} | Frames: {frames_processed} | Warnings: {cheating_detections} | Status: Running")

        # Show display
        img_rgb_disp = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_disp)
        img_tk = ImageTk.PhotoImage(img_pil)
        root.after(0, lambda: video_label.config(image=img_tk))
        video_label.image = img_tk

        time.sleep(0.03)

    cap.release()
    camera_active = False
    window_control_active = False
    print("Stopped Detection")

# ---------- Database Viewer ----------
def open_admin_panel():
    admin = Toplevel(root)
    admin.title("Admin Login")
    admin.geometry("350x250")
    admin.configure(bg="#1F2833")
    admin.resizable(False, False)

    Label(admin, text="Admin Login", font=("Arial", 16, "bold"), bg="#1F2833", fg="#66FCF1").pack(pady=15)
    
    form_frame = Frame(admin, bg="#1F2833")
    form_frame.pack(pady=10)
    
    Label(form_frame, text="Email:", bg="#1F2833", fg="white", font=("Arial", 10)).grid(row=0, column=0, sticky="e", padx=5, pady=8)
    email_entry = Entry(form_frame, width=25, font=("Arial", 10))
    email_entry.grid(row=0, column=1, padx=5, pady=8)
    
    Label(form_frame, text="Password:", bg="#1F2833", fg="white", font=("Arial", 10)).grid(row=1, column=0, sticky="e", padx=5, pady=8)
    pass_entry = Entry(form_frame, show="*", width=25, font=("Arial", 10))
    pass_entry.grid(row=1, column=1, padx=5, pady=8)

    def login_admin():
        if email_entry.get() == "admin@mail.com" and pass_entry.get() == "admin":
            admin.destroy()
            show_database()
        else:
            show_dialog("Access Denied", "Invalid admin credentials", "error")

    Button(admin, text="Login", command=login_admin, bg="#45A29E", fg="white", 
           font=("Arial", 11, "bold"), width=12, pady=5).pack(pady=15)

def show_database():
    db_win = Toplevel(root)
    db_win.title("Exam Logs Database")
    db_win.geometry("1000x500")
    db_win.configure(bg="#0B0C10")

    Label(db_win, text="Exam Event Logs", font=("Arial", 16, "bold"), bg="#0B0C10", fg="#66FCF1").pack(pady=10)

    conn = sqlite3.connect("exam_logs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()

    # Create a frame for the table headers
    header_frame = Frame(db_win, bg="#1F2833")
    header_frame.pack(fill="x", padx=20)
    
    headers = ["ID", "Timestamp", "Student Name", "Event Type", "Message"]
    widths = [50, 150, 150, 100, 300]
    
    for i, header in enumerate(headers):
        Label(header_frame, text=header, bg="#45A29E", fg="white", 
              font=("Arial", 10, "bold"), width=widths[i]//10, anchor="w").grid(row=0, column=i, sticky="ew", padx=1, pady=1)

    from tkinter.scrolledtext import ScrolledText
    text_frame = Frame(db_win, bg="#0B0C10")
    text_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    box = ScrolledText(text_frame, width=120, height=20, bg="#1F2833", fg="white", font=("Consolas", 8))
    box.pack(fill="both", expand=True)
    
    for row in data:
        # Handle cases where student_name might be None (for old records)
        student_name = row[2] if row[2] is not None else "Unknown"
        # Truncate long messages for display
        message = row[4] if len(row[4]) < 50 else row[4][:47] + "..."
        
        line = f"{row[0]:<4} {row[1]:<18} {student_name:<15} {row[3]:<12} {message:<50}\n"
        box.insert("end", line)
    
    box.config(state="disabled")

# ---------- Student Login ----------
def open_student_login():
    login = Toplevel(root)
    login.title("Student Login")
    login.geometry("400x300")
    login.configure(bg="#1B263B")
    login.resizable(False, False)

    Label(login, text="Student Login", font=("Arial", 18, "bold"), bg="#1B263B", fg="#F8F32B").pack(pady=20)

    form_frame = Frame(login, bg="#1B263B")
    form_frame.pack(pady=20)
    
    Label(form_frame, text="Full Name:", bg="#1B263B", fg="white", 
          font=("Arial", 12)).grid(row=0, column=0, sticky="e", padx=10, pady=15)
    name_entry = Entry(form_frame, font=("Arial", 12), width=20)
    name_entry.grid(row=0, column=1, padx=10, pady=15)
    
    Label(form_frame, text="Student ID:", bg="#1B263B", fg="white", 
          font=("Arial", 12)).grid(row=1, column=0, sticky="e", padx=10, pady=15)
    id_entry = Entry(form_frame, font=("Arial", 12), width=20)
    id_entry.grid(row=1, column=1, padx=10, pady=15)

    def confirm_login():
        global current_student_name
        name = name_entry.get().strip()
        student_id = id_entry.get().strip()
        
        if not name or not student_id:
            show_dialog("Error", "Please enter both name and student ID", "error")
            return
            
        current_student_name = f"{name} ({student_id})"
        login.destroy()
        open_exam_setup()

    Button(login, text="Login & Continue", command=confirm_login, bg="#3E92CC", fg="white",
           font=("Arial", 12, "bold"), width=15, pady=8).pack(pady=20)

# ---------- Exam Setup ----------
def open_exam_setup():
    setup = Toplevel(root)
    setup.title("Set Exam End Time")
    setup.geometry("500x350")
    setup.configure(bg="#1B263B")
    setup.resizable(False, False)

    Label(setup, text=f"Set Exam End Time for {current_student_name}", 
          bg="#1B263B", fg="white", font=("Arial", 14, "bold")).pack(pady=20)
    
    Label(setup, text="Enter End Time:", bg="#1B263B", fg="white",
          font=("Arial", 12)).pack(pady=10)
    
    time_frame = Frame(setup, bg="#1B263B")
    time_frame.pack(pady=10)
    
    # Time input frame
    input_frame = Frame(time_frame, bg="#1B263B")
    input_frame.pack(pady=5)
    
    # Hour entry
    hour_entry = Entry(input_frame, font=("Arial", 14), width=5, justify="center")
    hour_entry.grid(row=0, column=0, padx=5)
    Label(input_frame, text=":", bg="#1B263B", fg="white", font=("Arial", 14)).grid(row=0, column=1)
    
    # Minute entry
    minute_entry = Entry(input_frame, font=("Arial", 14), width=5, justify="center")
    minute_entry.grid(row=0, column=2, padx=5)
    
    # AM/PM dropdown
    period_var = StringVar(value="AM")
    period_dropdown = OptionMenu(input_frame, period_var, "AM", "PM")
    period_dropdown.config(font=("Arial", 12), width=4)
    period_dropdown.grid(row=0, column=3, padx=5)
    
    Label(setup, text="Example: 2:30 PM", bg="#1B263B", fg="#CCCCCC",
          font=("Arial", 10)).pack(pady=5)

    def confirm():
        global exam_end_time
        try:
            hour_str = hour_entry.get().strip()
            minute_str = minute_entry.get().strip()
            period = period_var.get()
            
            # Validate inputs
            if not hour_str or not minute_str:
                show_dialog("Error", "Please enter both hour and minute", "error")
                return
                
            hour = int(hour_str)
            minute = int(minute_str)
            
            # Validate ranges
            if hour < 1 or hour > 12:
                show_dialog("Error", "Hour must be between 1 and 12", "error")
                return
            if minute < 0 or minute > 59:
                show_dialog("Error", "Minute must be between 0 and 59", "error")
                return
            
            # Convert to 24-hour format
            hour_24, minute_24 = convert_12h_to_24h(f"{hour}:{minute}", period)
            
            now = datetime.now()
            end_time = now.replace(hour=hour_24, minute=minute_24, second=0, microsecond=0)
            
            # If the time has already passed today, schedule for tomorrow
            if end_time <= now:
                end_time += timedelta(days=1)
                
            exam_end_time = end_time
            
            # Format for display
            display_time = end_time.strftime("%I:%M %p on %Y-%m-%d")
            log_event(current_student_name, "Exam", f"Exam scheduled to end at {display_time}")
            
            show_dialog("Exam Scheduled", f"Exam will end at {display_time}", "info")
            setup.destroy()
            start_detection()
            
        except ValueError as e:
            show_dialog("Error", "Please enter valid numbers for hour and minute", "error")
        except Exception as e:
            show_dialog("Error", f"Invalid time format: {str(e)}", "error")

    Button(setup, text="Start Exam", command=confirm, bg="#4CAF50", fg="white",
           font=("Arial", 12, "bold"), width=15, pady=8).pack(pady=20)

# ---------- Control Buttons ----------
Button(button_frame, text="Start Exam", command=open_student_login, bg="#1C7293", fg="white",
       **BUTTON_STYLE).grid(row=0, column=0, padx=10, pady=5)

Button(button_frame, text="Stop Detection", command=lambda: stop_detection(), bg="#C3073F", fg="white",
       **BUTTON_STYLE).grid(row=0, column=1, padx=10, pady=5)

Button(button_frame, text="View Database (Admin)", command=open_admin_panel, bg="#45A29E", fg="white",
       **BUTTON_STYLE).grid(row=1, column=0, columnspan=2, padx=10, pady=5)

def start_detection():
    global running_flag, window_violation_count, window_control_active
    if running_flag:
        return
    running_flag = True
    window_violation_count = 0  # Reset violation counter
    window_control_active = False  # Reset window control flag
    threading.Thread(target=detection_loop, daemon=True).start()
    print("Detection started")

def stop_detection():
    global running_flag
    running_flag = False
    log_event(current_student_name, "Exam", "Exam manually stopped")
    speak("Exam stopped.")
    status_var.set("Frames: 0 | Warnings: 0 | Status: Stopped")
    print("Detection stopped")

def on_close():
    global running_flag
    running_flag = False
    root.destroy()

# ---------- Window Violation Handling ----------
# ---------- Window Warning System ----------
def show_window_warning(reason="window minimized"):
    """Show normal warning when camera/minimize detected."""
    global window_violation_count
    
    warn_win = Toplevel(root)
    warn_win.title("⚠️ Exam Warning")
    warn_win.geometry("500x300")
    warn_win.configure(bg="#1C2541")
    warn_win.resizable(False, False)
    
    # Make it a top-level window that grabs focus
    warn_win.transient(root)
    warn_win.grab_set()
    warn_win.focus_force()

    # Center the warning window
    warn_win.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - warn_win.winfo_width()) // 2
    y = root.winfo_y() + (root.winfo_height() - warn_win.winfo_height()) // 2
    warn_win.geometry(f"+{x}+{y}")

    Label(warn_win, text="⚠️ WARNING: UNAUTHORIZED ACTION DETECTED",
          font=("Arial", 14, "bold"), fg="red", bg="#1C2541").pack(pady=20)
    
    Label(warn_win, text=f"Reason: {reason}",
          font=("Arial", 12, "bold"), fg="white", bg="#1C2541").pack(pady=10)
    
    Label(warn_win, text="Please return to your exam immediately!",
          font=("Arial", 12), fg="white", bg="#1C2541").pack(pady=10)
    
    violation_text = f"Violation {window_violation_count + 1} of 3"
    if window_violation_count < 2:
        violation_text += " - Next violation will result in stricter action!"
    else:
        violation_text += " - Next violation will TERMINATE your exam!"
    
    Label(warn_win, text=violation_text,
          font=("Arial", 11, "bold"), fg="yellow", bg="#1C2541").pack(pady=15)

    def close_warning():
        nonlocal warn_win
        warn_win.destroy()
        speak("Warning acknowledged. Continue your exam carefully.")
        log_event(current_student_name, "Warning", f"Student {reason} - violation {window_violation_count}")
        
        # Reset focus to main window
        root.focus_force()
        root.lift()
        root.attributes('-topmost', True)
        root.after(1000, lambda: root.attributes('-topmost', False))

    Button(warn_win, text="I'm Back - Continue Exam", command=close_warning,
           font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", width=20, pady=8).pack(pady=20)

    speak(f"Warning! {reason} detected. Return to exam immediately!")

    # Force the warning window to stay on top
    warn_win.lift()
    warn_win.attributes('-topmost', True)

# ---------- Window Violation Handling ----------
def handle_window_violation(reason):
    """Handle student minimizing or switching windows."""
    global window_violation_count, running_flag, window_control_active

    if not running_flag or not window_control_active:
        return

    window_violation_count += 1

    if window_violation_count == 1:
        # 1st Violation → Show warning
        root.after(100, lambda: show_window_warning(reason))
        log_event(current_student_name, "Warning", f"Window violation: {reason} - First warning")
    
    elif window_violation_count == 2:
        # 2nd Violation → Show warning
        root.after(100, lambda: show_window_warning(f"{reason} - SECOND WARNING"))
        log_event(current_student_name, "Warning", f"Window violation: {reason} - Second warning")
    
    elif window_violation_count >= 3:
        # 3rd Violation → Terminate exam
        terminate_exam(f"Repeated window violations: {reason}")

def monitor_window_focus():
    """Monitor if the user minimizes or switches away from the exam window."""
    global window_control_active, running_flag
    if running_flag and window_control_active:
        # If window is minimized or not focused
        if not root.focus_displayof() or root.state() == "iconic":
            handle_window_violation("Window minimized or focus lost")
    # Check again every 2 seconds
    root.after(2000, monitor_window_focus)

        
root.protocol("WM_DELETE_WINDOW", on_close)

# 🟢 Start monitoring window minimize/focus loss
root.after(2000, monitor_window_focus)

root.mainloop()


        
root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
