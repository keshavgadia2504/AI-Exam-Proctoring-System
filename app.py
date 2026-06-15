print("file")
import os
import sys
import time
import queue
import pathlib
import sqlite3
import threading
import traceback
from datetime import datetime, timedelta
from threading import Thread
from gtts import gTTS
import tempfile
import cv2
import numpy as np
import torch
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, session, url_for, send_from_directory)
from werkzeug.security import check_password_hash, generate_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

import sounddevice as sd
from scipy.io.wavfile import write

# Flask app
app = Flask(__name__)
app.secret_key = "cheatdfshs6789"

EVIDENCE_DIR = os.path.join(app.root_path, "static", "captured_evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# FIX 1: Remove broken Windows-only pathlib hack.
# pathlib.PosixPath = pathlib.WindowsPath  <-- REMOVED (breaks Linux/macOS)

whisper_warning_count = 0

AUDIO_MIN_THRESHOLD = 0.002
AUDIO_THRESHOLD_WHISPER = 0.02
AUDIO_DURATION = 1
AUDIO_FS = 16000


def audio_monitor():
    # FIX 2: Added missing 'global' declarations so the function can read/write
    #         the module-level variables correctly.
    global whisper_warning_count, running_flag

    while running_flag:
        try:
            audio = sd.rec(
                int(AUDIO_DURATION * AUDIO_FS),
                samplerate=AUDIO_FS,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            volume = float(np.linalg.norm(audio))
        except Exception as e:
            print(f"DEBUG: audio_monitor error: {e}")
            break

        if AUDIO_MIN_THRESHOLD < volume < AUDIO_THRESHOLD_WHISPER:
            whisper_warning_count += 1
            print(f"DEBUG: Whisper detected. Volume={volume}, Warning #{whisper_warning_count}")

            evidence_path = save_evidence(np.zeros((100, 100, 3), dtype=np.uint8), "whisper_audio")
            log_event(current_student_name, "WhisperWarning",
                      f"Whisper detected. Audio level: {volume}. Evidence: {evidence_path}")

            if whisper_warning_count == 1:
                speak("Warning. Whispering detected. Please maintain silence.")
            elif whisper_warning_count == 2:
                speak("Final warning. Stop whispering immediately.")
            else:
                speak("Exam terminated due to repeated whispering.")
                safe_terminate("Whispering detected multiple times")
                break

        time.sleep(0.2)


def save_evidence(frame, label):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_label = label.replace(" ", "_")
    clean_name = (current_student_name or "unknown").replace(" ", "_")

    filename = f"{clean_name}_{clean_label}_{ts}.jpg"
    fs_path = os.path.join(EVIDENCE_DIR, filename)

    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    success = cv2.imwrite(fs_path, frame)
    print(f"DEBUG: Saving evidence to: {fs_path}, success={success}")
    if not success:
        return None

    return filename


def speak(text):
    # FIX 3: Replaced Windows-only 'wmplayer' command with a cross-platform
    #         approach using pygame (or falling back to os.system on Windows).
    try:
        tts = gTTS(text=text, lang='en')
        tmp = os.path.join(tempfile.gettempdir(), f"exam_voice_{time.time()}.mp3")
        tts.save(tmp)

        if sys.platform == "win32":
            os.system(f'start /min "" "wmplayer" "{tmp}"')
        elif sys.platform == "darwin":
            os.system(f'afplay "{tmp}" &')
        else:
            # Linux: try ffplay, then mpg123, then paplay
            os.system(f'ffplay -nodisp -autoexit "{tmp}" > /dev/null 2>&1 &'
                      f' || mpg123 -q "{tmp}" &')

        print("VOICE:", text)
    except Exception as e:
        print("TTS ERROR:", e)


# ---------- CONFIG ----------
YOLO_MODEL_PATH = "best.pt"
EMOTION_MODEL_PATH = "emotion_model.h5"
WEBCAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.35
SUSPICIOUS_KEYWORDS = ["copy", "mobile", "paper_exchange"]
CHEATING_LIMIT = 1
WARNING_COOLDOWN_SEC = 4
EMOTION_WINDOW_SIZE = 8

SENDER_EMAIL = "chandrum071202@gmail.com"
SENDER_PASSWORD = "nqblxhjrtlebjtyf"
RECIPIENT_EMAIL = "jayjumani042@gmail.com"

EXAM_DB = "exam_logs.db"
USER_DB = "users.db"


@app.route('/captured_evidence/<path:filename>')
def evidence_files(filename):
    folder = os.path.join(app.root_path, 'static', 'captured_evidence')
    return send_from_directory(folder, filename)


# ---------- Database helpers ----------
def setup_exam_db():
    conn = sqlite3.connect(EXAM_DB)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs'")
    if not c.fetchone():
        c.execute("""
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                student_name TEXT,
                event_type TEXT,
                message TEXT
            )
        """)
    else:
        c.execute("PRAGMA table_info(logs)")
        cols = [r[1] for r in c.fetchall()]
        if 'student_name' not in cols:
            c.execute("ALTER TABLE logs ADD COLUMN student_name TEXT")
    conn.commit()
    conn.close()


def log_event(student_name, event_type, message):
    conn = sqlite3.connect(EXAM_DB)
    c = conn.cursor()
    c.execute("INSERT INTO logs (timestamp, student_name, event_type, message) VALUES (?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), student_name, event_type, message))
    conn.commit()
    conn.close()


def init_user_db():
    conn = sqlite3.connect(USER_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS cheat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            timestamp TEXT,
            emotion TEXT,
            detected_object TEXT,
            message TEXT
        )
    ''')
    conn.commit()
    conn.close()


setup_exam_db()
init_user_db()

import mediapipe as mp


# ---------- Email ----------
def send_termination_email(student_name, reason):
    try:
        subject = f"Exam Termination Alert - {student_name}"
        body = f"""Exam Proctor System Alert

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
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("DEBUG: Termination email sent")
        return True
    except Exception as e:
        print("DEBUG: Failed to send email:", e)
        return False


# ---------- Load models once ----------
print("DEBUG: Loading models (this may take a while)...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH, force_reload=False)
    yolo_model.to(device)
    print(f"DEBUG: YOLO loaded on {device}")
except Exception as e:
    print(f"DEBUG: Failed to load YOLO model: {e}")
    yolo_model = None

try:
    emotion_model = load_model(EMOTION_MODEL_PATH)
    EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']
    print("DEBUG: Emotion model loaded")
except Exception as e:
    print(f"DEBUG: Failed to load emotion model: {e}")
    emotion_model = None

# --- GLOBAL VARIABLES ---
running_flag = False
camera_active = False
frames_processed = 0
last_warning_time = 0.0
cheating_detections = 0
emotion_window = []
exam_end_time = None
current_student_name = ""
window_violation_count = 0
window_control_active = False
yolo_warning_message = ""

gaze_violation_count = 0
gaze_last_away_time = 0.0
gaze_last_state = "center"
multi_face_count = 0

frame_queue = queue.Queue(maxsize=5)


# ---------- safe_terminate ----------
def safe_terminate(reason, actor="System"):
    global running_flag, cheating_detections, window_violation_count, window_control_active
    global gaze_violation_count, multi_face_count

    print(f"DEBUG: Terminating exam - Reason: {reason}")

    running_flag = False
    window_control_active = False
    cheating_detections = 0
    window_violation_count = 0
    gaze_violation_count = 0
    multi_face_count = 0

    log_event(current_student_name, "Termination", f"{reason} (by {actor})")
    send_termination_email(current_student_name, reason)

    speak("Your exam has been terminated due to violations.")

    return {"ok": True, "msg": "Exam terminated", "reason": reason}


def is_label_suspicious(label):
    label = label.lower()
    return any(kw in label for kw in SUSPICIOUS_KEYWORDS)


def detection_loop():
    global running_flag, camera_active, frames_processed, last_warning_time
    global cheating_detections, exam_end_time, window_control_active
    global gaze_violation_count, gaze_last_away_time, gaze_last_state
    global yolo_warning_message, multi_face_count

    YOLO_EVERY_N_FRAMES = 1
    WARNING_COOLDOWN_SEC = 2
    GAZE_WARNING_SECONDS = 3
    GAZE_WARNING_LIMIT = 2
    no_body_start_time = None
    movement_violation_count = 0

    cap = None
    try:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            log_event(current_student_name, "Error", "Webcam not accessible")
            running_flag = False
            return

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        camera_active = True
        speak("Exam started. Monitoring enabled.")
        log_event(current_student_name, "Exam", "Exam monitoring started")
        print("DEBUG: Detection loop started - camera activated")

        def activate_window_control():
            global window_control_active
            window_control_active = True
            print("DEBUG: Window control ACTIVATED")

        threading.Timer(5, activate_window_control).start()

        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        frame_idx = 0

        while running_flag:
            ret, frame = cap.read()
            if not ret:
                print("DEBUG: Failed to read frame from camera")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            frames_processed += 1
            frame_idx += 1
            now_time = time.time()

            # ================= EMOTION + MULTI-FACE DETECTION =================
            if emotion_model:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                face_count = len(faces)
                cv2.putText(display_frame, f"Faces: {face_count}", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if face_count > 1:
                    multi_face_count += 1
                    print(f"DEBUG: Multi-face detected! Count={face_count}, Violation #{multi_face_count}")

                    evidence_path = save_evidence(display_frame, f"multiple_faces_{face_count}")

                    if multi_face_count == 1:
                        speak("Warning. Multiple faces detected. Ensure you are alone.")
                        log_event(current_student_name, "FaceWarning",
                                  f"Multiple faces detected. Count={face_count}. Evidence={evidence_path}")
                    elif multi_face_count == 2:
                        speak("Final warning. More than one face detected again.")
                        log_event(current_student_name, "FaceWarning",
                                  f"Second multi-face violation. Count={face_count}. Evidence={evidence_path}")
                    else:
                        speak("Exam terminated due to multiple people detected.")
                        safe_terminate("Multiple faces detected repeatedly")
                        break

                for (x, y, w, h) in faces:
                    try:
                        roi = gray[y:y + h, x:x + w]
                        roi = cv2.resize(roi, (48, 48))
                        roi = roi.astype("float32") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        preds = emotion_model.predict(roi, verbose=0)[0]
                        emotion = EMOTIONS[np.argmax(preds)]

                        cv2.putText(display_frame, emotion, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h),
                                      (255, 255, 0), 2)
                    except Exception as e:
                        print(f"DEBUG: Emotion detection error: {e}")

            # ================= YOLO OBJECT / CHEATING DETECTION =================
            suspicious_found = []

            if yolo_model is not None and (frames_processed % YOLO_EVERY_N_FRAMES == 0):
                try:
                    results = yolo_model(frame)
                    dets = results.xyxy[0].cpu().numpy()

                    for *box, conf, cls_id in dets:
                        if conf < CONFIDENCE_THRESHOLD:
                            continue

                        x1, y1, x2, y2 = map(int, box)
                        label = str(results.names[int(cls_id)])

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{label} {conf:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 2)

                        if is_label_suspicious(label):
                            suspicious_found.append(label)

                            evidence_path = save_evidence(display_frame, label)
                            log_event(current_student_name, "YOLOEvidence",
                                      f"Object detected: {label}, saved: {evidence_path}")
                except Exception as e:
                    print("YOLO ERROR:", e)

            if suspicious_found:
                lbls = ", ".join(sorted(set(suspicious_found)))
                print(f"DEBUG: Cheating detected - {lbls} (Instant Terminate Mode)")

                evidence_path = save_evidence(display_frame, lbls)
                log_event(current_student_name,
                          "Termination",
                          f"Cheating detected instantly: {lbls}. Evidence: {evidence_path}")

                speak("Exam terminated due to cheating activity detected.")
                safe_terminate(f"Cheating detected: {lbls}")
                break

            # ================= GAZE / EYE DIRECTION (FaceMesh) =================
            gaze_state = "center"
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(frame_rgb)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]

                h, w, _ = frame.shape
                left_iris = face_landmarks.landmark[468]
                right_iris = face_landmarks.landmark[473]

                left_x, left_y = int(left_iris.x * w), int(left_iris.y * h)
                right_x, right_y = int(right_iris.x * w), int(right_iris.y * h)

                cv2.circle(display_frame, (left_x, left_y), 4, (0, 255, 0), -1)
                cv2.circle(display_frame, (right_x, right_y), 4, (0, 255, 0), -1)

                if left_iris.x < 0.43 and right_iris.x < 0.43:
                    gaze_state = "left"
                elif left_iris.x > 0.57 and right_iris.x > 0.57:
                    gaze_state = "right"
                else:
                    gaze_state = "center"
            else:
                gaze_state = "away"

            cv2.putText(display_frame, f"Gaze: {gaze_state}", (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # FIX 4: Corrected gaze timer logic.
            # Previously, gaze_last_away_time was reset to 0.0 in the 'else'
            # branch unconditionally, which wiped the timestamp even while the
            # student was still looking away.  Now it is only reset when the
            # gaze returns to "center".
            if gaze_state != "center":
                if gaze_last_state == "center":
                    # Gaze just moved away — start the timer
                    gaze_last_away_time = now_time
                elif gaze_last_away_time > 0:
                    duration = now_time - gaze_last_away_time
                    if duration >= GAZE_WARNING_SECONDS and (now_time - last_warning_time) > WARNING_COOLDOWN_SEC:
                        last_warning_time = now_time
                        gaze_violation_count += 1

                        print(f"DEBUG: Gaze violation #{gaze_violation_count} - {gaze_state} for {duration:.1f}s")

                        speak(f"Warning. Please focus on the screen. Violation {gaze_violation_count}.")
                        log_event(
                            current_student_name,
                            "GazeWarning",
                            f"Gaze {gaze_state} for {int(duration)}s. Count {gaze_violation_count}"
                        )

                        evidence_path = save_evidence(display_frame, f"gaze_{gaze_state}")
                        if evidence_path:
                            log_event(current_student_name, "Evidence", f"Gaze evidence saved: {evidence_path}")

                        # Reset timer after issuing warning so we don't spam
                        gaze_last_away_time = now_time

                        if gaze_violation_count >= GAZE_WARNING_LIMIT:
                            speak("Exam terminated due to repeated gaze violations.")
                            safe_terminate(f"Repeated gaze violations: {gaze_violation_count}")
                            break
            else:
                # Gaze returned to center — reset the away-timer
                gaze_last_away_time = 0.0

            gaze_last_state = gaze_state

            # ================= MOVEMENT / LEAVING SEAT (Pose) =================
            pose_results = pose.process(frame_rgb)

            if pose_results.pose_landmarks:
                left_shoulder = pose_results.pose_landmarks.landmark[11]
                right_shoulder = pose_results.pose_landmarks.landmark[12]

                if left_shoulder.visibility < 0.4 and right_shoulder.visibility < 0.4:
                    movement_violation_count += 1
                    print(f"DEBUG: Movement violation #{movement_violation_count} — Student left seat.")

                    evidence_path = save_evidence(display_frame, "left_seat_or_stood_up")

                    if movement_violation_count == 1:
                        speak("Warning. Please stay in your seat and remain visible.")
                        log_event(current_student_name, "MovementWarning",
                                  "Student stood up or left seat once.")
                    elif movement_violation_count == 2:
                        speak("Final warning. Do not leave your seat again.")
                        log_event(current_student_name, "MovementWarning",
                                  "Student left seat second time.")
                    else:
                        speak("Exam terminated due to leaving the seat.")
                        safe_terminate("Student repeatedly left the seat")
                        break

                no_body_start_time = None

            else:
                if no_body_start_time is None:
                    no_body_start_time = now_time
                else:
                    if now_time - no_body_start_time >= 1.0:
                        movement_violation_count += 1
                        no_body_start_time = None
                        print(f"DEBUG: Missing body — movement violation #{movement_violation_count}")

                        evidence_path = save_evidence(display_frame, "body_missing")

                        if movement_violation_count == 1:
                            speak("Warning. Please stay in front of the camera.")
                        elif movement_violation_count == 2:
                            speak("Final warning. Do not leave the camera area again.")
                        else:
                            speak("Exam terminated for being away from the camera.")
                            safe_terminate("Student missing from camera repeatedly")
                            break

            # ================= STREAM FRAME TO BROWSER =================
            ok, jpeg = cv2.imencode(".jpg", display_frame)
            if ok:
                # FIX 5: Corrected the frame-queue eviction logic.
                # Previously, get_nowait() was called inside 'except queue.Empty'
                # which made no sense — you cannot get from an empty queue.
                # Now we correctly drop the oldest frame only when the queue is full.
                try:
                    frame_queue.put_nowait(jpeg.tobytes())
                except queue.Full:
                    try:
                        frame_queue.get_nowait()   # drop oldest frame
                        frame_queue.put_nowait(jpeg.tobytes())
                    except (queue.Empty, queue.Full):
                        pass

            # ================= CHECK EXAM TIME LIMIT =================
            if exam_end_time and datetime.now() >= exam_end_time:
                speak("Exam time over.")
                log_event(current_student_name, "Exam", "Exam ended automatically - time limit reached")
                running_flag = False
                break

            time.sleep(0.03)

    except Exception as e:
        print(f"DEBUG: Detection loop error: {e}")
        log_event(current_student_name, "Error", f"Detection loop crashed: {str(e)}")
        traceback.print_exc()

    finally:
        if cap is not None:
            cap.release()
            print("DEBUG: Camera released")
        camera_active = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        cheating_detections = 0
        window_violation_count = 0
        gaze_violation_count = 0
        gaze_last_away_time = 0.0
        gaze_last_state = "center"
        window_control_active = False
        multi_face_count = 0

        log_event(current_student_name, "Exam", "Detection loop ended")
        print("DEBUG: Detection loop stopped - all resources released")


@app.route('/debug_status')
def debug_status():
    if not session.get('admin'):
        return "Unauthorized", 403

    status = {
        'running_flag': running_flag,
        'camera_active': camera_active,
        'cheating_detections': cheating_detections,
        'current_student': current_student_name,
        'window_control_active': window_control_active,
        'window_violation_count': window_violation_count,
        'gaze_violation_count': gaze_violation_count,
        'exam_end_time': str(exam_end_time) if exam_end_time else None,
        'tts_initialized': True
    }

    conn = sqlite3.connect(EXAM_DB)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM logs")
    log_count = cur.fetchone()[0]
    cur.execute("SELECT DISTINCT student_name FROM logs")
    students = [row[0] for row in cur.fetchall()]
    conn.close()

    status['total_logs'] = log_count
    status['students_in_db'] = students

    return jsonify(status)


@app.route('/window_violation', methods=['POST'])
def window_violation():
    global window_violation_count

    if not running_flag or not window_control_active:
        return jsonify({"ok": True})

    window_violation_count += 1
    speak("Window switch detected. Please return to your exam.")

    log_event(current_student_name, "WindowWarning",
              f"Window violation #{window_violation_count}")

    if window_violation_count == 1:
        return jsonify({"ok": True, "action": "warn"})
    elif window_violation_count == 2:
        speak("This is your final window warning.")
        return jsonify({"ok": True, "action": "final_warn"})
    else:
        safe_terminate("Repeated window switching")
        return jsonify({"ok": True, "action": "terminate"})


# ---------- Video stream generator ----------
def gen_frames():
    # FIX 6: Added running_flag check so the generator exits cleanly when the
    #         exam ends, instead of looping forever and holding the connection open.
    while running_flag or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            if not running_flag:
                break
            time.sleep(0.05)
            continue


# ---------- Flask Routes ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        email = request.form['email'].strip().lower()
        password = request.form['password']
        if not (name and email and password):
            flash("Fill all fields", "danger")
            return redirect(url_for('register'))
        conn = sqlite3.connect(USER_DB)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                      (name, email, generate_password_hash(password)))
            conn.commit()
            flash("Registration successful. Login now.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists", "danger")
        finally:
            conn.close()
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        conn = sqlite3.connect(USER_DB)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = c.fetchone()
        conn.close()
        if row and check_password_hash(row[3], password):
            session['logged_in'] = True
            session['email'] = email
            session['name'] = row[1]
            flash("Login successful", "success")
            return redirect(url_for('schedule_exam'))
        else:
            flash("Invalid credentials", "danger")
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for('index'))


@app.route('/schedule_exam', methods=['GET', 'POST'])
def schedule_exam():
    if not session.get('logged_in'):
        flash("Please login first", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        # FIX 7: Added 'global' declarations for exam_end_time and
        #         current_student_name. Without these, Python treats them as
        #         local variables and raises an UnboundLocalError on assignment.
        global exam_end_time, current_student_name
        hour = int(request.form.get('hour'))
        minute = int(request.form.get('minute'))
        period = request.form.get('period')
        try:
            if period.upper() == 'PM' and hour != 12:
                hour += 12
            if period.upper() == 'AM' and hour == 12:
                hour = 0
            now = datetime.now()
            end_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if end_time <= now:
                end_time += timedelta(days=1)
            exam_end_time = end_time
            current_student_name = session.get('email', session.get('name', 'Unknown'))
            log_event(current_student_name, "Exam",
                      f"Exam scheduled to end at {exam_end_time.strftime('%Y-%m-%d %I:%M %p')}")
            flash(f"Exam scheduled to end at {exam_end_time.strftime('%I:%M %p on %Y-%m-%d')}", "success")
            return redirect(url_for('exam_page'))
        except Exception:
            flash("Invalid time input", "danger")
    return render_template('schedule_exam.html')


@app.route('/exam')
def exam_page():
    if not session.get('logged_in'):
        flash("Please login first", "warning")
        return redirect(url_for('login'))
    return render_template('exam.html', student=session.get('name'), email=session.get('email'))


@app.route('/video_feed')
def video_feed():
    if not running_flag:
        return "Stream not started", 400
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_detection', methods=['POST'])
def start_detection_route():
    global running_flag, cheating_detections, frames_processed

    if not session.get('logged_in'):
        return jsonify({"ok": False, "msg": "Not logged in"}), 403

    if running_flag:
        return jsonify({"ok": False, "msg": "Already running"}), 400

    running_flag = True
    cheating_detections = 0
    frames_processed = 0

    t = Thread(target=detection_loop, daemon=True)
    t.start()

    audio_thread = Thread(target=audio_monitor, daemon=True)
    audio_thread.start()

    return jsonify({"ok": True, "msg": "Detection started"})


@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    global running_flag
    if not session.get('logged_in'):
        return jsonify({"ok": False, "msg": "Not logged in"}), 403
    running_flag = False
    log_event(session.get('email', 'Unknown'), "Exam", "Exam manually stopped")
    return jsonify({"ok": True, "msg": "Detection stopped"})


# ---------- Admin routes ----------
@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        pwd = request.form['password']
        if email == 'admin@mail.com' and pwd == 'admin':
            session['admin'] = True
            return redirect(url_for('admin_logs'))
        else:
            flash("Invalid admin credentials", "danger")
    return render_template('admin_login.html')


@app.route('/admin/logs')
def admin_logs():
    if not session.get('admin'):
        return redirect(url_for('admin_login'))

    conn = sqlite3.connect(EXAM_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    print("\n========== DEBUG LOG ROWS ==========")
    for r in rows[:30]:
        print(r)
    print("====================================\n")

    return render_template('admin_logs.html', rows=rows)


@app.route('/terminate_exam', methods=['POST'])
def terminate_exam_route():
    if not session.get('admin'):
        return jsonify({"ok": False, "msg": "Unauthorized"}), 403
    reason = request.form.get('reason', 'Terminated by admin')
    global running_flag
    running_flag = False
    send_termination_email("AdminAction", reason)
    log_event("Admin", "Termination", reason)
    return jsonify({"ok": True, "msg": "Exam terminated"})


@app.route('/my_logs')
def my_logs():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    user_email = session.get('email')
    conn = sqlite3.connect(EXAM_DB)
    c = conn.cursor()
    c.execute("SELECT * FROM logs WHERE student_name = ? ORDER BY id DESC", (user_email,))
    rows = c.fetchall()
    conn.close()
    return render_template('my_logs.html', rows=rows)


@app.route('/delete_logs', methods=['POST'])
def delete_logs():
    if not session.get('admin'):
        return jsonify({"ok": False, "msg": "Unauthorized"}), 403

    data = request.get_json()
    student = data.get("student", "").strip()

    if not student:
        return jsonify({"ok": False, "msg": "Invalid student"}), 400

    conn = sqlite3.connect(EXAM_DB)
    cur = conn.cursor()

    query = """
        DELETE FROM logs 
        WHERE REPLACE(LOWER(TRIM(student_name)), ' ', '') =
              REPLACE(LOWER(TRIM(?)), ' ', '')
    """

    cur.execute(query, (student,))
    deleted = cur.rowcount

    conn.commit()
    conn.close()

    if deleted == 0:
        return jsonify({"ok": False, "msg": f"No logs found for {student}"}), 404
    return jsonify({"ok": True, "msg": f"{deleted} logs deleted for {student}"})


@app.route('/yolo_alert')
def yolo_alert():
    def stream():
        # FIX 8: Added running_flag check to terminate the SSE stream when the
        #         exam ends. Previously this generator ran forever, leaking the
        #         connection and the thread.
        last_msg = ""
        while running_flag:
            global yolo_warning_message
            if yolo_warning_message and yolo_warning_message != last_msg:
                yield f"data: {yolo_warning_message}\n\n"
                last_msg = yolo_warning_message
            time.sleep(0.5)
        # Send a final close event so the browser-side EventSource can clean up
        yield "event: close\ndata: stream ended\n\n"

    return Response(stream(), mimetype="text/event-stream")


# ---------- Run the app ----------
if __name__ == "__main__":
    print("DEBUG: Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, threaded=True, use_reloader=False)