import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained emotion model
model = load_model("emotion_model.h5")

# Emotion labels (must match your training order)
emotions = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# To smooth predictions over frames
emotion_window = []
window_size = 10  # average over last 10 frames for stability

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Extract face ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        preds = model.predict(roi, verbose=0)
        emotion = emotions[np.argmax(preds[0])]

        # Smooth output
        emotion_window.append(emotion)
        if len(emotion_window) > window_size:
            emotion_window.pop(0)

        # Take most frequent emotion in recent frames
        final_emotion = max(set(emotion_window), key=emotion_window.count)

        # Draw face box and emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, final_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
