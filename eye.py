import cv2
import dlib
import numpy as np

# Load face detector and landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_eye_center(eye_points, facial_landmarks, frame_gray):
    eye_region = np.array(
        [(facial_landmarks.part(p).x, facial_landmarks.part(p).y) for p in eye_points],
        np.int32,
    )

    mask = np.zeros_like(frame_gray)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # FIX 1: Guard against zero-size crop (eye near frame edge)
    if max_x <= min_x or max_y <= min_y:
        return 1.0, None

    gray_eye = eye[min_y:max_y, min_x:max_x]

    # FIX 2: Use Otsu thresholding — adapts to lighting automatically
    _, thresh_eye = cv2.threshold(gray_eye, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # FIX 3: Guard against empty thresh result
    if thresh_eye is None or thresh_eye.size == 0:
        return 1.0, None

    height, width = thresh_eye.shape

    # Find pupil (largest contour)
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cx, cy = None, None  # FIX 4: Use None instead of (0,0) as sentinel
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(cnt)
        cx = int(x + w / 2) + min_x
        cy = int(y + h / 2) + min_y

    # Gaze ratio
    left_side = thresh_eye[:, : width // 2]
    right_side = thresh_eye[:, width // 2 :]
    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)

    # FIX 5: Avoid division by zero on both sides
    if left_white + right_white == 0:
        gaze_ratio = 1.0  # neutral fallback
    else:
        gaze_ratio = left_white / (right_white + 1)

    pupil_center = (cx, cy) if cx is not None else None
    return gaze_ratio, pupil_center


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # FIX 6: Skip iteration if frame capture failed
    if not ret or frame is None:
        continue

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame_gray)

    for face in faces:
        landmarks = predictor(frame_gray, face)

        left_ratio, left_center = get_eye_center([36, 37, 38, 39, 40, 41], landmarks, frame_gray)
        right_ratio, right_center = get_eye_center([42, 43, 44, 45, 46, 47], landmarks, frame_gray)

        gaze = (left_ratio + right_ratio) / 2

        # FIX 7: Only draw pupil dots when a valid center was detected
        if left_center is not None:
            cv2.circle(frame, left_center, 4, (0, 0, 255), -1)
        if right_center is not None:
            cv2.circle(frame, right_center, 4, (0, 0, 255), -1)

        # Gaze direction label
        if gaze <= 0.8:
            text = "Looking Right"
        elif gaze > 1.2:
            text = "Looking Left"
        else:
            text = "Looking Center"

        cv2.putText(frame, text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Eye Gaze Detection + Pupil", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()