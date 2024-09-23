import cv2
import dlib
import face_recognition
import numpy as np
import os
from app.models.face_recognition_model import load_model

# Load known faces and encodings
known_face_encodings, known_face_names = load_model('app/face_model.pkl')

# EAR threshold for blinking detection (lowered slightly for more accuracy)
EAR_THRESHOLD = 0.21
CONSECUTIVE_FRAMES = 5  # Increase the number of frames required to detect a blink

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('app/shape_predictor_68_face_landmarks.dat')

blink_counter = 0
blink_detected = False

def compute_ear(eye_landmarks):
    vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def is_blinking(landmarks):
    left_eye = np.array([landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[40], landmarks[41]])
    right_eye = np.array([landmarks[42], landmarks[43], landmarks[44], landmarks[45], landmarks[46], landmarks[47]])

    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear

def detect_face_liveness(landmarks):
    global blink_counter, blink_detected
    ear = is_blinking(landmarks)

    if ear < EAR_THRESHOLD:
        blink_counter += 1
    else:
        # If the EAR is above the threshold, reset the blink counter
        blink_counter = 0
        blink_detected = False  # Reset the blink detection flag when eyes open

    # If the blink counter exceeds the required consecutive frames, mark blink detected
    if blink_counter >= CONSECUTIVE_FRAMES:
        blink_detected = True

    return blink_detected

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return None, False

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * 3)  # Process the first 3 seconds of the video
    print(f"Frames per second: {fps}, Max frames: {max_frames}")

    # Scaling factor for resizing the frames
    scale_percent = 20  # Resize to 20% of the original size for faster processing

    frame_count = 0
    stop_processing = False
    recognized_name = None

    while frame_count < max_frames and not stop_processing:
        ret, img = cap.read()
        if not ret:
            print(f"Error: Couldn't read frame from the video.")
            break

        frame_count += 1

        # Resize the frame while maintaining the aspect ratio
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # Resize the image
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)

        if len(face_locations) == 0:
            continue

        if frame_count > 15:
            for (top, right, bottom, left) in face_locations:
                gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                rect = dlib.rectangle(left, top, right, bottom)
                shape = predictor(gray_img, rect)
                landmarks = np.array([(p.x, p.y) for p in shape.parts()])

                # Check for blinking
                if detect_face_liveness(landmarks):
                    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                    for face_encoding in face_encodings:
                        name = "Unknown"
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        if True in matches:
                            first_match_index = matches.index(True)
                            name = known_face_names[first_match_index]
                            recognized_name = name
                            stop_processing = True
                            break

    cap.release()
    return recognized_name, blink_detected
