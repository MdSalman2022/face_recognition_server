import cv2
import face_recognition
import numpy as np
import pickle
from fastapi import HTTPException

# Load classifier and label encoder
try:
    with open('app/augmented_face_recognition_model.pkl', 'rb') as f:
        clf, le = pickle.load(f)
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure the model is trained and saved.")

def process_image(image_path: str):
    """
    Process a single image for face recognition
    """
    try:
        # Load and process the image
        img = face_recognition.load_image_file(image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read the image.")

        # Resize the image while maintaining aspect ratio
        scale_percent = 50
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        rgb_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if len(face_encodings) == 0:
            return {"status": "error", "message": "No faces found in the image."}

        # Process each face found
        studentId = ""
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Predict using the classifier
            pred_label = clf.predict([face_encoding])[0]
            pred_name = le.inverse_transform([pred_label])[0]
            studentId = pred_name 
            print(f"Predicted: {pred_name}")

        return {
            "status": "success",
            "studentId": studentId
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")