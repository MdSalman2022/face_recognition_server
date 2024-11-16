import os
import shutil
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from app.controllers.face_detection_controller import process_video, detect_face_liveness

router = APIRouter()

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition_db"]
collection = db["recognized_faces"]

@router.post("/detect/")
async def detect_faces(file: UploadFile = File(...)):
    # Check if the file is a valid video format
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
    
    # Save the uploaded file temporarily
    video_file_path = f"./temp_{file.filename}"
    
    with open(video_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the video to recognize faces
    try:
        recognized_name, blink_detected = process_video(video_file_path)
    finally:
        # Clean up the uploaded file
        os.remove(video_file_path)

    if recognized_name:
        # Save the result to MongoDB
        result = {"student_id": recognized_name, "blink_detected": blink_detected}
        collection.insert_one(result)
        return JSONResponse(content={"student": recognized_name})
    else:
        return JSONResponse(content={"error": "No face recognized or no blinking detected."}, status_code=400)

@router.websocket("/ws/stream/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            recognized_name = None
            blink_detected = False

            if len(face_locations) > 0:
                for (top, right, bottom, left) in face_locations:
                    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    rect = dlib.rectangle(left, top, right, bottom)
                    shape = predictor(gray_img, rect)
                    landmarks = np.array([(p.x, p.y) for p in shape.parts()])
                    if detect_face_liveness(landmarks):
                        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                        for face_encoding in face_encodings:
                            name = "Unknown"
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                            if True in matches:
                                first_match_index = matches.index(True)
                                name = known_face_names[first_match_index]
                                recognized_name = name
                                blink_detected = True

            if recognized_name:
                result = {"student_id": recognized_name, "blink_detected": blink_detected}
                collection.insert_one(result)
                await websocket.send_json({"student": recognized_name})
            else:
                await websocket.send_json({"error": "No face recognized or no blinking detected."})
    except WebSocketDisconnect:
        print("Client disconnected")