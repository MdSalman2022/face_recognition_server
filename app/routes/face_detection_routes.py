from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.controllers.face_detection_controller import process_video
import shutil
import os

router = APIRouter()

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
        return JSONResponse(content={"student": recognized_name})
    else:
        return JSONResponse(content={"error": "No face recognized or no blinking detected."}, status_code=400)
