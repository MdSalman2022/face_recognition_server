import os
import shutil
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from app.controllers.face_detection_controller import process_image

router = APIRouter()

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017/")
db = client["face_recognition_db"]
collection = db["recognized_faces"]

@router.post("/detect/")
async def detect_faces(file: UploadFile = File(...)):
    # Check if the file is a valid image format
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an image file.")
    
    # Save the uploaded file temporarily
    image_file_path = f"./temp_{file.filename}"
    
    try:
        with open(image_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image
        result = process_image(image_file_path)
        
        # Save results to MongoDB if faces were detected
        if result["status"] == "success":
            collection.insert_one({
                "student_id": result["studentId"],
                "timestamp": datetime.utcnow(),
                "course_title": "Big Data and Iot Lab",
                "course_id": "CSE413"
            })

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up the temporary file
        if os.path.exists(image_file_path):
            os.remove(image_file_path)