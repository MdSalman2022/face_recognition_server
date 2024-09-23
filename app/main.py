from fastapi import FastAPI
from app.routes.face_detection_routes import router as face_detection_router

app = FastAPI()

# Include the face detection routes
app.include_router(face_detection_router)

# Root route to check if the API is running
@app.get("/")


async def root():
    return {"message": "Face detection API is running!"}
