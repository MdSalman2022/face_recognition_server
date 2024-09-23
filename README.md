# Face Recognition Server

This project is a FastAPI-based backend for processing video streams to detect faces and blinking (for liveness detection). It uses a pre-trained model to recognize faces from video uploads or WebSocket streams.

## Features

- **Face Recognition:** Detects faces from videos using pre-trained encodings.
- **Blink Detection:** Detects if a person blinks to verify liveness.
- **WebSocket Support:** Real-time video stream processing.
- **File Upload Support:** Accepts video file uploads for face recognition and blinking detection.

## Prerequisites

- **Python 3.7 or later**
- **Git**
- **Virtual Environment (Optional but recommended)**

## RUN THE PROJECT

uvicorn app.main:app --reload

# face_recognition_server
