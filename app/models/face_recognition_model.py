# face_recognition_model.py
import pickle
import os

# Load the face recognition model
def load_model(filename='app/augmented_face_recognition_model.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError(f"Model file '{filename}' not found.")
