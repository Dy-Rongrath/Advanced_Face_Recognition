import cv2
import face_recognition
import numpy as np
import os
import pickle
import threading
from datetime import datetime

# Configuration
SAVE_OUTPUT = True
output_dir = "./output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
OUTPUT_FILE = os.path.join(output_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
FRAME_RESIZE_SCALE = 1
ENCODINGS_FILE = 'known_faces_encodings.pkl'
SKIP_FRAMES = 2  # Increase to skip more frames

def save_encodings(encodings_file, known_encodings, known_names):
    with open(encodings_file, 'wb') as f:
        pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

def load_encodings(encodings_file):
    with open(encodings_file, 'rb') as f:
        data = pickle.load(f)
    return data['encodings'], data['names']

def load_known_faces(known_faces_dir, encodings_file, relearn=False):
    if os.path.exists(encodings_file) and not relearn:
        return load_encodings(encodings_file)
    known_encodings = []
    known_names = []
    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, file_name))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file_name)[0])
            else:
                print(f"No face detected in {file_name}")
    save_encodings(encodings_file, known_encodings, known_names)
    return known_encodings, known_names

def face_recognition_process(frame, known_encodings, known_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

def image_recognition(image_path, known_encodings, known_names):
    image = cv2.imread(image_path)
    image = face_recognition_process(image, known_encodings, known_names)
    cv2.imshow('Face Recognition', image)
    cv2.waitKey(0)  # Wait until any key is pressed
    cv2.destroyAllWindows()

def video_processing_thread(video_source, known_encodings, known_names, output_file, use_webcam=False):
    if use_webcam:
        video_source = 0  # Default webcam
    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        print("Error: Cannot open video source.")
        return
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * FRAME_RESIZE_SCALE)
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * FRAME_RESIZE_SCALE)
    video_writer = None
    frame_count = 0
    if SAVE_OUTPUT and not use_webcam:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame_count % SKIP_FRAMES == 0:  # Skip frames to reduce load
                frame = face_recognition_process(frame, known_encodings, known_names)
                cv2.imshow('Real-Time Face Recognition', frame)
                if SAVE_OUTPUT and video_writer:
                    video_writer.write(frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Responsive GUI interaction
                print("Stopping video processing...")
                break
    finally:
        video_capture.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    KNOWN_FACES_DIR = "./known_faces"
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: Directory '{KNOWN_FACES_DIR}' does not exist.")
        exit(1)
    relearn = input("Do you want to relearn the known faces? (yes/no): ").strip().lower() == 'yes'
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR, ENCODINGS_FILE, relearn)
    choice = input("Choose input method (image/video/webcam): ").strip().lower()
    if choice == 'video':
        video_source = input("Enter video file path: ").strip()
        threading.Thread(target=video_processing_thread, args=(video_source, known_encodings, known_names, OUTPUT_FILE)).start()
    elif choice == 'image':
        image_path = input("Enter image file path: ").strip()
        image_recognition(image_path, known_encodings, known_names)
    elif choice == 'webcam':
        threading.Thread(target=video_processing_thread, args=(None, known_encodings, known_names, OUTPUT_FILE, True)).start()
    else:
        print("Invalid input method selected.")
