import cv2
import face_recognition
import os
import pickle
import threading
import numpy as np
from datetime import datetime

# Configuration
SAVE_OUTPUT        = True
OUTPUT_DIR         = "./output"
OUTPUT_FILE        = os.path.join(OUTPUT_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
ENCODINGS_FILE     = 'known_faces_encodings.pkl'
FRAME_RESIZE_SCALE = 0.5  # Adjust for performance
SKIP_FRAMES        = 2  # Increase to skip more frames

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

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
    known_names     = []

    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_faces_dir, file_name)
            try:
                image     = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(file_name)[0])
                else:
                    print(f"No face detected in {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    save_encodings(encodings_file, known_encodings, known_names)

    return known_encodings, known_names

def face_recognition_process(frame, known_encodings, known_names):

    rgb_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    names          = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances) if distances.size > 0 else None
        name = "Unknown"

        if best_match_index is not None and distances[best_match_index] < 0.6:
            name = known_names[best_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        names.append(name)

    return frame, names

def image_recognition(image_path, known_encodings, known_names):

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' does not exist.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be loaded.")
        return

    processed_image, detected_names = face_recognition_process(image, known_encodings, known_names)
    cv2.imshow('Face Recognition', processed_image)

    if detected_names:
        output_image_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, processed_image)
        print(f"Detected faces saved to {output_image_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_processing_thread(video_source, known_encodings, known_names, output_file, use_webcam=False):

    if use_webcam:
        video_source = 0  # Default webcam

    if not use_webcam and not os.path.exists(video_source):
        print(f"Error: Video file '{video_source}' does not exist.")
        return

    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        print("Error: Cannot open video source.")
        return
    
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * FRAME_RESIZE_SCALE)
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * FRAME_RESIZE_SCALE)
    video_writer = None

    if SAVE_OUTPUT and not use_webcam:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    try:
        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)

            if frame_count % SKIP_FRAMES == 0:
                frame, _ = face_recognition_process(frame, known_encodings, known_names)
                cv2.imshow('Real-Time Face Recognition', frame)
                if SAVE_OUTPUT and video_writer:
                    video_writer.write(frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
