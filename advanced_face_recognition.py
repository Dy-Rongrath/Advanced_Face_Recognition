import cv2
import face_recognition
import os
import pickle
import threading
import numpy as np
from datetime import datetime
import logging
import csv

# Configuration
SAVE_OUTPUT        = True
STUDENT_LIST_FILE  = "./Student-list.csv"
KNOWN_FACES_DIR    = "./GIC_24-Images"
OUTPUT_DIR         = "./output"
OUTPUT_FILE        = os.path.join(OUTPUT_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
ENCODINGS_FILE     = 'known_faces_encodings.pkl'
FRAME_RESIZE_SCALE = 2   # Adjust for performance
SKIP_FRAMES        = 10  # Increase to skip more frames


# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_encodings(encodings_file, known_encodings, known_names):
    try:
        with open(encodings_file, 'wb') as f:
            pickle.dump({'encodings': known_encodings, 'names': known_names}, f)
        logging.info(f"Encodings saved to {encodings_file}")
    except Exception as e:
        logging.error(f"Error saving encodings: {e}")

def load_encodings(encodings_file):
    try:
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
        return data['encodings'], data['names']
    except Exception as e:
        logging.error(f"Error loading encodings: {e}")
        return [], []

def load_student_data(student_list_file):
    student_data = {}
    try:
        with open(student_list_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                student_data[row['ID-Card']] = {
                    'Name': row['Student Name'],
                    'Department': row['Department-Code'],
                    'Year': row['Year'],
                    'Semester': row['Semester'],
                    'Group': row['Group']
                }
        logging.info(f"Student data loaded from {student_list_file}")
    except Exception as e:
        logging.error(f"Error loading student data: {e}")
    return student_data

def load_known_faces(known_faces_dir, encodings_file, relearn=False):
    if os.path.exists(encodings_file) and not relearn:
        return load_encodings(encodings_file)
    
    known_encodings = []
    known_names = []

    for file_name in os.listdir(known_faces_dir):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_faces_dir, file_name)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.splitext(file_name)[0])
                else:
                    logging.warning(f"No face detected in {file_name}")
            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")

    save_encodings(encodings_file, known_encodings, known_names)
    return known_encodings, known_names

def face_recognition_process(frame, known_encodings, known_names, student_data):
    
    rgb_frame      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    names          = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        distances        = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances) if distances.size > 0 else None
        name             = "Unknown"
        confidence       = 0

        if best_match_index is not None:
            confidence = 1 - distances[best_match_index]  # Convert distance to confidence
            if confidence >= 0.6:  # Threshold for a valid match
                student_id = known_names[best_match_index]
                if student_id in student_data:
                    student = student_data[student_id]
                    name = f"{student_id}, {student['Name']}, {student['Department']}, {confidence * 100:.2f}%"
                else:
                    name = f"{student_id}, Unknown, Unknown, {confidence * 100:.2f}%"

        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        font           = cv2.FONT_HERSHEY_SIMPLEX
        font_scale     = 0.7
        font_thickness = 2
        text_size      = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
        text_color     = (255 , 0, 0)
        text_x         = left + (right - left - text_size[0]) // 2
        text_y         = bottom + text_size[1] + 10

        # Draw text
        cv2.putText(frame, name, (text_x, text_y), font, font_scale, text_color, font_thickness)
        names.append(name)

    return frame, names

def image_recognition(image_path, known_encodings, known_names, student_data):
    if not os.path.exists(image_path):
        logging.error(f"Image file '{image_path}' does not exist.")
        return

    image = cv2.imread(image_path)
    if image is None:
        logging.error("Image could not be loaded.")
        return

    processed_image, detected_names = face_recognition_process(image, known_encodings, known_names, student_data)
    cv2.imshow('Face Recognition', processed_image)

    if detected_names:
        output_image_path = os.path.join(OUTPUT_DIR, f"detected_{os.path.basename(image_path)}")
        cv2.imwrite(output_image_path, processed_image)
        logging.info(f"Detected faces saved to {output_image_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_processing_thread(video_source, known_encodings, known_names, student_data, output_file, use_webcam=False):
    if use_webcam:
        video_source = 0  # Default webcam

    if not use_webcam and not os.path.exists(video_source):
        logging.error(f"Video file '{video_source}' does not exist.")
        return

    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        logging.error("Cannot open video source.")
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
                frame, _ = face_recognition_process(frame, known_encodings, known_names, student_data)
                cv2.imshow('Real-Time Face Recognition', frame)

                if SAVE_OUTPUT and video_writer:
                    video_writer.write(frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Stopping video processing...")
                break

    finally:
        video_capture.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    if not os.path.exists(KNOWN_FACES_DIR):
        logging.error(f"Directory '{KNOWN_FACES_DIR}' does not exist.")
        exit(1)

    ensure_dir(OUTPUT_DIR)

    # Load student data from CSV
    student_data = load_student_data(STUDENT_LIST_FILE)

    relearn = input("Do you want to relearn the known faces? (yes/no): ").strip().lower() == 'yes'
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR, ENCODINGS_FILE, relearn)
    
    choice = input("Choose input method (image/video/webcam): ").strip().lower()

    if choice == 'video':
        video_source = input("Enter video file path: ").strip()
        threading.Thread(target=video_processing_thread, args=(video_source, known_encodings, known_names, student_data, OUTPUT_FILE)).start()

    elif choice == 'image':
        image_path = input("Enter image file path: ").strip()
        image_recognition(image_path, known_encodings, known_names, student_data)

    elif choice == 'webcam':
        threading.Thread(target=video_processing_thread, args=(None, known_encodings, known_names, student_data, OUTPUT_FILE, True)).start()

    else:
        logging.error("Invalid input method selected.")