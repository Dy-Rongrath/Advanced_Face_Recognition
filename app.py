from flask import Flask, request, jsonify , render_template
import face_recognition
import cv2
import numpy as np
from io import BytesIO
import base64
from advanced_face_recognition import load_encodings , face_recognition_process , load_student_data

ENCODINGS_FILE     = 'known_faces_encodings.pkl'
STUDENT_LIST_FILE  = "./Student-list.csv"

def draw_circle_on_frame(frame):
    height, width, _ = frame.shape
    center = (width // 2, height // 2)
    radius = min(width, height) // 10  # 10% of the smaller dimension
    color = (0, 255, 0)  # Green color
    thickness = 3
    
    cv2.circle(frame, center, radius, color, thickness)
    return frame

app = Flask(__name__)

# Load known faces and encodings
known_encodings, known_names = load_encodings(ENCODINGS_FILE)
student_data    = load_student_data(STUDENT_LIST_FILE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data['image']
        
        # Decode the base64 image string
        img_data = base64.b64decode(image_data.split(",")[1])
        img_arr = np.asarray(bytearray(img_data), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        # Process the frame for face recognition
        processed_frame, recognized_names = face_recognition_process(frame, known_encodings, known_names, student_data)

        print(process_frame , recognized_names)
        # Convert the processed frame back to base64 to send it back to the frontend
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': img_str,  # send the processed frame as base64
            'recognized_names': recognized_names
        })
    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/test_draw', methods=['POST'])
def test_draw():
    try:
        data = request.json
        image_data = data['image']
        
        # Decode the base64 image string
        img_data = base64.b64decode(image_data.split(",")[1])
        img_arr = np.asarray(bytearray(img_data), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        # Draw a circle in the middle of the frame
        processed_frame = draw_circle_on_frame(frame)
        
        # Convert the processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'image': img_str})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
