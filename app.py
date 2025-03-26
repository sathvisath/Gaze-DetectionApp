from flask import Flask, render_template, Response
import cv2
import numpy as np
import threading

app = Flask(__name__)

# Load the pre-trained Haar Cascade Classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Variable to store the direction text
direction_text = "Center"

def detect_eyes_and_faces():
    global direction_text
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop over the faces detected
        for (x, y, w, h) in faces:
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Region of interest (ROI) for eyes detection
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) >= 2:
                # Calculate midpoint of eyes for direction
                ex1, ey1, ew1, eh1 = eyes[0]
                ex2, ey2, ew2, eh2 = eyes[1]

                eye_center_x1 = x + ex1 + ew1 // 2
                eye_center_y1 = y + ey1 + eh1 // 2
                eye_center_x2 = x + ex2 + ew2 // 2
                eye_center_y2 = y + ey2 + eh2 // 2

                eye_midpoint_x = (eye_center_x1 + eye_center_x2) // 2

                # Set the direction based on the eyes' position
                if eye_midpoint_x < face_center_x - 10:
                    direction_text = 'Left'
                elif eye_midpoint_x > face_center_x + 10:
                    direction_text = 'Right'
                else:
                    direction_text = 'Center'

        # Flip the frame horizontally to create a mirror effect
        frame = cv2.flip(frame, 1)

        # Encode the frame in JPEG format for serving via Flask
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield the frame as a response for the video stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_eyes_and_faces(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/direction')
def get_direction():
    return direction_text

if __name__ == '__main__':
    app.run(debug=True)
