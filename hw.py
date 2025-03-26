import cv2
import numpy as np
import math

# Load the pre-trained Haar Cascade Classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale for better detection accuracy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better visibility in low light
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)  # Apply CLAHE on the grayscale image
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop over the faces detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Find the center of the face
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        
        # Region of interest (ROI) for eyes detection (only detect eyes within the face region)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # If at least two eyes are detected, calculate the midpoint between them
        if len(eyes) >= 2:
            # Assume the first two eyes are the left and right eyes
            ex1, ey1, ew1, eh1 = eyes[0]  # Left eye
            ex2, ey2, ew2, eh2 = eyes[1]  # Right eye
            
            # Find the center of both eyes
            eye_center_x1 = x + ex1 + ew1 // 2
            eye_center_y1 = y + ey1 + eh1 // 2
            eye_center_x2 = x + ex2 + ew2 // 2
            eye_center_y2 = y + ey2 + eh2 // 2
            
            # Calculate the midpoint between the eyes
            eye_midpoint_x = (eye_center_x1 + eye_center_x2) // 2
            eye_midpoint_y = (eye_center_y1 + eye_center_y2) // 2
            
            # Draw circles for both eye centers and the midpoint
            cv2.circle(frame, (eye_center_x1, eye_center_y1), 5, (0, 255, 0), -1)  # Left eye center
            cv2.circle(frame, (eye_center_x2, eye_center_y2), 5, (0, 255, 0), -1)  # Right eye center
            cv2.circle(frame, (eye_midpoint_x, eye_midpoint_y), 5, (0, 255, 255), -1)  # Midpoint

            # Calculate the direction based on the eye midpoint relative to the face center
            if eye_midpoint_x < face_center_x - 10:  # Left of center
                direction = 'Left'
            elif eye_midpoint_x > face_center_x + 10:  # Right of center
                direction = 'Right'
            else:  # Close to the center
                direction = 'Center'
            
            # Display the direction text on the image
            cv2.putText(frame, f"Direction: {direction}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # If only one eye is detected, use that eye for direction detection
        elif len(eyes) == 1:
            ex, ey, ew, eh = eyes[0]
            eye_center_x = x + ex + ew // 2
            eye_center_y = y + ey + eh // 2
            
            # Draw a circle for the single eye center
            cv2.circle(frame, (eye_center_x, eye_center_y), 5, (0, 255, 0), -1)  # Single eye center
            
            # Calculate the direction based on the eye's position
            if eye_center_x < face_center_x - 10:  # Left of center
                direction = 'Left'
            elif eye_center_x > face_center_x + 10:  # Right of center
                direction = 'Right'
            else:  # Close to the center
                direction = 'Center'
            
            # Display the direction text on the image
            cv2.putText(frame, f"Direction: {direction}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Display the center of the face as a point
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 0, 255), -1)
    
    # Display the output frame
    cv2.imshow('Eye Direction Detection', frame)
    
    # Exit the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
