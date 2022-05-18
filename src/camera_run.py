"""
run_camera.py
-------------
Script to run model with the connected camera
"""
import numpy as np
import cv2

def main():
    # Detect Faces
    # Check more cascades in the cv2 folder!!!
    face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontal_face_alt2.xml")

    # Video capture
    cap = cv2.VideCapture(0)

    while (True):
        # Capture frame-by-frame from camera
        ret, frame = cap.read()

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        # Process ROI
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]

            # Recognize face

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), [0,255,0], 2)

        # Display result
        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Close everything
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
