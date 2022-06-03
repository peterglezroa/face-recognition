"""
run_camera.py
-------------
Script to run model with the connected camera
"""
import time
import numpy as np
import cv2
from face_recognition import get_recognition_model, RECOGNITION_MODELS
from keras_vggface.vggface import VGGFace
from scipy.ndimage import zoom
from tensorflow.keras.models import Model

def main():
    # Detect Faces
    # Check more cascades in the cv2 folder!!!
    face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")

    # Video capture
    cap = cv2.VideoCapture(0)

    vggmodel = VGGFace(model="vgg16")
    conv_output = vggmodel.get_layer("fc7").output
    pred_output = vggmodel.get_layer("fc8/softmax").output
    vggmodel = Model(vggmodel.input, outputs=[conv_output, pred_output])
    print(vggmodel.summary())

    frame_time = time.time()
    while (True):
        # Capture frame-by-frame from camera
        ret, frame = cap.read()

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

        # Process ROI
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), [0,255,0], 2)

            if (time.time() - frame_time) >= 10:
                frame_time = time.time()
                roi = frame[y:y+h, x:x+w]
    #            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, [224,224], interpolation = cv2.INTER_AREA)
                roi = np.expand_dims(roi, axis=0).astype(np.float32)

                # Recognize face
                conv, pred = vggmodel.predict(roi)
                target = np.argmax(pred, axis=1).squeeze()
                w, b = vggmodel.get_layer("fc8/softmax").weights
                weights = w[:, target].numpy()
                heatmap = conv.squeeze() @ weights
                cv2.imshow("face", zoom(conv, zoom=[224,224]) )#, cmap="jet", alpha=0.5)

#            cv2.imshow("face", roi)

        # Display result
        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Close everything
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
