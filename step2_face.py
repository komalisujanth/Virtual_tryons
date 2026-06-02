import cv2
import os

# Load image
img = cv2.imread("model.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use OpenCV built-in face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

if len(faces) > 0:
    print(f"✅ Face detected!")
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        face_crop = img[y:y+h, x:x+w]
        cv2.imwrite("output/face_detected.jpg", img)
        cv2.imwrite("output/face_crop.jpg", face_crop)
    print("✅ Saved to output/face_detected.jpg")
else:
    print("❌ No face detected")