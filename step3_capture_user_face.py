import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("📷 Webcam opening - look at the camera and press SPACE to capture, Q to quit")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot access webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, "Face found! Press SPACE to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Capture Your Face", frame)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_crop = frame[y:y+h, x:x+w]
            cv2.imwrite("output/user_face.jpg", face_crop)
            cv2.imwrite("output/user_full.jpg", frame)
            print("✅ Your face captured and saved!")
            break
        else:
            print("⚠️ No face detected yet, keep trying")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()