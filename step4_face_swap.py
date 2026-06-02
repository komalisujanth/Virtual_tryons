import cv2
import numpy as np

def swap_face(source_face, target_img, target_face_coords):
    x, y, w, h = target_face_coords
    
    # Resize source face to match target face size
    source_resized = cv2.resize(source_face, (w, h))
    
    # Create mask for seamless blending
    mask = 255 * np.ones(source_resized.shape, source_resized.dtype)
    
    # Center point of target face
    center = (x + w//2, y + h//2)
    
    # Seamless clone
    result = cv2.seamlessClone(source_resized, target_img, mask, center, cv2.NORMAL_CLONE)
    
    return result

# Load images
model_img = cv2.imread("model.jpg")
user_face = cv2.imread("output/user_face.jpg")

if model_img is None:
    print("❌ Cannot load model.jpg")
    exit()

if user_face is None:
    print("❌ Cannot load user_face.jpg")
    exit()

# Detect face in model image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(model_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

if len(faces) == 0:
    print("❌ No face detected in model image")
    exit()

print(f"✅ Found face in model image")

# Swap face
result = swap_face(user_face, model_img, faces[0])

cv2.imwrite("output/face_swapped.jpg", result)
print("✅ Face swap done! Saved to output/face_swapped.jpg")