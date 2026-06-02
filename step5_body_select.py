import cv2
import insightface
from insightface.app import FaceAnalysis
import os

# Setup insightface
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False)

# User selections - change these to test
BODY_TYPE = "fat"      # slim / athletic / average / fat
HEIGHT = "medium"      # short / medium / tall
SKIN_TONE = 4          # 1=very light, 2=light, 3=medium, 4=olive, 5=brown, 6=dark

# Skin tone color map (BGR)
skin_tone_map = {
    1: (220, 200, 190),
    2: (195, 170, 150),
    3: (170, 130, 100),
    4: (140, 100, 70),
    5: (110, 75, 50),
    6: (70, 45, 30),
}

# Load model image
model_path = f"models/{BODY_TYPE}_{HEIGHT}.jpg"
if not os.path.exists(model_path):
    print(f"❌ Model image not found: {model_path}")
    exit()

model_img = cv2.imread(model_path)
user_img = cv2.imread("output/user_full.jpg")

if model_img is None:
    print("❌ Cannot load model image")
    exit()

if user_img is None:
    print("❌ Cannot load user photo")
    exit()

# Detect faces
model_faces = app.get(model_img)
user_faces = app.get(user_img)

if len(model_faces) == 0:
    print("❌ No face detected in model image - try a different photo")
    exit()

if len(user_faces) == 0:
    print("❌ No face detected in your photo")
    exit()

print("✅ Faces detected in both images")

# Swap face
result = model_img.copy()
result = swapper.get(result, model_faces[0], user_faces[0], paste_back=True)

# Apply skin tone to neck/body area
skin_color = skin_tone_map[SKIN_TONE]
hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
lower = (0, 30, 60)
upper = (25, 180, 255)
import numpy as np
mask = cv2.inRange(hsv, lower, upper)
mask = cv2.dilate(mask, None, iterations=2)
colored = result.copy()
colored[mask > 0] = skin_color
result = cv2.addWeighted(result, 0.6, colored, 0.4, 0)

cv2.imwrite("output/final_result.jpg", result)
print(f"✅ Done! Body: {BODY_TYPE} {HEIGHT} | Skin tone: {SKIN_TONE}")
print("✅ Saved to output/final_result.jpg")