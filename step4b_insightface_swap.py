import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Setup
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load swapper model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)

# Load images
model_img = cv2.imread('model.jpg')
user_img = cv2.imread('output/user_full.jpg')

# Detect faces
model_faces = app.get(model_img)
user_faces = app.get(user_img)

if len(model_faces) == 0:
    print("❌ No face in model image")
    exit()

if len(user_faces) == 0:
    print("❌ No face in your photo")
    exit()

print("✅ Faces detected in both images")

# Swap
result = model_img.copy()
result = swapper.get(result, model_faces[0], user_faces[0], paste_back=True)

cv2.imwrite('output/face_swapped_v2.jpg', result)
print("✅ Done! Saved to output/face_swapped_v2.jpg")