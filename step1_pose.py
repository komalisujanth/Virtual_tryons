import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp

# Download pose model if not exists
model_path = "pose_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        model_path
    )
    print("✅ Model downloaded")

# Load image
image = mp.Image.create_from_file("model.jpg")

# Detect pose
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(image)

    if result.pose_landmarks:
        print(f"✅ Body detected! Found {len(result.pose_landmarks)} person(s)")

        img_cv = cv2.imread("model.jpg")
        for person in result.pose_landmarks:
            for landmark in person:
                h, w = img_cv.shape[:2]
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img_cv, (x, y), 5, (0, 255, 0), -1)

        cv2.imwrite("output/pose_detected.jpg", img_cv)
        print("✅ Saved to output/pose_detected.jpg")
    else:
        print("❌ No body detected")