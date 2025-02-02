import os
import cv2
import numpy as np
from joblib import load
from ultralytics import YOLO
from pathlib import Path

classifier_path = "classifier.pkl"
test_dir = r"C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications\assets\datasets\classifier\test"
fine_dir = os.path.join(test_dir, "fine")
needhelp_dir = os.path.join(test_dir, "needhelp")

print("Loading classifier...")
with open(classifier_path, 'rb') as f:
    classifier = load(f)

print("Classifier Loaded:", classifier)

print("Loading YOLO model...")
yolo_model = YOLO("yolo11n-pose.pt")
print("YOLO Model Loaded.")

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # 调整图片大小以适应 YOLO 模型
    img = img.astype(np.uint8)

    results = yolo_model(img)
    keypoints = results[0].keypoints

    if keypoints is not None:
        features = keypoints.xy.numpy().flatten()
    else:
        features = np.zeros(34)

    return features

def predict_folder(folder_path):
    predictions = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path) and image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            features = extract_features(image_path)
            features = np.pad(features, (0, 34 - len(features)), 'constant') if len(features) < 34 else features[:34]
            prediction = classifier.predict([features])
            predictions.append((image_name, "Need Help" if prediction[0] == 1 else "Fine"))
    return predictions

print("Predicting images in 'fine' folder...")
fine_predictions = predict_folder(fine_dir)

print("Predicting images in 'needhelp' folder...")
needhelp_predictions = predict_folder(needhelp_dir)

print("\nPredictions for 'fine' folder:")
for image_name, result in fine_predictions:
    print(f"{image_name}: {result}")

print("\nPredictions for 'needhelp' folder:")
for image_name, result in needhelp_predictions:
    print(f"{image_name}: {result}")
