import os
import cv2
import numpy as np
import joblib
from ultralytics import YOLO
from .config import POSE_MODEL_PATH, CLASSIFIER_PATH


class PoseClassifier:
    def __init__(self, pose_model_path=None, classifier_path=None):
        self.pose_model_path = pose_model_path or POSE_MODEL_PATH
        self.classifier_path = classifier_path or CLASSIFIER_PATH

        self.pose_model = self._load_pose_model()
        self.classifier = self._load_classifier()

    def _load_pose_model(self):
        if not os.path.exists(self.pose_model_path):
            raise FileNotFoundError(f"Pose model not fond: {self.pose_model_path}")

        print(f"Loading pose model: {self.pose_model_path}")
        return YOLO(self.pose_model_path)

    def _load_classifier(self):
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"Classifier not fond: {self.classifier_path}")

        print(f"Loading classifier: {self.classifier_path}")
        return joblib.load(self.classifier_path)

    def classify_person(self, frame, box):
        x1, y1, x2, y2 = map(int, box)

        person_patch = frame[y1:y2, x1:x2]
        if person_patch.size == 0:
            return "Unknown", 0.0

        person_patch = cv2.resize(person_patch, (224, 224))
        person_patch = person_patch.astype(np.uint8)

        results = self.pose_model(person_patch)
        keypoints = results[0].keypoints

        if keypoints is not None:
            features = keypoints.xy.cpu().numpy().flatten()
            features = np.pad(features, (0, 34 - len(features)), 'constant') if len(features) < 34 else features[:34]
        else:
            features = np.zeros(34)

        try:
            prediction = self.classifier.predict([features])[0]
            if hasattr(self.classifier, "predict_proba"):
                probabilities = self.classifier.predict_proba([features])[0]
                confidence = probabilities[prediction]
            else:
                confidence = 1.0

            return "Need Help" if prediction == 0 else "Fine", confidence
        except Exception as e:
            print(f"classifier error: {e}")
            return "Unknown", 0.0