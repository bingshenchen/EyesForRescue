import cv2
import numpy as np
from ultralytics import YOLO
from joblib import load
import time

# Load the classifier
classifier_path = "classifier.pkl"
with open(classifier_path, 'rb') as f:
    classifier = load(f)
print("Classifier Loaded:", classifier)

# Load the YOLO model
yolo_model = YOLO("yolo11n-pose.pt")
print("YOLO Model Loaded.")

# Dictionary to store the duration of the "Need Help" state
help_durations = {}

def extract_features_from_keypoints(keypoints):
    if keypoints is not None:
        features = keypoints.xy.numpy().flatten()
    else:
        features = np.zeros(34)
    return features

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        detections = results[0].boxes
        keypoints_list = results[0].keypoints

        current_time = time.time()

        if detections is not None:
            for i, box in enumerate(detections):
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                label = "Unknown"
                bbox_id = f"{x1}_{y1}_{x2}_{y2}"  # Unique ID for the bounding box

                if keypoints_list is not None and i < len(keypoints_list):
                    features = extract_features_from_keypoints(keypoints_list[i])
                    features = np.pad(features, (0, 34 - len(features)), 'constant') if len(features) < 34 else features[:34]

                    prediction = classifier.predict([features])
                    label = "Need Help" if prediction[0] == 0 else "Fine"

                # Manage the help duration
                if label == "Need Help":
                    if bbox_id not in help_durations:
                        help_durations[bbox_id] = current_time  # Start timestamp
                elif label == "Fine":
                    if bbox_id in help_durations:
                        elapsed_time = current_time - help_durations[bbox_id]
                        if elapsed_time > 3:  # Remove if duration exceeds threshold
                            del help_durations[bbox_id]

                # Display bounding box and label
                if bbox_id in help_durations:
                    elapsed_time = current_time - help_durations[bbox_id]
                    label = f"Need Help ({int(elapsed_time)}s)"
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                    label = "Fine"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Video Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

video_path = r"C:\Users\Bingshen\Videos\train\train\fall.mp4"
process_video(video_path)
