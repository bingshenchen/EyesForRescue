import cv2
import numpy as np
from ultralytics import YOLO
from joblib import load


classifier_path = "classifier.pkl"
with open(classifier_path, 'rb') as f:
    classifier = load(f)
print("Classifier Loaded:", classifier)


yolo_model = YOLO("yolo11n-pose.pt")
print("YOLO Model Loaded.")

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

        if detections is not None:
            for i, box in enumerate(detections):
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                label = "Unknown"

                if keypoints_list is not None and i < len(keypoints_list):

                    features = extract_features_from_keypoints(keypoints_list[i])
                    features = np.pad(features, (0, 34 - len(features)), 'constant') if len(features) < 34 else features[:34]

                    prediction = classifier.predict([features])
                    label = "Need Help" if prediction[0] == 0 else "Fine"

                color = (0, 255, 0) if label == "Fine" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        cv2.imshow("Video Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


video_path = r"C:\Users\Bingshen\Videos\train\train\fall.mp4"
process_video(video_path)
