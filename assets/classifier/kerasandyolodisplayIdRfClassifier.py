import os
import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from collections import deque
import joblib

# 加载环境变量
load_dotenv()

class SORT:
    def __init__(self):
        self.trackers = []
        self.track_id_count = 0

    def update(self, detections):
        """
        更新跟踪器，根据检测结果分配唯一 ID。
        """
        updated_tracks = []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            cls = detection[4]
            matched = False

            # 匹配现有跟踪器
            for tracker in self.trackers:
                if self.iou(tracker['bbox'], detection[:4]) > 0.3:
                    tracker['bbox'] = detection[:4]
                    tracker['hits'] += 1
                    tracker['misses'] = 0
                    tracker['cls'] = cls
                    updated_tracks.append(tracker)
                    matched = True
                    break

            # 如果未匹配，创建新的跟踪器
            if not matched:
                new_tracker = {
                    'id': self.track_id_count,
                    'bbox': detection[:4],
                    'hits': 1,
                    'misses': 0,
                    'cls': cls
                }
                self.track_id_count += 1
                updated_tracks.append(new_tracker)

        # 更新跟踪器列表
        self.trackers = [t for t in updated_tracks if t['misses'] < 3]

        return [[*tracker['bbox'], tracker['id'], tracker['cls']] for tracker in self.trackers]

    @staticmethod
    def iou(bbox1, bbox2):
        """
        计算两个边框的 IOU。
        """
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x4 - x3) * (y4 - y3)

        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

def load_yolo_model(model_path):
    print(f"Loading YOLOv8 model from: {model_path}")
    return YOLO(model_path)

def load_classifier(classifier_path_env):
    classifier_path = os.getenv(classifier_path_env)
    if not classifier_path or not os.path.exists(classifier_path):
        print(f"Error: {classifier_path_env} not found or does not exist in environment variables.")
        exit(1)
    print(f"Loading classifier from: {classifier_path}")
    return joblib.load(classifier_path)

def classify_person(frame, box, pose_model, classifier):
    x1, y1, x2, y2 = map(int, box)
    person_patch = frame[y1:y2, x1:x2]
    if person_patch.size == 0:
        return "Unknown"

    person_patch = cv2.resize(person_patch, (224, 224))
    person_patch = person_patch.astype(np.uint8)

    results = pose_model(person_patch)
    keypoints = results[0].keypoints
    if keypoints is not None:
        features = keypoints.xy.cpu().numpy().flatten()
        features = np.pad(features, (0, 34 - len(features)), 'constant') if len(features) < 34 else features[:34]
    else:
        features = np.zeros(34)

    prediction = classifier.predict([features])[0]
    return "Need Help" if prediction == 0 else "Fine"

def track_objects_with_yolo(frame, tracking_model, pose_model, classifier, mot_tracker):
    results = tracking_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    valid_detections = []
    for box, confidence, cls in zip(boxes, confidences, classes):
        if confidence > 0.5:
            valid_detections.append([*box, int(cls)])

    tracks = mot_tracker.update(valid_detections)

    for track in tracks:
        x1, y1, x2, y2, track_id, cls = map(int, track)
        label = f"ID: {track_id} | {tracking_model.names[cls]}"
        prediction = classify_person(frame, (x1, y1, x2, y2), pose_model, classifier)
        label += f" | {prediction}"

        color = (0, 0, 255) if "falling_person" in label else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def process_single_video(video_path, tracking_model, pose_model, classifier):
    cap = cv2.VideoCapture(video_path)
    mot_tracker = SORT()

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing video: {video_path}")
            break

        frame_idx += 1
        frame = track_objects_with_yolo(frame, tracking_model, pose_model, classifier, mot_tracker)
        cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格键跳到下一帧
                break
            elif key == ord('q'):  # 按 'q' 键退出
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

def process_videos(video_paths, tracking_model, pose_model, classifier):
    for video_path in video_paths:
        process_single_video(video_path, tracking_model, pose_model, classifier)

if __name__ == "__main__":
    old_model_path = os.getenv("OLD_MODEL_PATH")
    pose_model_path = os.getenv("YOLO_MODEL_PATH")
    video_paths_env = os.getenv("VIDEO_PATHS")

    if not old_model_path or not pose_model_path or not video_paths_env:
        print("Error: Required paths not set in environment variables.")
        exit(1)

    tracking_model = load_yolo_model(old_model_path)
    pose_model = load_yolo_model(pose_model_path)
    classifier = load_classifier("CLASSIFIER_PATH")

    video_paths = [path.strip() for path in video_paths_env.split(',')]
    process_videos(video_paths, tracking_model, pose_model, classifier)
