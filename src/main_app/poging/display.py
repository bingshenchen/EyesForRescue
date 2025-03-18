import os
import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from .config import *
from .tracker import SORT
from .classifier import PoseClassifier


class YOLODisplay:
    def __init__(self, tracking_model_path=None, enable_tracking=None, enable_classification=None):
        load_dotenv()

        self.enable_tracking = ENABLE_TRACKING if enable_tracking is None else enable_tracking
        self.enable_classification = ENABLE_CLASSIFICATION if enable_classification is None else enable_classification

        self.tracking_model_path = tracking_model_path or YOLO_MODEL_PATH

        self.tracking_model = self._load_tracking_model()

        self.tracker = SORT(
            max_miss=TRACKER_MAX_MISS,
            min_hits=TRACKER_MIN_HITS,
            iou_threshold=IOU_THRESHOLD
        ) if self.enable_tracking else None

        self.classifier = PoseClassifier() if self.enable_classification else None

        self.frame_idx = 0

    def _load_tracking_model(self):
        print(f"Loading yolomodel: {self.tracking_model_path}")
        return YOLO(self.tracking_model_path)

    def process_frame(self, frame):
        self.frame_idx += 1

        results = self.tracking_model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        valid_detections = []
        for box, confidence, cls in zip(boxes, confidences, classes):
            if confidence > CONFIDENCE_THRESHOLD:
                valid_detections.append([*box, int(cls)])

        if self.enable_tracking and self.tracker:
            tracks = self.tracker.update(valid_detections)
        else:
            tracks = [[*det[:4], -1, det[4]] for det in valid_detections]

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])

            track_id = int(track[4]) if self.enable_tracking else -1
            cls = int(track[5])

            class_name = self.tracking_model.names[cls]

            classification_result = ""
            if self.enable_classification and self.classifier and class_name in ["person", "falling_person",
                                                                                 "lying_person"]:
                prediction, confidence = self.classifier.classify_person(frame, (x1, y1, x2, y2))
                classification_result = f" | {prediction} ({confidence:.2f})"

            label = f"ID: {track_id}" if track_id >= 0 else ""
            label += f" | {class_name}" if label else class_name
            label += classification_result

            color = CLASS_COLORS.get(class_name, (255, 255, 255))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Frame: {self.frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

        return frame

    def process_video(self, video_path, output_path=None):
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
            cap = cv2.VideoCapture(int(video_path))
        else:
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Videos cant open {video_path}")
            return

        video_writer = None
        if output_path or SAVE_VIDEO:
            save_path = output_path or OUTPUT_VIDEO_PATH
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        self.frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)


            if DISPLAY_RESULTS:
                cv2.imshow("Frame", processed_frame)


            if video_writer:
                video_writer.write(processed_frame)


            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord(' '):
                        break
                    elif key == ord('q'):
                        break
            if key == ord('q'):
                break
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()