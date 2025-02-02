import os
import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
from collections import deque

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
                    tracker['cls'] = cls  # 更新类别信息
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
                    'cls': cls  # 新增类别字段
                }
                self.track_id_count += 1
                updated_tracks.append(new_tracker)

        # 更新跟踪器列表
        self.trackers = [t for t in updated_tracks if t['misses'] < 3]

        # 返回简化版本的跟踪器信息，用于显示
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
    """
    Load YOLOv8 model from the specified path.
    """
    print(f"Loading YOLOv8 model from: {model_path}")
    model = YOLO(model_path)
    return model


def track_objects_with_yolo(frame, model, mot_tracker):
    """
    Detect and track objects in a frame using YOLOv8 and SORT.
    Adds bounding boxes, labels, and unique IDs for tracked objects.
    """
    results = model(frame)
    boxes = results[0].boxes  # YOLOv8的检测结果
    detections = boxes.xyxy.cpu().numpy()  # 检测框坐标
    confidences = boxes.conf.cpu().numpy()  # 置信度
    classes = boxes.cls.cpu().numpy()  # 类别索引

    # 仅保留置信度高于阈值的检测
    valid_detections = []
    for box, confidence, cls in zip(detections, confidences, classes):
        if confidence > 0.5:  # 过滤低置信度的框
            valid_detections.append([*box, int(cls)])  # 包括 cls

    # 更新跟踪器并获取带ID的跟踪结果
    tracks = mot_tracker.update(valid_detections)  # 返回带ID的跟踪器信息

    for track in tracks:
        x1, y1, x2, y2, track_id, cls = map(int, track)
        label = f"ID: {track_id} | {model.names[cls]}"

        # 根据类别选择颜色
        if model.names[cls] == "falling_person":  # 直接匹配类别名称
            color = (0, 0, 255)  # 红色框用于 falling_person
        elif model.names[cls] == "sitting_person":
            color = (255, 255, 0)  # 青色框用于 sitting_person
        elif model.names[cls] == "lying_person":
            color = (255, 0, 255)  # 洋红色框用于 lying_person
        else:
            color = (0, 255, 0)  # 默认绿色框

        # 绘制边框和标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def display_frame(frame, frame_idx, model, mot_tracker):
    """
    Display the current frame with frame index and YOLO detections overlay.
    """
    frame = track_objects_with_yolo(frame, model, mot_tracker)
    cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)


def process_single_video(video_path, model, mot_tracker):
    """
    Process a single video file and display its frames with YOLO detections.
    """
    video_name = os.path.basename(video_path)
    print(f"Processing video: {video_name}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video {video_name}. Skipping...")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Finished processing video: {video_name}")
            break

        frame_idx += 1
        display_frame(frame, frame_idx, model, mot_tracker)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格键跳到下一帧
                break
            elif key == ord('q'):  # 按 'q' 键退出
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()


def process_videos(video_paths, model):
    """
    Process a list of video paths and display their frames with YOLO detections.
    """
    mot_tracker = SORT()  # 创建多目标跟踪器
    for video_path in video_paths:
        process_single_video(video_path, model, mot_tracker)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 加载 YOLO 模型
    old_model_path = os.getenv("OLD_MODEL_PATH")
    if not old_model_path:
        print("Error: OLD_MODEL_PATH not found in environment variables.")
        exit(1)
    model = load_yolo_model(old_model_path)

    # 加载视频路径
    video_paths = os.getenv("VIDEO_PATHS")
    if not video_paths:
        print("Error: VIDEO_PATHS not found in environment variables.")
        exit(1)

    # 验证视频路径
    video_paths = [path.strip() for path in video_paths.split(',')]
    for path in video_paths:
        if not os.path.exists(path):
            print(f"Error: Video path does not exist: {path}")
            exit(1)

    print("Loaded video paths:")
    print("\n".join(video_paths))

    process_videos(video_paths, model)
