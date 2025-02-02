import os
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()


def load_yolo_model(model_path):
    """
    Load YOLOv8 model from the specified path.
    """
    print(f"Loading YOLOv8 model from: {model_path}")
    model = YOLO(model_path)
    return model


def track_objects_with_yolo(frame, model):
    """
    Detect and track objects in a frame using YOLOv8.
    Adds bounding boxes and labels for detected objects on the frame.
    """
    results = model(frame)
    boxes = results[0].boxes  # YOLOv8的检测结果
    detections = boxes.xyxy.cpu().numpy()  # 检测框坐标
    confidences = boxes.conf.cpu().numpy()  # 置信度
    classes = boxes.cls.cpu().numpy()  # 类别索引

    for box, confidence, cls in zip(detections, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {confidence:.2f}"
        color = (0, 255, 0)  # 默认绿色
        if "falling_person" in label:
            color = (0, 0, 255)  # 红色框
        elif "sitting_person" in label:
            color = (255, 255, 0)  # 青色框
        elif "lying_person" in label:
            color = (255, 0, 255)  # 洋红色框

        # 绘制边框和标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def display_frame(frame, frame_idx, model):
    """
    Display the current frame with frame index and YOLO detections overlay.
    """
    frame = track_objects_with_yolo(frame, model)
    cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)


def process_single_video(video_path, model):
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
        display_frame(frame, frame_idx, model)

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
    for video_path in video_paths:
        process_single_video(video_path, model)

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