import os
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO
import joblib
import numpy as np

load_dotenv()


def load_yolo_model(model_path):
    """
    Load YOLOv8 model from the specified path.
    """
    print(f"Loading YOLOv8 model from: {model_path}")
    model = YOLO(model_path)
    return model


def load_classifier(classifier_path_env):
    """
    Load the Random Forest classifier from the specified environment variable.

    Args:
        classifier_path_env (str): The name of the environment variable that stores the classifier path.

    Returns:
        classifier: The loaded Random Forest classifier.
    """
    classifier_path = os.getenv(classifier_path_env)
    if not classifier_path or not os.path.exists(classifier_path):
        print(f"Error: {classifier_path_env} not found or does not exist in environment variables.")
        exit(1)
    print(f"Loading classifier from: {classifier_path}")
    return joblib.load(classifier_path)


def classify_person(frame, box, pose_model, classifier):
    """
    Classify a detected person as 'Fine' or 'Need Help' using keypoints extracted by pose_model.
    """
    x1, y1, x2, y2 = map(int, box)

    # 提取边框区域
    person_patch = frame[y1:y2, x1:x2]
    if person_patch.size == 0:
        return "Unknown"

    # 调整大小
    person_patch = cv2.resize(person_patch, (224, 224))
    person_patch = person_patch.astype(np.uint8)

    # 使用 pose_model 提取关键点
    results = pose_model(person_patch)
    keypoints = results[0].keypoints
    if keypoints is not None:
        features = keypoints.xy.cpu().numpy().flatten()
        if len(features) < 34:
            features = np.pad(features, (0, 34 - len(features)), 'constant')
        else:
            features = features[:34]
    else:
        features = np.zeros(34)

    # 使用分类器预测
    prediction = classifier.predict([features])[0]
    return "Need Help" if prediction == 0 else "Fine"



def track_objects_with_yolo(frame, tracking_model, pose_model, classifier):
    """
    Detect and track objects in a frame using the tracking YOLO model.
    Classify persons using another YOLO model and a Random Forest classifier.
    """
    results = tracking_model(frame)  # 使用高精度模型进行检测
    boxes = results[0].boxes.xyxy.cpu().numpy()  # 检测框坐标
    confidences = results[0].boxes.conf.cpu().numpy()  # 置信度
    classes = results[0].boxes.cls.cpu().numpy()  # 类别索引

    for box, confidence, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{tracking_model.names[int(cls)]} {confidence:.2f}"

        # 使用 pose_model 提取关键点并分类
        prediction = classify_person(frame, box, pose_model, classifier)
        label += f" | {prediction}"  # 添加分类结果到标签

        # 根据类别选择颜色
        if "falling_person" in label:
            color = (0, 0, 255)  # 红色框
        elif "sitting_person" in label:
            color = (255, 255, 0)  # 青色框
        elif "lying_person" in label:
            color = (255, 0, 255)  # 洋红色框
        else:
            color = (0, 255, 0)  # 默认绿色

        # 绘制边框和标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame




def display_frame(frame, frame_idx, model, classifier):
    """
    Display the current frame with frame index and YOLO detections overlay.
    Args:
        frame (numpy.ndarray): The current video frame.
        frame_idx (int): The frame index.
        model (YOLO): The YOLO model used for detection.
        classifier: The Random Forest classifier.
    """
    frame = track_objects_with_yolo(frame, model, classifier)  # 传递 classifier
    cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)



def process_single_video(video_path, tracking_model, pose_model, classifier):
    """
    Process a single video file and display its frames with YOLO detections and classifications.
    Supports skipping frames with spacebar and quitting with 'q'.

    Args:
        video_path (str): Path to the video file.
        tracking_model (YOLO): YOLO model for detection and tracking.
        pose_model (YOLO): YOLO model for pose keypoint extraction.
        classifier: The Random Forest classifier for classification.
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

        # 进行目标检测和分类
        frame = track_objects_with_yolo(frame, tracking_model, pose_model, classifier)

        # 显示帧编号
        cv2.putText(frame, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)

        # 控制跳帧和退出
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格键跳到下一帧
                break
            elif key == ord('q'):  # 按 'q' 键退出
                cap.release()
                cv2.destroyAllWindows()
                return
        # # 控制跳帧和退出
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord(' '):  # 空格键跳到下一帧
        #     continue
        # elif key == ord('q'):  # 按 'q' 键退出
        #     break
    cap.release()


def process_videos(video_path, tracking_model, pose_model, classifier):
    """
    Process a list of video paths and display their frames with YOLO detections.
    Args:
        video_paths (list): List of video file paths.
        model (YOLO): The YOLO model used for detection.
        classifier: The Random Forest classifier.
    """
    for video_path in video_paths:
        process_single_video(video_path, tracking_model, pose_model, classifier)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 加载 YOLO 模型
    old_model_path = os.getenv("OLD_MODEL_PATH")  # best1.4.pt
    pose_model_path = os.getenv("YOLO_MODEL_PATH")  # yolov11n-pose

    if not old_model_path or not pose_model_path:
        print("Error: Model paths not found in environment variables.")
        exit(1)

    # 追踪模型 (best1.4.pt)
    tracking_model = load_yolo_model(old_model_path)

    # 分类用的姿态检测模型 (yolov11n-pose)
    pose_model = load_yolo_model(pose_model_path)

    # 加载分类器
    clf = load_classifier('CLASSIFIER_PATH')
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

    process_videos(video_paths, tracking_model, pose_model, clf)
