import os
import cv2
import logging
from dotenv import load_dotenv
from src.utils import calculate_danger
from ultralytics import YOLO


def detect_fall_in_video(video_path, model_path, classes, conf_threshold=0.7, batch_size=16):
    """
    Detect falls in the given video using a YOLO model and return tracking data.

    Args:
        video_path (str): Path to the video file.
        model_path (str): Path to the trained YOLO model (.pt file).
        classes (list): List of class names.
        conf_threshold (float): Confidence threshold for YOLO predictions (default: 0.7).
        batch_size (int): Number of frames to process in a batch to speed up detection (default: 16).

    Returns:
        list: A list of lists containing fall detection results for each frame.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        return []

    logging.info(f"Processing video: {video_path}")
    results_list = []  # List to store detection results for each frame
    frames = []

    # Create a mapping from class names to integer IDs
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.info("Reached the end of the video.")
            break  # Video ends

        frames.append(frame)
        if len(frames) == batch_size:
            # Detect objects in the batch of frames using YOLO
            batch_results = model.predict(frames, conf=conf_threshold, stream=True)

            for frame_results in batch_results:
                detections = []
                for box in frame_results.boxes:
                    class_id = int(box.cls[0])
                    class_name = classes[class_id] if class_id < len(classes) else "Unknown"
                    detections.append(class_to_id.get(class_name, 0))  # Default to 0 if class not found
                results_list.append(detections)

            frames = []  # Clear the batch

    # Process remaining frames
    if frames:
        batch_results = model.predict(frames, conf=conf_threshold, stream=True)
        for frame_results in batch_results:
            detections = []
            for box in frame_results.boxes:
                class_id = int(box.cls[0])
                class_name = classes[class_id] if class_id < len(classes) else "Unknown"
                detections.append(class_to_id.get(class_name, 0))  # Default to 0 if class not found
            results_list.append(detections)

    # Release video capture
    cap.release()
    cv2.destroyAllWindows()

    return results_list


def main():
    """
    Main function to detect falls in a video and calculate the danger score.
    """
    # Load environment variables
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get paths from environment variables
    model_path = os.getenv('YOLO_MODEL_PATH')
    video_path = os.getenv('TEST_VIDEO_PATH')
    classes_path = os.getenv('CLASSES_PATH', 'classes.txt')

    if not model_path or not video_path:
        logging.error("Error: YOLO_MODEL_PATH or VIDEO_PATH not set in environment variables.")
        return

    # Load class names
    try:
        with open(classes_path, 'r') as file:
            classes = [line.strip() for line in file.readlines()]
    except Exception as e:
        logging.error(f"Error loading classes file: {e}. Using default classes.")
        classes = ["person", "falling_person", "sitting_person", "lying_person"]

    logging.info(f"Using model: {model_path}")
    logging.info(f"Processing video: {video_path}")

    # Detect falls in the video
    tracking_data = detect_fall_in_video(video_path, model_path, classes)

    # Print the tracking data (e.g., detected classes)
    logging.info(f"Tracking data: {tracking_data}")

    # Calculate danger score based on fall detection
    danger_score = calculate_danger.calculate_danger(tracking_data)
    logging.info(f"Danger score: {danger_score}")


if __name__ == "__main__":
    main()
