import os
import cv2
import logging
from dotenv import load_dotenv
from ultralytics import YOLO


def load_classes(classes_path):
    """
    Load class names from a text file. If the file is not found or an error occurs, return default class names.

    Args:
        classes_path (str): Path to the file containing class names.

    Returns:
        list: A list of class names.
    """
    default_classes = ["person", "falling_person", "sitting_person", "lying_person"]
    try:
        with open(classes_path, 'r') as file:
            classes = [line.strip() for line in file.readlines()]
        return classes
    except Exception as e:
        logging.error(f"Error loading classes file: {e}. Using default classes.")
        return default_classes


def process_video_with_detection(video_path, output_dir, model_path, classes, conf_threshold=0.7, batch_size=16):
    """
    Use the YOLO model to detect objects in a video and save the output video with bounding boxes.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the output video with detection visualized.
        model_path (str): Path to the trained YOLO model (.pt file).
        classes (list): List of class names.
        conf_threshold (float): Confidence threshold for YOLO predictions (default: 0.7).
        batch_size (int): Number of frames to process in a batch to speed up detection (default: 16).
    """
    # Load the trained YOLO model
    model = YOLO(model_path)
    logging.info(f"Using model: {model_path}")

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Unable to open video file {video_path}")
        return

    logging.info(f"Processing video: {video_path}")

    # Get the video properties (width, height, FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Generate the output video file name based on input video name
    video_name = os.path.basename(video_path)
    output_video_name = f"tracked_{video_name}"
    output_video_path = os.path.join(output_dir, output_video_name)

    # Initialize the video writer to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # Colors for different classes

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.info("Reached the end of the video.")
            break

        frames.append(frame)
        if len(frames) == batch_size:
            # Run YOLO detection on the batch of frames
            results = model.predict(frames, conf=conf_threshold, stream=True)

            for i, frame_results in enumerate(results):
                for box in frame_results.boxes:
                    # Extract box coordinates, confidence, and class ID
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Get class name and color
                    class_name = classes[class_id] if class_id < len(classes) else "Unknown"
                    color = colors[class_id % len(colors)]

                    # Draw the bounding box and label
                    cv2.rectangle(frames[i], (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frames[i], f'{class_name}: {confidence:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Write the frame to the output video
                out.write(frames[i])

            frames = []

    # Process remaining frames
    if frames:
        results = model.predict(frames, conf=conf_threshold, stream=True)
        for i, frame_results in enumerate(results):
            for box in frame_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                class_name = classes[class_id] if class_id < len(classes) else "Unknown"
                color = colors[class_id % len(colors)]

                cv2.rectangle(frames[i], (x1, y1), (x2, y2), color, 2)
                cv2.putText(frames[i], f'{class_name}: {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frames[i])

    # Release video capture and writer
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"Processed video saved as {output_video_path}")


def main():
    """
    Main function to process video for fall detection visualization.
    """
    # Load environment variables
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get paths from environment variables
    video_path = os.getenv('TEST_VIDEO_PATH')
    model_path = os.getenv('YOLO_MODEL_PATH')
    classes_path = os.getenv('CLASSES_PATH', 'classes.txt')
    output_dir = os.getenv('OUTPUT_DIR', 'output_videos')

    if not video_path or not model_path:
        logging.error("Error: VIDEO_PATH or YOLO_MODEL_PATH not set in environment variables.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load class names
    classes = load_classes(classes_path)
    if not classes:
        logging.error("Error: Unable to load class names.")
        return

    logging.info(f"Input video path: {video_path}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Classes path: {classes_path}")
    logging.info(f"Output directory: {output_dir}")

    # Process video with YOLO model
    process_video_with_detection(video_path, output_dir, model_path, classes)


if __name__ == "__main__":
    main()
