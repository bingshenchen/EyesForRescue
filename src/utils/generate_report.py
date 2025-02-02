import os
import pandas as pd
import logging
from dotenv import load_dotenv
from src.detection.fall_detection import detect_fall_in_video


def read_labels(label_dir):
    """
    Read all YOLO label files in a directory to count the total occurrences of class 1 (falls).

    Args:
        label_dir (str): Directory containing label files.

    Returns:
        int: Total count of labeled falls (class 1) in all label files.
    """
    truth = 0

    if not os.path.exists(label_dir):
        logging.warning(f"Label directory not found: {label_dir}")
        return truth

    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            file_path = os.path.join(label_dir, label_file)
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if parts and parts[0] == '1':  # Class '1' indicates a fall
                        truth += 1
            logging.info(f"Processed label file: {file_path}")

    return truth


def calculate_metrics(truth, found):
    """
    Calculate metrics like correct, false, missed, recall, and response time based on truth count and found data.

    Args:
        truth (int): Total count of ground truth falls.
        found (list): List of frames where falls were detected.

    Returns:
        dict: Dictionary containing 'correct', 'false', 'missed', 'recall', 'response_time'.
    """
    correct = min(truth, len(found))
    false = max(0, correct - truth)
    missed = max(0, truth - len(found))
    recall = correct / truth if truth else 0

    response_time = "N/A"

    return {
        'correct': correct,
        'false': false,
        'missed': missed,
        'recall': recall,
        'response_time': response_time
    }


def generate_batch_report(video_dir, model_path, classes, frame_skip=5):
    """
    Generate a batch report for fall detection over multiple videos with detailed metrics.

    Args:
        video_dir (str): Directory containing video files and subdirectories.
        model_path (str): Path to the YOLO model.
        classes (list): List of class names.
        frame_skip (int): Number of frames to skip between detections.

    Returns:
        list: A list of dictionaries containing the report data.
    """
    report_data = []
    index = 0
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    for root, dirs, files in os.walk(video_dir):
        for video_name in files:
            if any(video_name.endswith(ext) for ext in supported_extensions):
                video_path = os.path.join(root, video_name)
                logging.info(f"Processing video: {video_path}")

                # Detect falls in the video
                tracking_data = detect_fall_in_video(video_path, model_path, classes)
                if not tracking_data:
                    logging.warning(f"No tracking data found for {video_name}, skipping.")
                    continue

                # Extract detected frames
                found = [i for i, frame in enumerate(tracking_data) if sum(frame) > 0]

                # Read ground truth count from all label files in the same directory
                truth = read_labels(root)

                # Calculate metrics
                metrics = calculate_metrics(truth, found)

                # Record data for the report
                report_data.append({
                    'index': index,
                    'videoname': video_name,
                    'truth': truth,
                    'found': len(found),
                    'correct': metrics['correct'],
                    'false_positive': metrics['false'],
                    'missed': metrics['missed']
                })

                index += 1

    return report_data


def main():
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load paths and settings from environment variables
    model_path = os.getenv('YOLO_MODEL_PATH')
    video_dir = os.getenv('VIDEO_DIR')

    classes = os.getenv('CLASSES', "person,falling_person,sitting_person,lying_person").split(',')

    if not model_path or not video_dir:
        logging.error("Error: YOLO_MODEL_PATH or VIDEO_DIR not set in environment variables.")
        return

    output_dir = os.getenv('OUTPUT_DIR', 'reports')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate the report
    report = generate_batch_report(video_dir, model_path, classes, frame_skip=5)

    # Save the report to an Excel file
    report_df = pd.DataFrame(report)
    output_file = os.path.join(output_dir, "batch_report.xlsx")
    report_df.to_excel(output_file, index=False)
    logging.info(f"Report saved to {output_file}")


if __name__ == "__main__":
    main()
