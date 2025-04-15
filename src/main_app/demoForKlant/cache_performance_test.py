import os
import sys
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Add project root to path
project_root = os.getenv('PROJECT_ROOT')
if project_root:
    sys.path.append(project_root)
else:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

from src.main_app.utils.cache_manager import DetectionCache
from src.main_app.utils.performance_analyzer import PerformanceAnalyzer
from ultralytics import YOLO
import cv2


def process_video(video_path, model, use_cache=False):
    """
    Process a video with YOLO detection, with or without cache.

    Args:
        video_path: Path to the video file
        model: The YOLO model to use
        use_cache: Whether to use cache

    Returns:
        total_frames: Total number of frames processed
        total_time: Total processing time
        detection_time: Time spent on YOLO detection
    """
    try:
        cache_manager = DetectionCache()
        # Handle the case where model.ckpt_path might not exist
        try:
            model_name = os.path.basename(model.ckpt_path)
        except AttributeError:
            model_name = "yolo_model"
            print(f"Warning: Using default model name: {model_name}")

        # Variables for performance measurement
        total_frames = 0
        detection_time = 0

        # Get cached detections if using cache
        cached_detections = None
        current_detections = {}

        if use_cache and cache_manager.cache_exists(video_path, model_name):
            cached_detections = cache_manager.load_detections(video_path, model_name)
            print(f"Using cached detections for {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0, 0, 0

        frame_idx = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            # Check if we have cached detections for this frame
            if cached_detections and frame_idx in cached_detections:
                # Use cached detection results
                detections = cached_detections[frame_idx]
                # Visualize the cached detections if needed
                # This is just for demonstration, you can comment out if not needed
                if len(detections) > 0:
                    for box_data in detections:
                        if 'xyxy' in box_data:
                            # If the cached format contains xyxy
                            xyxy = box_data['xyxy'][0]  # First item in the list
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                # Perform detection and time it
                detection_start = time.time()
                results = model(frame, imgsz=320, verbose=False)
                detection_end = time.time()
                detection_time += detection_end - detection_start

                # Process results for caching
                detections = process_results(results)
                current_detections[frame_idx] = detections

                # Visualize the detections
                # This is just for demonstration, you can comment out if not needed
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Display the frame (optional)
            cv2.imshow("Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Display progress
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames...")

            frame_idx += 1

        total_time = time.time() - start_time
        cv2.destroyAllWindows()

        # Save cache if needed
        if use_cache and current_detections and not cached_detections:
            cache_manager.save_detections(video_path, model_name, current_detections)

        cap.release()
        return total_frames, total_time, detection_time
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return 0, 0, 0


def process_results(results):
    """
    Process YOLO results for caching.

    Args:
        results: The results from YOLO model

    Returns:
        Processed detection data
    """
    boxes_data = []
    for r in results:
        boxes = r.boxes
        for i in range(len(boxes)):
            try:
                box = boxes[i]
                box_data = {
                    'xyxy': box.xyxy.cpu().numpy().tolist(),
                    'conf': box.conf.cpu().numpy().tolist(),
                    'cls': box.cls.cpu().numpy().tolist(),
                }
                if hasattr(box, 'id') and box.id is not None:
                    box_data['id'] = box.id.cpu().numpy().tolist()
                boxes_data.append(box_data)
            except Exception as e:
                print(f"Error processing box {i}: {str(e)}")
    return boxes_data


def main():
    parser = argparse.ArgumentParser(description='Test YOLO detection caching performance')
    parser.add_argument('--videos', nargs='+', help='List of video paths to process')
    parser.add_argument('--model', help='YOLO model to use')
    parser.add_argument('--output', help='Output path for performance plot')
    args = parser.parse_args()

    print("Environment Variables:")
    print(f"- PROJECT_ROOT: {os.getenv('PROJECT_ROOT')}")
    print(f"- VIDEO_PATHS: {os.getenv('VIDEO_PATHS')}")
    print(f"- YOLO_MODEL_PATH: {os.getenv('YOLO_MODEL_PATH')}")

    # Use environment variables if arguments are not provided
    video_paths = args.videos
    if not video_paths:
        # Get video paths from environment variable
        env_videos = os.getenv('VIDEO_PATHS')
        if env_videos:
            # Split by comma and clean up paths
            video_paths = [path.strip() for path in env_videos.split(',')]
            print(f"Using videos from environment: {video_paths}")
        else:
            print(
                "Error: No videos specified. Please provide --videos argument or set VIDEO_PATHS environment variable.")
            return

    model_path = args.model
    if not model_path:
        model_path = os.getenv('YOLO_MODEL_PATH')
        if model_path:
            print(f"Using model from environment: {model_path}")
        else:
            model_path = 'yolov8n.pt'  # Default model
            print(f"Using default model: {model_path}")

    # Initialize model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer()

    # Process each video twice - once without cache, once with cache
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"Warning: Video path does not exist: {video_path}")
            continue

        video_name = os.path.basename(video_path)

        print(f"\nProcessing {video_name} without cache...")
        total_frames, total_time, detection_time = process_video(video_path, model, use_cache=False)
        analyzer.add_result(video_name, False, total_frames, total_time, detection_time)

        print(f"\nProcessing {video_name} with cache...")
        total_frames, total_time, detection_time = process_video(video_path, model, use_cache=True)
        analyzer.add_result(video_name, True, total_frames, total_time, detection_time)

    # Print and plot results
    analyzer.print_summary()

    # Generate plot if needed
    output_path = args.output
    if not output_path:
        # Use default output path
        output_path = os.path.join(os.getenv('PROJECT_ROOT', '.'), 'reports', 'cache_performance.png')
        print(f"Using default output path: {output_path}")

    analyzer.plot_comparison(output_path)


if __name__ == "__main__":
    main()