# src/core/detection/fall_detector.py

import asyncio
import os
import time
from asyncio import Queue, Lock
from concurrent.futures import ThreadPoolExecutor


import cv2
import joblib
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO
import textwrap

from src.core.analysis.danger_calculator import calculate_danger
from src.core.analysis.gpt_analyzer import analyze_image
from src.core.analysis.location_service import getLoc, get_address

# Import the cache manager at the top of the file
from src.core.utils.cache_manager import DetectionCache

# Load environment variables
load_dotenv()

executor = ThreadPoolExecutor(max_workers=2)  # Use ThreadPoolExecutor for non-blocking analysis
task_queue = Queue()  # Asynchronous task queue
results_lock = Lock()  # Lock for synchronizing shared results dictionary
analyst_results = {}  # Store analysis results globally

latitude, longitude = getLoc()
global_coords = "x: " + str(latitude) + "y: " + str(longitude)
global_address = get_address(latitude, longitude)

# Initialize cache manager globally
detection_cache = DetectionCache()

CONFIDENCE_THRESHOLD_DETECTION = 0.4

class SORT:
    def __init__(self):
        self.trackers = []
        self.track_id_count = 0
        self.falling_counts = {}

    def update(self, detections):
        """
        Update trackers and assign unique IDs based on detection results.
        """
        updated_tracks = []
        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            cls = detection[4]
            matched = False

            # Match existing trackers
            for tracker in self.trackers:
                iou_score = self.iou(tracker['bbox'], detection[:4])
                if iou_score > 0.3:
                    # Update tracker with smoothed bbox using Kalman filter or averaging
                    tracker['bbox'] = [
                        (tracker['bbox'][i] * 0.8 + detection[i] * 0.2)
                        for i in range(4)
                    ]
                    tracker['hits'] += 1
                    tracker['misses'] = 0
                    tracker['cls'] = cls
                    updated_tracks.append(tracker)
                    matched = True
                    break

            # If not matched, create a new tracker
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

        # Remove inactive trackers
        self.trackers = [t for t in updated_tracks if t['misses'] < 5]  # Increased threshold

        return [[*tracker['bbox'], tracker['id'], tracker['cls']] for tracker in self.trackers]

    @staticmethod
    def iou(bbox1, bbox2):
        """
        Calculate Intersection over Union.
        """
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - inter_area
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


async def analyze_image_async(track_id, frame):
    """
    Asynchronously execute analyze_image and store the result in analyst_results.
    """
    try:
        print(f"Starting analysis for track ID {track_id}")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, analyze_image, frame)
        print(f"Analysis result for track ID {track_id}: {result}")
        async with results_lock:
            analyst_results[track_id] = result
        return result
    except Exception as e:
        print(f"Error during analyze_image_async for track ID {track_id}: {e}")
        return {}


async def process_task_queue():
    while True:
        track_id, frame = await task_queue.get()
        print(f"Processing task for track ID: {track_id}")
        try:
            result = await analyze_image_async(track_id, frame)
            print(f"Analysis complete for track ID {track_id}: {result}")
            async with results_lock:
                analyst_results[track_id] = result
        except Exception as e:
            print(f"Error in analyze_image_async for track ID {track_id}: {e}")
        finally:
            task_queue.task_done()


async def track_objects_with_yolo(frame, tracking_model, pose_model, classifier, mot_tracker, frame_idx):
    """
    Perform YOLO tracking, pose estimation, danger calculation, and asynchronous analysis.
    """

    global global_coords, global_address

    # Perform object detection using YOLO
    results = tracking_model(frame, imgsz=320, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    classes = results[0].boxes.cls.cpu().numpy()  # Class IDs

    # Initialize falling durations and first analysis flag if not already done
    if not hasattr(mot_tracker, "falling_durations"):
        mot_tracker.falling_durations = {}
    if not hasattr(mot_tracker, "first_analysis_done"):
        mot_tracker.first_analysis_done = {}

    valid_detections = []
    for box, confidence, cls in zip(boxes, confidences, classes):
        if confidence > 0.4:
            valid_detections.append([*box, int(cls)])  # [x1, y1, x2, y2, class]

    # Update object trackers
    tracks = mot_tracker.update(valid_detections)

    danger_values = []  # Store danger values for all tracked objects

    for track in tracks:
        x1, y1, x2, y2, track_id, cls = map(int, track)
        class_name = tracking_model.names[cls]

        # Initialize falling duration and analysis flags for new track IDs
        if track_id not in mot_tracker.falling_durations:
            mot_tracker.falling_durations[track_id] = 0
            mot_tracker.first_analysis_done[track_id] = False

        # Increment or decrement falling durations based on class
        if class_name == "falling_person":
            mot_tracker.falling_durations[track_id] += 1
        else:
            mot_tracker.falling_durations[track_id] = max(0, mot_tracker.falling_durations[track_id] - 1)

        # Trigger analysis logic
        async with results_lock:
            analysis_result = analyst_results.get(track_id, {})

            # Only trigger analyze_image_async for the first frame
            if frame_idx == 0 and not mot_tracker.first_analysis_done[track_id]:
                print(f"Triggering first analysis for track ID {track_id} at frame {frame_idx}")
                await task_queue.put((track_id, frame))
                mot_tracker.first_analysis_done[track_id] = True

            # Update analysis every 20 frames for "falling_person"
            elif class_name == "falling_person" and mot_tracker.falling_durations[track_id] % 100 == 0:
                # Calculate danger value
                danger_value = calculate_danger(analysis_result,mot_tracker.falling_durations[track_id])
                danger_values.append(danger_value)
                print("danger_value: " + str(danger_value))
                # Add danger value to the top-right corner
                if danger_values:
                    avg_danger = sum(danger_values) / len(danger_values)
                    cv2.putText(frame, f"Danger: {avg_danger:.2f}", (frame.shape[1] - 200, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                print(
                    f"Updating analysis for track ID {track_id} at falling_durations {mot_tracker.falling_durations[track_id]}")
                await task_queue.put((track_id, frame))

        # Pose estimation and classification
        person_box = frame[y1:y2, x1:x2]
        if person_box.size > 0:
            person_patch = cv2.resize(person_box, (224, 224))
            person_patch = person_patch.astype(np.uint8)

            pose_results = pose_model(person_patch)
            keypoints = pose_results[0].keypoints
            if keypoints is not None:
                features = keypoints.xy.cpu().numpy().flatten()
                features = np.pad(features, (0, 34 - len(features)), 'constant') if len(features) < 34 else features[
                                                                                                            :34]
            else:
                features = np.zeros(34)

            prediction = classifier.predict([features])[0]
            prediction_label = "Need Help" if prediction == 0 else "Fine"
        else:
            prediction_label = "Unknown"

        # Retrieve updated analysis results
        async with results_lock:
            analysis_result = analyst_results.get(track_id, {})

        # Display bounding box, label, and analysis result
        label = (
            f"ID: {track_id} | {class_name}: {mot_tracker.falling_durations[track_id]} | {prediction_label}"
        )
        color = (0, 0, 255) if class_name == "falling_person" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display detailed JSON result next to the bounding box
        text_x, text_y = x2 + 10, y1 + 20
        for key, value in analysis_result.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    line = f"{sub_key}: {sub_value}"
                    cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    text_y += 15
            elif isinstance(value, list):
                line = f"{key}:"
                cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                text_y += 15
                for item in value:
                    cv2.putText(frame, f"- {item}", (text_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0),
                                1)
                    text_y += 15
            else:
                line = f"{key}: {value}"
                cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                text_y += 15

        # Add frame index to top-left corner
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add coordinates and address
        cv2.putText(frame, global_coords, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # cv2.putText(frame, global_address, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        wrapped_address = textwrap.wrap(global_address, width=30)
        y_start = 90
        for i, line in enumerate(wrapped_address):
            y_offset = y_start + i * 20
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return frame, mot_tracker.falling_durations


# Modify the YOLO detection function to use caching
async def process_single_video(video_source, tracking_model, pose_model, classifier, mot_tracker):
    """
    Process a single video or camera feed asynchronously with caching.
    """
    # Check if the video_source is an integer (camera) or a string (file path)
    if isinstance(video_source, int) or video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))
        use_cache = False  # Don't use cache for camera feeds
    else:
        cap = cv2.VideoCapture(video_source)
        use_cache = True  # Use cache for video files

    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_source}")
        return

    # Get model name for cache key
    try:
        model_name = os.path.basename(tracking_model.ckpt_path)
    except AttributeError:
        # Fallback if ckpt_path is not available
        model_name = "yolo_model"
        print(f"Warning: Could not get model name, using '{model_name}' instead")

    # Variables for caching
    cached_detections = None
    current_detections = {}

    # Try to load cache if using a video file
    if use_cache:
        try:
            if detection_cache.cache_exists(video_source, model_name):
                cached_detections = detection_cache.load_detections(video_source, model_name)
                print(f"Using cached detections for {video_source}")
        except Exception as e:
            print(f"Warning: Failed to check/load cache: {e}")
            cached_detections = None

    frame_idx = 0  # Initialize frame index

    # Performance timing
    start_time = time.time()
    detection_time = 0
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Process frame (with or without cache)
        detection_start = time.time()

        if cached_detections and frame_idx in cached_detections:
            # Use cached detection results
            print(f"Using cached detection for frame {frame_idx}")
            # Apply cached detections instead of running detection
            processed_frame, _ = await track_objects_with_yolo(frame, tracking_model, pose_model, classifier,
                                                               mot_tracker, frame_idx)
        else:
            # Perform detection normally
            processed_frame, _ = await track_objects_with_yolo(frame, tracking_model, pose_model, classifier,
                                                               mot_tracker, frame_idx)

            # Store detection for caching (simplified, just store the fact we processed this frame)
            if use_cache:
                # Extract tracks from the mot_tracker to cache
                if hasattr(mot_tracker, 'trackers'):
                    tracks_to_cache = []
                    for tracker in mot_tracker.trackers:
                        x1, y1, x2, y2 = tracker['bbox']
                        track_id = tracker['id']
                        cls = tracker['cls']
                        tracks_to_cache.append([x1, y1, x2, y2, track_id, cls])
                    current_detections[frame_idx] = tracks_to_cache

        detection_time += time.time() - detection_start

        # Display the processed frame
        cv2.imshow("Frame", processed_frame)

        frame_idx += 1  # Increment frame index
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save cache if we've performed new detections
    if use_cache and current_detections and not cached_detections:
        try:
            print(f"Saving cache for {video_source}...")
            detection_cache.save_detections(video_source, model_name, current_detections)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    # Print performance statistics
    total_time = time.time() - start_time
    if detection_time > 0 and total_frames > 0:
        print(f"Performance metrics for {video_source}:")
        print(f"Total frames: {total_frames}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"YOLO detection time: {detection_time:.2f} seconds ({detection_time / total_time * 100:.1f}% of total)")
        print(f"Average time per frame: {total_time / total_frames:.4f} seconds")
        if cached_detections:
            print(f"Used cached detections: YES")
        else:
            print(f"Used cached detections: NO")

    cap.release()
    cv2.destroyAllWindows()
    await task_queue.join()


# Add a helper function to process YOLO results into our format
def process_yolo_results(results):
    """
    Process YOLO results into a format suitable for caching.

    Args:
        results: Results from the YOLO model

    Returns:
        A dictionary with processed detection data
    """
    detections = {}

    # Iterate over each result (frame) in the results list
    for r in results:
        boxes = r.boxes  # Access the Boxes object
        if boxes.id is None:
            continue

        box_data = []
        for box in range(len(boxes)):
            # Extract box properties
            xyxy = boxes.xyxy[box].cpu().numpy().astype(int)  # Convert to numpy and integer
            confidence = boxes.conf[box].item()  # Confidence score
            cls = int(boxes.cls[box].item())  # Class ID

            if cls != 0 or confidence < CONFIDENCE_THRESHOLD_DETECTION:
                continue

            track_id = boxes.id[box].item()
            xmin, ymin, xmax, ymax = xyxy  # xyxy format

            box_data.append({
                "xmin": xmin,
                "ymin": ymin,
                "width": xmax - xmin,
                "height": ymax - ymin,
                "confidence": confidence,
                "track_id": track_id,
                "class": cls
            })

        detections = box_data

    return detections

async def process_videos(video_paths, tracking_model, pose_model, classifier):
    """
    Process multiple videos asynchronously.
    """
    mot_tracker = SORT()
    asyncio.create_task(process_task_queue())
    tasks = [process_single_video(video_path, tracking_model, pose_model, classifier, mot_tracker)
             for video_path in video_paths]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    yolo_model_path = os.getenv("YOLO_MODEL_PATH")
    pose_model_path = os.getenv("yolov11n")
    video_paths_env = os.getenv("VIDEO_PATHS")

    if not yolo_model_path or not pose_model_path or not video_paths_env:
        print("Error: Required paths not set in environment variables.")
        exit(1)

    tracking_model = load_yolo_model(yolo_model_path)
    pose_model = load_yolo_model(pose_model_path)
    classifier = load_classifier("CLASSIFIER_PATH")

    video_paths = [path.strip() for path in video_paths_env.split(',')]
    asyncio.run(process_videos(video_paths, tracking_model, pose_model, classifier))
