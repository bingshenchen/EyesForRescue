# src/core/detection/fall_detector.py

import asyncio
import logging
import time
import textwrap
from asyncio import Queue, Lock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from config.settings import get_settings
from src.core.analysis.danger_calculator import calculate_danger
from src.core.analysis.gpt_analyzer import analyze_image
from src.core.analysis.location_service import getLoc, get_address
from src.core.detection.optimized_fall_detector import OptimizedFallDetector
from src.core.tracking.sort_tracker import SORT
from src.core.utils.cache_manager import DetectionCache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration settings
settings = get_settings()

# Initialize detection cache
detection_cache = DetectionCache(cache_dir=settings.CACHE_DIR)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=2)
task_queue = Queue()
results_lock = Lock()
analyst_results = {}

# Location data
try:
    latitude, longitude = getLoc()
    global_coords = f"Lat: {latitude:.4f}, Lon: {longitude:.4f}"
    global_address = get_address(latitude, longitude)
    logger.info(f"Location initialized: {global_coords}")
except Exception as e:
    logger.warning(f"Failed to get location: {e}")
    latitude, longitude = 0.0, 0.0
    global_coords = "Location unavailable"
    global_address = "Address unavailable"

# Detection settings
CONFIDENCE_THRESHOLD_DETECTION = settings.CONFIDENCE_THRESHOLD


def load_yolo_model(model_path=None):
    """
    Load YOLO model from specified path.

    Args:
        model_path: Path to YOLO model file

    Returns:
        Loaded YOLO model
    """
    if model_path is None:
        model_path = settings.YOLO_MODEL_PATH

    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    logger.info(f"Loading YOLO model from: {model_path}")
    return YOLO(str(model_path))


def load_classifier(classifier_path=None):
    """
    Load pose classifier from specified path.

    Args:
        classifier_path: Path to classifier file

    Returns:
        Loaded classifier model
    """
    if classifier_path is None:
        classifier_path = settings.CLASSIFIER_PATH

    if not Path(classifier_path).exists():
        logger.warning(f"Classifier not found: {classifier_path}")
        return None

    logger.info(f"Loading classifier from: {classifier_path}")
    return joblib.load(str(classifier_path))


async def analyze_image_async(track_id, frame, bbox=None):
    """
    Asynchronously execute analyze_image with proper bounding box extraction.

    Args:
        track_id: Unique track identifier
        frame: Video frame to analyze
        bbox: Person bounding box (x1, y1, x2, y2) - NEW PARAMETER

    Returns:
        Analysis result dictionary
    """
    try:
        logger.debug(f"Starting analysis for track ID {track_id}")

        # CRITICAL FIX: Extract person crop if bbox provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Ensure valid coordinates
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))

            # Extract person crop
            person_crop = frame[y1:y2, x1:x2]

            # Validate crop size
            if person_crop.size > 0:
                analysis_frame = person_crop
                logger.debug(f"Using person crop: {person_crop.shape} for track {track_id}")
            else:
                logger.warning(f"Invalid person crop for track {track_id}, using full frame")
                analysis_frame = frame
        else:
            # Fallback to full frame if no bbox
            analysis_frame = frame
            logger.debug(f"No bbox provided for track {track_id}, using full frame")

        # Execute analysis on the appropriate frame/crop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, analyze_image, analysis_frame)
        logger.debug(f"Analysis completed for track ID {track_id}")

        async with results_lock:
            analyst_results[track_id] = result
        return result

    except Exception as e:
        logger.error(f"Error during analyze_image_async for track ID {track_id}: {e}")
        return {}


async def process_task_queue():
    """
    Process analysis tasks from the queue asynchronously.
    """
    while True:
        try:
            # Use timeout to prevent hanging
            try:
                track_id, frame = await asyncio.wait_for(task_queue.get(), timeout=1.0)
                logger.debug(f"Processing task for track ID: {track_id}")

                result = await analyze_image_async(track_id, frame)

                async with results_lock:
                    analyst_results[track_id] = result

                task_queue.task_done()

            except asyncio.TimeoutError:
                # No task available, continue waiting
                continue

        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            logger.info("Task queue processing cancelled")
            break
        except Exception as e:
            logger.error(f"Error in task queue processing: {e}")
            try:
                task_queue.task_done()
            except ValueError:
                # task_done called too many times, ignore
                pass


async def track_objects_with_yolo(frame, tracking_model, pose_model, classifier, mot_tracker, frame_idx):
    """
    Perform YOLO tracking, pose estimation, danger calculation, and analysis.

    Args:
        frame: Input video frame
        tracking_model: YOLO tracking model
        pose_model: YOLO pose estimation model
        classifier: Pose classifier model
        mot_tracker: SORT tracker instance
        frame_idx: Current frame index

    Returns:
        Tuple of (processed_frame, falling_durations)
    """
    global global_coords, global_address

    # Perform object detection using YOLO
    try:
        results = tracking_model(frame, imgsz=320, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}")
        return frame, {}

    # Initialize tracking attributes if not present
    if not hasattr(mot_tracker, "falling_durations"):
        mot_tracker.falling_durations = {}
    if not hasattr(mot_tracker, "first_analysis_done"):
        mot_tracker.first_analysis_done = {}

    # Filter valid detections
    valid_detections = []
    for box, confidence, cls in zip(boxes, confidences, classes):
        if confidence > CONFIDENCE_THRESHOLD_DETECTION:
            valid_detections.append([*box, int(cls)])

    # Update object trackers
    tracks = mot_tracker.update(valid_detections)
    danger_values = []

    for track in tracks:
        x1, y1, x2, y2, track_id, cls = map(int, track)

        try:
            class_name = tracking_model.names[cls]
        except (IndexError, KeyError):
            class_name = "unknown"
            logger.warning(f"Unknown class ID: {cls}")

        # Initialize tracking data for new tracks
        if track_id not in mot_tracker.falling_durations:
            mot_tracker.falling_durations[track_id] = 0
            mot_tracker.first_analysis_done[track_id] = False

        # Update falling duration based on class
        if class_name == "falling_person":
            mot_tracker.falling_durations[track_id] += 1
        else:
            mot_tracker.falling_durations[track_id] = max(0, mot_tracker.falling_durations[track_id] - 1)

        # Analysis logic
        async with results_lock:
            analysis_result = analyst_results.get(track_id, {})

            # Trigger first analysis
            if frame_idx == 0 and not mot_tracker.first_analysis_done[track_id]:
                logger.debug(f"Triggering first analysis for track ID {track_id}")
                await task_queue.put((track_id, frame))
                mot_tracker.first_analysis_done[track_id] = True

            # Update analysis for falling persons
            elif class_name == "falling_person" and mot_tracker.falling_durations[track_id] % 100 == 0:
                danger_value = calculate_danger(analysis_result, mot_tracker.falling_durations[track_id])
                danger_values.append(danger_value)
                logger.debug(f"Danger value calculated: {danger_value:.2f}")

                logger.debug(f"Updating analysis for track ID {track_id}")
                await task_queue.put((track_id, frame))

        # Pose estimation and classification if classifier is available
        prediction_label = "Unknown"
        if classifier is not None:
            try:
                person_box = frame[y1:y2, x1:x2]
                if person_box.size > 0:
                    person_patch = cv2.resize(person_box, (224, 224))
                    person_patch = person_patch.astype(np.uint8)

                    pose_results = pose_model(person_patch)
                    keypoints = pose_results[0].keypoints

                    if keypoints is not None:
                        features = keypoints.xy.cpu().numpy().flatten()
                        features = np.pad(features, (0, 34 - len(features)), 'constant') if len(
                            features) < 34 else features[:34]
                    else:
                        features = np.zeros(34)

                    prediction = classifier.predict([features])[0]
                    prediction_label = "Need Help" if prediction == 0 else "Fine"

            except Exception as e:
                logger.error(f"Pose classification failed for track {track_id}: {e}")

        # Display bounding box and information
        color = (0, 0, 255) if class_name == "falling_person" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Main label
        label = f"ID: {track_id} | {class_name}: {mot_tracker.falling_durations[track_id]} | {prediction_label}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display analysis results
        display_analysis_results(frame, analysis_result, x2 + 10, y1 + 20)

    # Display danger score if available
    if danger_values:
        avg_danger = sum(danger_values) / len(danger_values)
        cv2.putText(frame, f"Danger: {avg_danger:.2f}", (frame.shape[1] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frame information
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, global_coords, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Display address with text wrapping
    wrapped_address = textwrap.wrap(global_address, width=30)
    y_start = 90
    for i, line in enumerate(wrapped_address):
        y_offset = y_start + i * 20
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return frame, mot_tracker.falling_durations


def display_analysis_results(frame, analysis_result, start_x, start_y):
    """
    Display GPT analysis results on the frame.

    Args:
        frame: Video frame to draw on
        analysis_result: Analysis result dictionary
        start_x: Starting x coordinate
        start_y: Starting y coordinate
    """
    text_x, text_y = start_x, start_y

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
                cv2.putText(frame, f"- {item}", (text_x + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                text_y += 15
        else:
            line = f"{key}: {value}"
            cv2.putText(frame, line, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            text_y += 15


async def process_single_video(video_source, tracking_model, pose_model, classifier, mot_tracker):
    """
    Process a single video or camera feed asynchronously with optimized classifier integration.

    Args:
        video_source: Video file path or camera index
        tracking_model: YOLO tracking model
        pose_model: YOLO pose model
        classifier: Pose classifier
        mot_tracker: SORT tracker instance
    """
    # Initialize optimized fall detector
    optimized_detector = OptimizedFallDetector(settings)
    optimized_detector.classifier = classifier
    optimized_detector.pose_model = pose_model

    # Determine if source is camera or video file
    if isinstance(video_source, int) or str(video_source).isdigit():
        cap = cv2.VideoCapture(int(video_source))
        use_cache = False
        logger.info(f"Processing camera feed: {video_source}")
    else:
        cap = cv2.VideoCapture(str(video_source))
        use_cache = settings.CACHE_ENABLED
        logger.info(f"Processing video file: {video_source}")

    if not cap.isOpened():
        logger.error(f"Unable to open video source: {video_source}")
        return

    # Get model name for cache key
    try:
        model_name = Path(tracking_model.ckpt_path).name if hasattr(tracking_model, 'ckpt_path') else "yolo_model"
    except:
        model_name = "yolo_model"

    # Cache management
    cached_detections = None
    current_detections = {}

    if use_cache and detection_cache.cache_exists(video_source, model_name):
        try:
            cached_detections = detection_cache.load_detections(video_source, model_name)
            logger.info(f"Using cached detections for {video_source}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    # Performance tracking
    frame_idx = 0
    start_time = time.time()
    detection_time = 0
    total_frames = 0

    # Simple tracking for demo - use consistent track IDs
    track_id_mapping = {}  # Map detection index to consistent track ID
    next_track_id = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            detection_start = time.time()

            # Get YOLO detections
            try:
                results = tracking_model(frame, imgsz=320, verbose=False)
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                else:
                    boxes = np.array([])
                    confidences = np.array([])
                    classes = np.array([])
            except Exception as e:
                logger.error(f"YOLO detection error: {e}")
                boxes = np.array([])
                confidences = np.array([])
                classes = np.array([])

            # Prepare detections for optimized processor
            detections = []
            for i, (box, confidence, cls) in enumerate(zip(boxes, confidences, classes)):
                if confidence > CONFIDENCE_THRESHOLD_DETECTION:
                    x1, y1, x2, y2 = map(int, box)

                    # Ensure valid bounding box
                    if x2 > x1 and y2 > y1:
                        class_name = tracking_model.names.get(int(cls), 'unknown') if hasattr(tracking_model,
                                                                                              'names') else 'unknown'

                        # Use consistent track ID mapping
                        detection_key = f"{x1}_{y1}_{x2}_{y2}"  # Simple spatial mapping
                        if detection_key not in track_id_mapping:
                            track_id_mapping[detection_key] = next_track_id
                            next_track_id += 1

                        track_id = track_id_mapping[detection_key]

                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(confidence),
                            'class_name': class_name,
                            'track_id': track_id
                        })

            # Use optimized fall detection
            try:
                fall_results = optimized_detector.smart_fall_detection(frame, detections, frame_idx)

                # Add bbox information to results for drawing
                for i, result in enumerate(fall_results):
                    if i < len(detections):
                        result['bbox'] = detections[i]['bbox']

            except Exception as e:
                logger.error(f"Optimized fall detection error: {e}")
                fall_results = []

            # Draw results on frame
            try:
                frame = draw_optimized_results(frame, fall_results)
            except Exception as e:
                logger.error(f"Drawing results error: {e}")

            detection_time += time.time() - detection_start

            # Display frame
            cv2.imshow("Optimized Fall Detection", frame)

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Processing stopped by user")
                break

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        optimized_detector.cleanup()

    # Print performance statistics
    total_time = time.time() - start_time
    if total_frames > 0:
        logger.info(f"Optimized Performance metrics:")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average FPS: {total_frames / total_time:.2f}")
        logger.info(f"  Detection time: {detection_time:.2f}s ({detection_time / total_time * 100:.1f}%)")

        # Get optimized detector stats
        try:
            stats = optimized_detector.get_performance_stats()
            logger.info(f"  Classifier queue size: {stats['classification_queue_size']}")
            logger.info(f"  Cached classifications: {stats['cached_classifications']}")
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")


def draw_optimized_results(frame, fall_results):
    """
    Draw optimized fall detection results on frame.

    Args:
        frame: Video frame to draw on
        fall_results: Results from OptimizedFallDetector

    Returns:
        Modified frame with drawn results
    """
    # Limit the number of information displays to prevent clutter
    max_displays = 3
    active_tracks = []

    for result in fall_results:
        track_id = result['track_id']
        class_name = result['class_name']
        is_falling = result['is_falling']
        classification = result['classification']
        danger_level = result['danger_level']
        needs_help = result['needs_help']

        # Choose color based on danger level
        if needs_help:
            color = (0, 0, 255)  # Red for high danger
        elif is_falling:
            color = (0, 165, 255)  # Orange for falling
        elif danger_level > 0.3:
            color = (0, 255, 255)  # Yellow for medium danger
        else:
            color = (0, 255, 0)  # Green for safe

        # Draw bounding box if bbox information is available
        if 'bbox' in result:
            x1, y1, x2, y2 = result['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label above bounding box
            label = f"ID:{track_id} {class_name}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Only show detailed info for active/important tracks
        if len(active_tracks) < max_displays and (needs_help or is_falling or danger_level > 0.1):
            active_tracks.append(result)

    # Draw information text on left side only for active tracks
    for idx, result in enumerate(active_tracks):
        track_id = result['track_id']
        class_name = result['class_name']
        classification = result['classification']
        danger_level = result['danger_level']
        needs_help = result['needs_help']

        # Choose color based on danger level
        if needs_help:
            color = (0, 0, 255)
        elif result['is_falling']:
            color = (0, 165, 255)
        elif danger_level > 0.3:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        y_offset = 30 + idx * 80  # Use index instead of track_id to prevent jumping

        cv2.putText(frame, f"ID: {track_id}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Class: {class_name}", (10, y_offset + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"Classification: {classification}", (10, y_offset + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"Danger: {danger_level:.2f}", (10, y_offset + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Show global alerts
    if any(result['needs_help'] for result in fall_results):
        cv2.putText(frame, "NEEDS HELP!", (frame.shape[1] - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame


async def process_videos(video_paths, tracking_model=None, pose_model=None, classifier=None):
    """
    Process multiple videos asynchronously with optimized fall detection.

    Args:
        video_paths: List of video file paths
        tracking_model: YOLO tracking model (optional, will load default)
        pose_model: YOLO pose model (optional, will load default)
        classifier: Pose classifier (optional, will load default)
    """
    # Load models if not provided
    if tracking_model is None:
        tracking_model = load_yolo_model()
    if pose_model is None:
        pose_model = load_yolo_model(settings.POSE_MODEL_PATH)
    if classifier is None:
        classifier = load_classifier()

    # Initialize tracker using centralized SORT implementation
    tracker_settings = settings.TRACKING_SETTINGS
    mot_tracker = SORT(
        max_miss=tracker_settings['max_miss'],
        min_hits=tracker_settings['min_hits'],
        iou_threshold=tracker_settings['iou_threshold']
    )

    # Start task queue processor
    task_processor = asyncio.create_task(process_task_queue())

    try:
        # Process videos with optimized detector
        tasks = [
            process_single_video(video_path, tracking_model, pose_model, classifier, mot_tracker)
            for video_path in video_paths
        ]
        await asyncio.gather(*tasks)
    finally:
        # Clean up task processor
        task_processor.cancel()
        try:
            await task_processor
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    # Test with configuration
    logger.info("Starting fall detection system...")

    try:
        # Test video paths from settings
        test_video = settings.TEST_VIDEO_PATH
        if test_video and test_video.exists():
            video_paths = [str(test_video)]
        else:
            # Use camera if no test video
            video_paths = [0]
            logger.info("No test video found, using camera")

        # Run detection
        asyncio.run(process_videos(video_paths))

    except KeyboardInterrupt:
        logger.info("Fall detection stopped by user")
    except Exception as e:
        logger.error(f"Fall detection failed: {e}")
        raise