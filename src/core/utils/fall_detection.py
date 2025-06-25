# src/core/utils/fall_detection.py

import logging
from pathlib import Path
from typing import List, Optional

import cv2
from ultralytics import YOLO

from config.settings import get_settings
from src.core.analysis.danger_calculator import calculate_danger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration settings
settings = get_settings()


def detect_fall_in_video(video_path: str,
                         model_path: Optional[str] = None,
                         classes: Optional[List[str]] = None,
                         conf_threshold: Optional[float] = None,
                         batch_size: int = 16) -> List[List[int]]:
    """
    Detect falls in the given video using a YOLO model and return tracking data.

    Args:
        video_path: Path to the video file
        model_path: Path to the trained YOLO model (.pt file). If None, uses default from settings
        classes: List of class names. If None, uses default from settings
        conf_threshold: Confidence threshold for YOLO predictions. If None, uses default from settings
        batch_size: Number of frames to process in a batch to speed up detection

    Returns:
        List of lists containing fall detection results for each frame
        Each inner list contains class IDs detected in that frame
    """
    # Use default values from settings if not provided
    if model_path is None:
        model_path = settings.YOLO_MODEL_PATH
    if classes is None:
        classes = settings.CLASSES
    if conf_threshold is None:
        conf_threshold = settings.CONFIDENCE_THRESHOLD

    # Validate model path
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    # Validate video path
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logger.info(f"Loading YOLO model from: {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Using classes: {classes}")
    logger.info(f"Confidence threshold: {conf_threshold}")

    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Unable to open video file: {video_path}")
        return []

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")

    results_list = []
    frames = []
    frame_count = 0

    # Create a mapping from class names to integer IDs
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    logger.debug(f"Class mapping: {class_to_id}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("Reached the end of the video")
                break

            frames.append(frame)
            frame_count += 1

            # Process batch when full or at end of video
            if len(frames) == batch_size or frame_count == total_frames:
                logger.debug(f"Processing batch of {len(frames)} frames (frame {frame_count}/{total_frames})")

                try:
                    # Detect objects in the batch of frames using YOLO
                    batch_results = model.predict(frames, conf=conf_threshold, stream=True, verbose=False)

                    for frame_results in batch_results:
                        detections = []

                        if frame_results.boxes is not None and len(frame_results.boxes) > 0:
                            for box in frame_results.boxes:
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])

                                # Get class name and map to ID
                                if class_id < len(classes):
                                    class_name = classes[class_id]
                                    mapped_id = class_to_id.get(class_name, 0)
                                    detections.append(mapped_id)

                                    logger.debug(
                                        f"Detected {class_name} (ID: {class_id}) -> mapped to {mapped_id}, confidence: {confidence:.2f}")
                                else:
                                    logger.warning(f"Unknown class ID: {class_id}")
                                    detections.append(0)  # Default to first class

                        results_list.append(detections)

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Add empty results for failed batch
                    for _ in range(len(frames)):
                        results_list.append([])

                frames = []  # Clear the batch

            # Progress logging
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                logger.info(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")

    except Exception as e:
        logger.error(f"Error during video processing: {e}")
        raise
    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info(f"Processing completed. Analyzed {len(results_list)} frames")
    logger.info(f"Total detections: {sum(len(frame_detections) for frame_detections in results_list)}")

    return results_list


def analyze_fall_patterns(detection_results: List[List[int]],
                          classes: Optional[List[str]] = None,
                          fall_class_names: Optional[List[str]] = None) -> dict:
    """
    Analyze fall patterns from detection results.

    Args:
        detection_results: List of detection results from detect_fall_in_video
        classes: List of class names. If None, uses default from settings
        fall_class_names: Names of classes that indicate falls. If None, uses ['falling_person']

    Returns:
        Dictionary containing analysis results
    """
    if classes is None:
        classes = settings.CLASSES
    if fall_class_names is None:
        fall_class_names = ['falling_person']

    # Create mapping from class names to IDs
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    fall_class_ids = [class_to_id.get(name, -1) for name in fall_class_names if name in class_to_id]

    logger.info(f"Analyzing fall patterns for classes: {fall_class_names} (IDs: {fall_class_ids})")

    total_frames = len(detection_results)
    frames_with_falls = 0
    fall_sequences = []
    current_sequence_length = 0

    for frame_idx, detections in enumerate(detection_results):
        has_fall = any(class_id in fall_class_ids for class_id in detections)

        if has_fall:
            frames_with_falls += 1
            current_sequence_length += 1
        else:
            if current_sequence_length > 0:
                fall_sequences.append(current_sequence_length)
                current_sequence_length = 0

    # Add final sequence if video ends with a fall
    if current_sequence_length > 0:
        fall_sequences.append(current_sequence_length)

    analysis = {
        'total_frames': total_frames,
        'frames_with_falls': frames_with_falls,
        'fall_percentage': (frames_with_falls / total_frames * 100) if total_frames > 0 else 0,
        'fall_sequences': fall_sequences,
        'longest_fall_sequence': max(fall_sequences) if fall_sequences else 0,
        'total_fall_sequences': len(fall_sequences),
        'average_fall_sequence_length': sum(fall_sequences) / len(fall_sequences) if fall_sequences else 0
    }

    logger.info(f"Fall analysis results:")
    logger.info(f"  Total frames: {analysis['total_frames']}")
    logger.info(f"  Frames with falls: {analysis['frames_with_falls']} ({analysis['fall_percentage']:.1f}%)")
    logger.info(f"  Fall sequences: {analysis['total_fall_sequences']}")
    logger.info(f"  Longest sequence: {analysis['longest_fall_sequence']} frames")

    return analysis


def save_detection_results(results: List[List[int]],
                           output_path: Optional[str] = None,
                           video_name: str = "unknown_video") -> Path:
    """
    Save detection results to a file.

    Args:
        results: Detection results from detect_fall_in_video
        output_path: Path to save results. If None, saves to default reports directory
        video_name: Name of the video (used in filename)

    Returns:
        Path to the saved file
    """
    if output_path is None:
        output_dir = settings.REPORTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fall_detection_{video_name}_{timestamp}.txt"
        output_path = output_dir / filename
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Fall Detection Results\n")
            f.write(f"Video: {video_name}\n")
            f.write(f"Total frames: {len(results)}\n")
            f.write(f"Classes: {', '.join(settings.CLASSES)}\n")
            f.write(f"Confidence threshold: {settings.CONFIDENCE_THRESHOLD}\n")
            f.write(f"\nFrame-by-frame detection results:\n")
            f.write("Frame_Index,Detected_Classes\n")

            for frame_idx, detections in enumerate(results):
                class_names = [settings.CLASSES[class_id] for class_id in detections
                               if 0 <= class_id < len(settings.CLASSES)]
                f.write(f"{frame_idx},{';'.join(class_names) if class_names else 'none'}\n")

        logger.info(f"Detection results saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to save detection results: {e}")
        raise


def process_video_with_danger_assessment(video_path: str,
                                         model_path: Optional[str] = None,
                                         save_results: bool = True) -> dict:
    """
    Process video with fall detection and danger assessment.

    Args:
        video_path: Path to the video file
        model_path: Path to YOLO model. If None, uses default from settings
        save_results: Whether to save results to file

    Returns:
        Dictionary containing detection results and danger assessment
    """
    logger.info(f"Processing video with danger assessment: {video_path}")

    # Perform fall detection
    detection_results = detect_fall_in_video(video_path, model_path)

    # Analyze fall patterns
    fall_analysis = analyze_fall_patterns(detection_results)

    # Calculate danger score
    danger_score = calculate_danger({'gpt_analysis': {}}, fall_analysis['frames_with_falls'])

    # Compile results
    results = {
        'video_path': str(video_path),
        'detection_results': detection_results,
        'fall_analysis': fall_analysis,
        'danger_score': danger_score,
        'settings_used': {
            'model_path': str(model_path) if model_path else str(settings.YOLO_MODEL_PATH),
            'classes': settings.CLASSES,
            'confidence_threshold': settings.CONFIDENCE_THRESHOLD
        }
    }

    # Save results if requested
    if save_results:
        video_name = Path(video_path).stem
        try:
            save_detection_results(detection_results, video_name=video_name)
        except Exception as e:
            logger.warning(f"Failed to save detection results: {e}")

    logger.info(f"Video processing completed. Danger score: {danger_score:.2f}")

    return results


def main():
    """
    Main function to detect falls in a video and calculate the danger score.
    Uses configuration from settings.py and environment variables.
    """
    logger.info("Starting fall detection utility")

    # Get test video path from settings
    test_video = settings.TEST_VIDEO_PATH
    if not test_video or not test_video.exists():
        logger.error("No test video found in configuration")
        logger.info("Please set TEST_VIDEO_PATH in your .env file")
        return

    # Get model path from settings
    model_path = settings.YOLO_MODEL_PATH
    if not model_path.exists():
        logger.error(f"YOLO model not found: {model_path}")
        logger.info("Please ensure the model file exists or update YOLO_MODEL_PATH in .env")
        return

    try:
        logger.info(f"Using model: {model_path}")
        logger.info(f"Processing video: {test_video}")
        logger.info(f"Classes: {settings.CLASSES}")

        # Process video with danger assessment
        results = process_video_with_danger_assessment(
            str(test_video),
            str(model_path),
            save_results=True
        )

        # Display summary
        print("\n" + "=" * 50)
        print("FALL DETECTION SUMMARY")
        print("=" * 50)
        print(f"Video: {results['video_path']}")
        print(f"Total frames: {results['fall_analysis']['total_frames']}")
        print(f"Frames with falls: {results['fall_analysis']['frames_with_falls']}")
        print(f"Fall percentage: {results['fall_analysis']['fall_percentage']:.1f}%")
        print(f"Fall sequences: {results['fall_analysis']['total_fall_sequences']}")
        print(f"Longest sequence: {results['fall_analysis']['longest_fall_sequence']} frames")
        print(f"Danger score: {results['danger_score']:.2f}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Fall detection failed: {e}")
        raise


if __name__ == "__main__":
    main()