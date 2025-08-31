# src/core/detection/fall_detector.py
"""
Unified Fall Detection System
Combines all optimizations and fixes the classifier integration issue
"""

import cv2
import numpy as np
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
import joblib
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any

from config.settings import get_settings
from src.core.tracking.sort_tracker import SORT
from src.core.utils.cache_manager import DetectionCache

logger = logging.getLogger(__name__)

# Constants
STANDARD_INPUT_SIZE = (224, 224)  # Classifier training size
YOLO_INPUT_SIZE = 224  # Must be multiple of 32
PERSON_CLASS_ID = 0  # COCO dataset person class


class UnifiedFallDetector:
    """
    Unified fall detection system with proper bounding box extraction.
    Fixes Issue #19: Classifier receives person bounding box instead of full frame.
    """

    def __init__(self, settings=None):
        """Initialize the unified fall detector."""
        self.settings = settings or get_settings()

        # Model paths
        self.yolo_path = self.settings.YOLO_MODEL_PATH
        self.pose_path = self.settings.POSE_MODEL_PATH
        self.classifier_path = self.settings.CLASSIFIER_PATH

        # Performance parameters
        self.confidence_threshold = self.settings.CONFIDENCE_THRESHOLD
        self.fall_confirmation_frames = 15
        self.classifier_interval = 10  # Frames between classifications

        # Load models
        self._load_models()

        # Initialize components
        self.tracker = self._init_tracker()
        self.cache = DetectionCache(cache_dir=self.settings.CACHE_DIR)

        # State management
        self.person_states = {}  # Track each person's state
        self.frame_count = 0

        # Performance monitoring
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'classifications': 0,
            'alerts': 0
        }

    def _load_models(self):
        """Load all required models."""
        try:
            # YOLO detection model
            logger.info(f"Loading YOLO model: {self.yolo_path}")
            self.yolo_model = YOLO(str(self.yolo_path))

            # Pose model for keypoint extraction
            logger.info(f"Loading pose model: {self.pose_path}")
            self.pose_model = YOLO(str(self.pose_path))

            # Classifier for fall verification
            if Path(self.classifier_path).exists():
                logger.info(f"Loading classifier: {self.classifier_path}")
                self.classifier = joblib.load(str(self.classifier_path))
            else:
                logger.warning("Classifier not found, fall verification disabled")
                self.classifier = None

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _init_tracker(self):
        """Initialize SORT tracker."""
        tracker_config = self.settings.TRACKING_SETTINGS
        return SORT(
            max_miss=tracker_config['max_miss'],
            min_hits=tracker_config['min_hits'],
            iou_threshold=tracker_config['iou_threshold']
        )

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame with proper bounding box extraction.

        Args:
            frame: Input video frame

        Returns:
            Dictionary containing detection results and alerts
        """
        self.frame_count += 1
        results = {
            'frame_id': self.frame_count,
            'detections': [],
            'alerts': [],
            'stats': {}
        }

        try:
            # Step 1: Detect persons in frame
            detections = self._detect_persons(frame)

            if not detections:
                return results

            # Step 2: Track persons across frames
            tracks = self._track_persons(detections)

            # Step 3: Process each tracked person
            for track in tracks:
                person_result = self._process_person(frame, track)
                if person_result:
                    results['detections'].append(person_result)

                    # Check for alerts
                    if person_result.get('alert_triggered', False):
                        results['alerts'].append(self._create_alert(person_result))

            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['detections'] += len(results['detections'])
            results['stats'] = self.stats.copy()

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

        return results

    def _detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame using YOLO.

        Args:
            frame: Input frame

        Returns:
            List of detection dictionaries
        """
        detections = []

        try:
            # Run YOLO detection
            results = self.yolo_model(frame, imgsz=320, verbose=False)

            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                # Filter for persons only
                for box, conf, cls in zip(boxes, confidences, classes):
                    if int(cls) == PERSON_CLASS_ID and conf > self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf),
                            'class': 'person'
                        })

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return detections

    def _track_persons(self, detections: List[Dict]) -> List[np.ndarray]:
        """
        Track detected persons using SORT tracker.

        Args:
            detections: List of detection dictionaries

        Returns:
            List of tracks with IDs
        """
        if not detections:
            return []

        # Convert to SORT format: [[x1,y1,x2,y2,cls]]
        det_array = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det_array.append([x1, y1, x2, y2, 0])  # cls=0 for person

        # Update tracker
        tracks = self.tracker.update(det_array)
        return tracks

    def _process_person(self, frame: np.ndarray, track: np.ndarray) -> Optional[Dict]:
        """
        Process individual tracked person with PROPER BOUNDING BOX EXTRACTION.

        This is the CRITICAL FIX: Extract person region before classification.

        Args:
            frame: Full video frame
            track: Track array [x1, y1, x2, y2, track_id, cls]

        Returns:
            Person processing result dictionary
        """
        x1, y1, x2, y2, track_id, cls = map(int, track)

        # Initialize person state if new
        if track_id not in self.person_states:
            self.person_states[track_id] = {
                'fall_history': deque(maxlen=30),
                'last_classification': None,
                'last_classification_frame': 0,
                'fall_start_frame': None,
                'alert_sent': False
            }

        state = self.person_states[track_id]

        # CRITICAL: Extract person bounding box
        person_crop = self._extract_person_region(frame, (x1, y1, x2, y2))

        if person_crop is None:
            return None

        # Classify if needed (not every frame for performance)
        classification = None
        if self.classifier and (self.frame_count - state['last_classification_frame']) >= self.classifier_interval:
            classification = self._classify_person(person_crop)
            state['last_classification'] = classification
            state['last_classification_frame'] = self.frame_count
            self.stats['classifications'] += 1
        else:
            classification = state['last_classification']

        # Update fall history
        is_fall = self._is_fall_posture(person_crop, classification)
        state['fall_history'].append(is_fall)

        # Check for sustained fall
        alert_triggered = False
        if self._check_sustained_fall(state):
            if not state['alert_sent']:
                alert_triggered = True
                state['alert_sent'] = True
                self.stats['alerts'] += 1
                logger.warning(f"ALERT: Person {track_id} has fallen and needs help!")
        else:
            state['alert_sent'] = False

        return {
            'track_id': track_id,
            'bbox': (x1, y1, x2, y2),
            'classification': classification,
            'is_falling': is_fall,
            'fall_duration': sum(state['fall_history']),
            'alert_triggered': alert_triggered
        }

    def _extract_person_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        CRITICAL METHOD: Extract person region from frame.
        This fixes the main issue - classifier should receive only the person, not full frame.

        Args:
            frame: Full video frame
            bbox: Person bounding box (x1, y1, x2, y2)

        Returns:
            Cropped and resized person region
        """
        try:
            x1, y1, x2, y2 = bbox

            # Validate bounding box
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Extract person region
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                logger.debug("Empty person crop")
                return None

            # Resize to standard size for classifier
            person_crop = cv2.resize(person_crop, STANDARD_INPUT_SIZE)

            return person_crop

        except Exception as e:
            logger.error(f"Bounding box extraction error: {e}")
            return None

    def _classify_person(self, person_crop: np.ndarray) -> Dict[str, Any]:
        """
        Classify person pose using the trained classifier.

        Args:
            person_crop: Extracted person region (224x224)

        Returns:
            Classification result dictionary
        """
        try:
            # Extract pose features
            features = self._extract_pose_features(person_crop)

            if features is None or self.classifier is None:
                return {'class': 'unknown', 'confidence': 0.0}

            # Classify
            prediction = self.classifier.predict([features])[0]

            # Get confidence if available
            confidence = 1.0
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba([features])[0]
                confidence = float(max(probs))

            # Map prediction to class name
            class_name = 'need_help' if prediction == 0 else 'fine'

            return {
                'class': class_name,
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {'class': 'unknown', 'confidence': 0.0}

    def _extract_pose_features(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose features from person region.

        Args:
            person_crop: Person region (224x224)

        Returns:
            Feature vector for classifier
        """
        try:
            # Run pose detection
            results = self.pose_model(person_crop, verbose=False, imgsz=YOLO_INPUT_SIZE)

            if results[0].keypoints is not None:
                # Extract keypoint coordinates
                keypoints = results[0].keypoints.xy.cpu().numpy().flatten()

                # Ensure correct feature dimension (34 for compatibility)
                if len(keypoints) >= 34:
                    features = keypoints[:34]
                else:
                    features = np.pad(keypoints, (0, 34 - len(keypoints)), 'constant')

                return features.astype(np.float32)

            return np.zeros(34, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    def _is_fall_posture(self, person_crop: np.ndarray, classification: Optional[Dict]) -> bool:
        """
        Determine if person is in fall posture.

        Args:
            person_crop: Person region
            classification: Classification result

        Returns:
            True if fall detected
        """
        if classification and classification.get('class') == 'need_help':
            return classification.get('confidence', 0) > 0.6
        return False

    def _check_sustained_fall(self, state: Dict) -> bool:
        """
        Check if fall has been sustained long enough to trigger alert.

        Args:
            state: Person state dictionary

        Returns:
            True if sustained fall detected
        """
        if len(state['fall_history']) < self.fall_confirmation_frames:
            return False

        # Check recent history
        recent_falls = sum(state['fall_history'][-self.fall_confirmation_frames:])
        threshold = self.fall_confirmation_frames * 0.7  # 70% of frames must show fall

        return recent_falls >= threshold

    def _create_alert(self, person_result: Dict) -> Dict:
        """
        Create alert message for fallen person.

        Args:
            person_result: Person detection result

        Returns:
            Alert dictionary
        """
        return {
            'timestamp': time.time(),
            'track_id': person_result['track_id'],
            'bbox': person_result['bbox'],
            'message': f"Person {person_result['track_id']} has fallen and may need help",
            'confidence': person_result.get('classification', {}).get('confidence', 0.0)
        }

    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Video frame
            results: Detection results

        Returns:
            Annotated frame
        """
        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            track_id = detection['track_id']

            # Choose color based on status
            if detection.get('alert_triggered'):
                color = (0, 0, 255)  # Red for alert
                label = f"ALERT: Person {track_id}"
            elif detection.get('is_falling'):
                color = (0, 165, 255)  # Orange for falling
                label = f"Falling: Person {track_id}"
            else:
                color = (0, 255, 0)  # Green for normal
                label = f"Person {track_id}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add label with classification info
            if detection.get('classification'):
                cls_info = detection['classification']
                label += f" [{cls_info['class']}: {cls_info['confidence']:.2f}]"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw statistics
        stats_text = f"Frame: {results['frame_id']} | Detections: {len(results['detections'])} | Alerts: {len(results['alerts'])}"
        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def process_video(self, video_path: str, output_path: Optional[str] = None, show: bool = False):
        """
        Process entire video file.

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show: Whether to display video during processing
        """
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logger.info(f"Processing video: {video_path}")
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = self.process_frame(frame)

                # Draw results
                annotated_frame = self.draw_results(frame.copy(), results)

                # Write to output
                if writer:
                    writer.write(annotated_frame)

                # Display if requested
                if show:
                    cv2.imshow('Fall Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Log alerts
                for alert in results['alerts']:
                    logger.warning(f"ALERT: {alert['message']}")

        finally:
            cap.release()
            if writer:
                writer.release()
            if show:
                cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            total_frames = self.stats['frames_processed']
            fps = total_frames / elapsed if elapsed > 0 else 0

            logger.info(f"Processing complete: {total_frames} frames in {elapsed:.2f}s ({fps:.2f} FPS)")
            logger.info(f"Statistics: {self.stats}")

    def cleanup(self):
        """Clean up resources."""
        self.person_states.clear()
        logger.info("Detector cleanup complete")


# Convenience function for quick testing
def test_unified_detector(video_path: str):
    """
    Test the unified fall detector.

    Args:
        video_path: Path to test video
    """
    from config.settings import get_settings

    settings = get_settings()
    detector = UnifiedFallDetector(settings)

    try:
        detector.process_video(video_path, show=True)
    finally:
        detector.cleanup()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_unified_detector(sys.argv[1])
    else:
        print("Usage: python fall_detector.py <video_path>")