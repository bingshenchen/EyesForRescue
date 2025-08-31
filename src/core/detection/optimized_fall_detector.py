# src/core/detection/optimized_fall_detector.py

import time
import threading
import cv2
from queue import Queue, Empty
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)
STANDARD_INPUT_SIZE = (224, 224)  # Consistent with classifier training
YOLO_INPUT_SIZE = 224  # Must be multiple of 32 for YOLO


class OptimizedFallDetector:
    """
    Optimized fall detector that solves classifier integration performance issues.

    This class addresses Issue #19 by implementing asynchronous classification processing,
    batch processing for better GPU utilization, smart classification triggering to reduce
    computational overhead, and feature caching to avoid redundant computations.
    """

    def __init__(self, settings):
        self.settings = settings

        # Performance optimization parameters
        self.classifier_call_interval = 30
        self.confidence_threshold = 0.7
        self.batch_size = 4

        # IMPROVED: Better queue management
        self.max_queue_size = 50
        self.queue_cleanup_threshold = 25

        # Caching and queuing systems
        self.classification_cache = {}
        self.pose_queue = Queue(maxsize=self.max_queue_size)
        self.classification_thread = None
        self.thread_running = False

        # Performance monitoring
        self.dropped_tasks = 0
        self.processed_tasks = 0

        # Tracking history for each detected person
        self.track_histories = {}
        self.last_classifications = {}

        # Models (will be loaded when needed)
        self.classifier = None
        self.pose_model = None

        # Start background classification thread
        self.start_classification_thread()

    def start_classification_thread(self):
        """
        Start background classification thread with improved queue management.
        """

        def classification_worker():
            batch_poses = []
            batch_track_ids = []

            while self.thread_running:
                try:
                    # IMPROVED: Handle queue overflow
                    self._manage_queue_overflow()

                    # Collect batch processing data with timeout
                    try:
                        item = self.pose_queue.get(timeout=0.1)
                        if item is None:  # Shutdown signal
                            break

                        track_id, pose_features = item
                        self.processed_tasks += 1

                        # Validate pose features
                        if pose_features is not None and len(pose_features) > 0:
                            batch_poses.append(pose_features)
                            batch_track_ids.append(track_id)

                        # Process when batch is full or timeout
                        if len(batch_poses) >= self.batch_size:
                            self.process_classification_batch(batch_poses, batch_track_ids)
                            batch_poses.clear()
                            batch_track_ids.clear()

                    except Empty:
                        # Process remaining items in batch if any
                        if batch_poses:
                            self.process_classification_batch(batch_poses, batch_track_ids)
                            batch_poses.clear()
                            batch_track_ids.clear()
                        continue

                except Exception as e:
                    logger.error(f"Classification worker error: {e}")
                    # Clear batches and continue
                    batch_poses.clear()
                    batch_track_ids.clear()
                    continue

            # Process any remaining items before shutdown
            if batch_poses:
                self.process_classification_batch(batch_poses, batch_track_ids)

        self.thread_running = True
        self.classification_thread = threading.Thread(target=classification_worker, daemon=True)
        self.classification_thread.start()
        logger.info("Classification thread started with queue management")

    def _manage_queue_overflow(self):
        """
        ADDED: Prevent memory leaks by managing queue size.
        """
        queue_size = self.pose_queue.qsize()

        if queue_size > self.max_queue_size * 0.8:  # 80% capacity warning
            logger.warning(f"Classification queue approaching capacity: {queue_size}/{self.max_queue_size}")

        if queue_size >= self.max_queue_size:
            # Drop oldest tasks to prevent memory overflow
            dropped_count = 0
            try:
                while self.pose_queue.qsize() > self.queue_cleanup_threshold:
                    self.pose_queue.get_nowait()
                    dropped_count += 1
                    self.dropped_tasks += 1

                logger.warning(f"Dropped {dropped_count} old classification tasks to prevent overflow")
            except Empty:
                pass

    def smart_fall_detection(self, frame, detections, frame_idx):
        """
        IMPROVED: Smart fall detection with better queue management.
        """
        results = []

        try:
            for detection in detections:
                track_id = detection.get('track_id', 0)
                class_name = detection.get('class_name', 'person')

                # Initialize tracking history for new tracks
                if track_id not in self.track_histories:
                    self.track_histories[track_id] = {
                        'fall_frames': deque(maxlen=30),
                        'last_classification_frame': -1,
                        'stable_classification': None,
                        'confidence_score': 0.0
                    }

                history = self.track_histories[track_id]
                is_falling = (class_name == "falling_person")
                history['fall_frames'].append(is_falling)

                # Smart classifier triggering
                try:
                    should_classify = self.should_trigger_classification(
                        track_id, frame_idx, is_falling, history
                    )

                    if should_classify:
                        # Extract pose features and add to queue with overflow protection
                        pose_features = self.extract_pose_features_fast(frame, detection)
                        if pose_features is not None:
                            # IMPROVED: Non-blocking queue add with overflow handling
                            try:
                                self.pose_queue.put_nowait((track_id, pose_features))
                                history['last_classification_frame'] = frame_idx
                                logger.debug(f"Added classification task for track {track_id}")
                            except:
                                # Queue full, manage overflow
                                self._manage_queue_overflow()
                                try:
                                    self.pose_queue.put_nowait((track_id, pose_features))
                                    history['last_classification_frame'] = frame_idx
                                except:
                                    logger.debug(f"Queue still full, skipping classification for track {track_id}")
                                    self.dropped_tasks += 1

                    # Get latest classification result
                    classification_result = self.get_latest_classification(track_id)

                    # Calculate danger level
                    danger_level = self.calculate_danger_level(
                        track_id, is_falling, classification_result, history
                    )

                    results.append({
                        'track_id': track_id,
                        'class_name': class_name,
                        'is_falling': is_falling,
                        'classification': classification_result,
                        'danger_level': danger_level,
                        'needs_help': danger_level > 0.7
                    })

                except Exception as e:
                    logger.error(f"Error processing detection for track {track_id}: {e}")
                    # Add basic result even if processing failed
                    results.append({
                        'track_id': track_id,
                        'class_name': class_name,
                        'is_falling': is_falling,
                        'classification': 'unknown',
                        'danger_level': 0.5,
                        'needs_help': False
                    })

        except Exception as e:
            logger.error(f"Error in smart_fall_detection: {e}")
            return []

        return results

    def process_classification_batch(self, poses, track_ids):
        """
        Process pose classification in batches for improved efficiency.

        Args:
            poses: List of pose feature vectors
            track_ids: Corresponding track IDs for each pose
        """
        if not poses or not self.classifier:
            return

        try:
            # Convert to numpy array for batch prediction
            poses_array = np.array(poses)

            # Ensure poses_array has the right shape
            if poses_array.ndim == 1:
                poses_array = poses_array.reshape(1, -1)

            # Batch prediction for better GPU utilization
            predictions = self.classifier.predict(poses_array)

            # Handle single prediction case
            if not isinstance(predictions, (list, np.ndarray)):
                predictions = [predictions]

            # Update classification cache
            for track_id, prediction in zip(track_ids, predictions):
                # Convert prediction to string label
                if hasattr(self.classifier, 'classes_'):
                    if isinstance(prediction, (int, np.integer)) and prediction < len(self.classifier.classes_):
                        prediction_label = self.classifier.classes_[prediction]
                    else:
                        prediction_label = "unknown"
                else:
                    prediction_label = "need_help" if prediction == 0 else "fine"

                self.last_classifications[track_id] = {
                    'prediction': prediction_label,
                    'timestamp': time.time(),
                    'confidence': 1.0
                }

        except Exception as e:
            logger.error(f"Batch classification error: {e}")
            # Continue processing without classification
            pass

    def smart_fall_detection(self, frame, detections, frame_idx):
        """
        Smart fall detection that reduces unnecessary classifier calls.
        Implements intelligent decision-making to determine when classifier invocation is necessary.

        Args:
            frame: Current video frame
            detections: YOLO detection results
            frame_idx: Current frame index

        Returns:
            List of detection results with classification and danger levels
        """
        results = []

        try:
            for detection in detections:
                track_id = detection.get('track_id', 0)
                class_name = detection.get('class_name', 'person')

                # Initialize tracking history for new tracks
                if track_id not in self.track_histories:
                    self.track_histories[track_id] = {
                        'fall_frames': deque(maxlen=30),
                        'last_classification_frame': -1,
                        'stable_classification': None,
                        'confidence_score': 0.0
                    }

                history = self.track_histories[track_id]

                # Record fall detection history
                is_falling = (class_name == "falling_person")
                history['fall_frames'].append(is_falling)

                # Smart classifier triggering strategy
                try:
                    should_classify = self.should_trigger_classification(
                        track_id, frame_idx, is_falling, history
                    )

                    if should_classify:
                        # Extract pose features and add to queue
                        pose_features = self.extract_pose_features_fast(frame, detection)
                        if pose_features is not None:
                            try:
                                self.pose_queue.put_nowait((track_id, pose_features))
                                history['last_classification_frame'] = frame_idx
                            except:
                                pass

                    # Get latest classification result
                    classification_result = self.get_latest_classification(track_id)

                    # Calculate final danger level
                    danger_level = self.calculate_danger_level(
                        track_id, is_falling, classification_result, history
                    )

                    results.append({
                        'track_id': track_id,
                        'class_name': class_name,
                        'is_falling': is_falling,
                        'classification': classification_result,
                        'danger_level': danger_level,
                        'needs_help': danger_level > 0.7
                    })

                except Exception as e:
                    logger.error(f"Error processing detection for track {track_id}: {e}")
                    # Add a basic result even if processing failed
                    results.append({
                        'track_id': track_id,
                        'class_name': class_name,
                        'is_falling': is_falling,
                        'classification': 'unknown',
                        'danger_level': 0.5,
                        'needs_help': False
                    })

        except Exception as e:
            logger.error(f"Error in smart_fall_detection: {e}")
            # Return empty results if everything fails
            return []

        return results

    def should_trigger_classification(self, track_id, frame_idx, is_falling, history):
        """
        Intelligent decision-making for classifier invocation.
        Implements smart triggering logic to reduce unnecessary classifier calls.

        Args:
            track_id: Unique identifier for the tracked person
            frame_idx: Current frame index
            is_falling: Whether current detection shows falling
            history: Historical tracking data for this person

        Returns:
            bool: Whether to trigger classification for this detection
        """

        # Skip if recently classified
        if frame_idx - history['last_classification_frame'] < self.classifier_call_interval:
            return False

        # Reduce classification frequency when no fall detected
        if not is_falling:
            # Convert deque to list and get recent falls
            fall_frames_list = list(history['fall_frames'])
            recent_falls = sum(fall_frames_list[-10:]) if len(fall_frames_list) >= 10 else sum(fall_frames_list)
            if recent_falls == 0:
                return False

        # Reduce frequency if we have stable classification results
        if history['stable_classification'] and history['confidence_score'] > 0.8:
            consecutive_falls = self.count_consecutive_falls(history['fall_frames'])
            if consecutive_falls < 5:
                return False

        # Prioritize newly detected falls
        if is_falling:
            fall_frames_list = list(history['fall_frames'])
            recent_fall_count = sum(fall_frames_list[-5:]) if len(fall_frames_list) >= 5 else sum(fall_frames_list)
            if recent_fall_count >= 3:
                return True

        return True

    def extract_pose_features_fast(self, frame, detection):
        """
        Fast pose feature extraction with consistent preprocessing.
        FIXED: Use standard 224x224 input size for consistency.

        Args:
            frame: Current video frame
            detection: Detection bounding box information

        Returns:
            numpy.ndarray: Extracted pose features or None if extraction fails
        """
        try:
            # Extract bounding box
            bbox = detection.get('bbox', (0, 0, 100, 100))
            x1, y1, x2, y2 = bbox

            # Ensure valid bounding box
            if x2 <= x1 or y2 <= y1:
                logger.debug("Invalid bounding box dimensions")
                return None

            # Validate coordinates against frame dimensions
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, min(x1, frame_w))
            y1 = max(0, min(y1, frame_h))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))

            # Extract person crop
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                logger.debug("Empty person crop")
                return None

            # FIXED: Use consistent input size (224x224)
            person_crop = cv2.resize(person_crop, STANDARD_INPUT_SIZE)
            person_crop = person_crop.astype(np.uint8)

            # Pose detection with proper YOLO input size
            if self.pose_model is not None:
                results = self.pose_model(person_crop, verbose=False, imgsz=YOLO_INPUT_SIZE)

                # Feature extraction
                if results[0].keypoints is not None:
                    keypoints = results[0].keypoints.xy.cpu().numpy().flatten()
                    # Ensure exactly 34 features for classifier compatibility
                    if len(keypoints) >= 34:
                        features = keypoints[:34]
                    else:
                        # Pad to 34 features if fewer detected
                        features = np.pad(keypoints, (0, 34 - len(keypoints)), 'constant', constant_values=0)

                    logger.debug(f"Extracted {len(features)} features")
                    return features.astype(np.float32)

            # Return zero vector if no keypoints detected
            logger.debug("No keypoints detected, returning zero features")
            return np.zeros(34, dtype=np.float32)

        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return None

    def get_latest_classification(self, track_id):
        """
        Retrieve the most recent classification result for a track.

        Args:
            track_id: Unique identifier for the tracked person

        Returns:
            str: Latest classification result or "unknown" if none available
        """
        if track_id in self.last_classifications:
            classification = self.last_classifications[track_id]
            # Check if result is still valid (not older than 5 seconds)
            if time.time() - classification['timestamp'] < 5.0:
                return classification['prediction']

        return "unknown"

    def calculate_danger_level(self, track_id, is_falling, classification, history):
        """
        Calculate danger level by combining multiple factors.
        Provides comprehensive risk assessment by considering fall detection ratio,
        classification results, and duration of falling state.

        Args:
            track_id: Unique identifier for the tracked person
            is_falling: Current fall detection state
            classification: Latest classification result
            history: Historical tracking data

        Returns:
            float: Danger level between 0.0 and 1.0
        """
        danger_score = 0.0

        # Fall detection weight (40%)
        fall_frames_list = list(history['fall_frames'])
        if fall_frames_list:
            fall_ratio = sum(fall_frames_list) / len(fall_frames_list)
            danger_score += fall_ratio * 0.4

        # Classifier result weight (35%)
        if classification == "need_help":
            danger_score += 0.35
        elif classification == "fine":
            danger_score -= 0.1

        # Duration weight (25%)
        consecutive_falls = self.count_consecutive_falls(history['fall_frames'])
        duration_score = min(consecutive_falls / 30.0, 1.0)
        danger_score += duration_score * 0.25

        return max(0.0, min(1.0, danger_score))

    def count_consecutive_falls(self, fall_frames):
        """
        Count consecutive fall detection frames.

        Args:
            fall_frames: Deque of recent fall detection results

        Returns:
            int: Number of consecutive frames with fall detection
        """
        consecutive = 0
        fall_list = list(fall_frames)
        for is_fall in reversed(fall_list):
            if is_fall:
                consecutive += 1
            else:
                break
        return consecutive

    def get_performance_stats(self):
        """
        IMPROVED: Get comprehensive performance statistics.
        """
        return {
            'classification_queue_size': self.pose_queue.qsize(),
            'cached_classifications': len(self.last_classifications),
            'tracked_objects': len(self.track_histories),
            'classification_thread_alive': self.classification_thread.is_alive() if self.classification_thread else False,
            'processed_tasks': self.processed_tasks,
            'dropped_tasks': self.dropped_tasks,
            'queue_utilization': f"{self.pose_queue.qsize()}/{self.max_queue_size}"
        }

    def cleanup(self):
        """
        IMPROVED: Enhanced cleanup with better resource management.
        """
        logger.info("Starting optimized fall detector cleanup...")

        # Stop thread
        self.thread_running = False

        # Clear queue and send shutdown signal
        try:
            # Clear existing items
            while not self.pose_queue.empty():
                try:
                    self.pose_queue.get_nowait()
                except:
                    break

            # Send shutdown signal
            self.pose_queue.put_nowait(None)
        except:
            pass

        # Wait for thread to finish
        if self.classification_thread and self.classification_thread.is_alive():
            self.classification_thread.join(timeout=2.0)
            if self.classification_thread.is_alive():
                logger.warning("Classification thread did not shut down gracefully")

        # Clear caches
        self.classification_cache.clear()
        self.last_classifications.clear()
        self.track_histories.clear()

        logger.info(f"Cleanup completed. Stats - Processed: {self.processed_tasks}, Dropped: {self.dropped_tasks}")


class AlertSystem:
    """Simple alert system for fall detection notifications."""

    def __init__(self):
        self.alert_count = 0

    def trigger_alert(self, detection_result):
        """
        Trigger an alert based on detection result.

        Args:
            detection_result: Dictionary containing detection information
        """
        self.alert_count += 1
        logger.info(f"ALERT #{self.alert_count}: Fall detected for track {detection_result['track_id']}")
        logger.info(f"  Classification: {detection_result['classification']}")
        logger.info(f"  Danger Level: {detection_result['danger_level']:.2f}")


class IntegratedFallDetectionSystem:
    """
    Integrated fall detection system example showing optimized workflow.
    Demonstrates how to integrate the optimized fall detector into a complete system.
    """

    def __init__(self):
        try:
            from config.settings import get_settings
            settings = get_settings()
        except ImportError:
            settings = None

        self.detector = OptimizedFallDetector(settings)
        self.alert_system = AlertSystem()

    def process_frame(self, frame, frame_idx):
        """
        Process a single frame with optimized workflow.

        Args:
            frame: Current video frame
            frame_idx: Frame index for tracking

        Returns:
            list: Detection results with classifications
        """

        detections = []
        results = self.detector.smart_fall_detection(frame, detections, frame_idx)

        # Alert processing
        for result in results:
            if result['needs_help']:
                self.alert_system.trigger_alert(result)

        # Performance monitoring
        if frame_idx % 100 == 0:
            stats = self.detector.get_performance_stats()
            logger.info(f"Performance: {stats}")

        return results
