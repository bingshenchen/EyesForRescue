# src/core/detection/fall_detector.py

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
from src.core.analysis.gpt_analyzer import analyze_image
from src.core.analysis.location_service import getLoc
from src.core.analysis.weather_service import get_weather

logger = logging.getLogger(__name__)

# Constants
STANDARD_INPUT_SIZE = (224, 224)
YOLO_INPUT_SIZE = 224
PERSON_CLASS_ID = 0


class UnifiedFallDetector:
    """
    Enhanced fall detection system with all fixes integrated.
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
        self.classifier_interval = 10
        self.classifier_stability_window = 5  # FIX 1: Add stability window

        # Load models
        self._load_models()

        # Initialize components
        self.tracker = self._init_tracker()
        self.cache = DetectionCache(cache_dir=self.settings.CACHE_DIR)

        # Enhanced state management
        self.person_states = {}  # Track each person's state
        self.danger_values = {}  # FIX 2: Track danger values per person
        self.gpt_analysis_cache = {}  # Cache GPT analysis results
        self.frame_count = 0

        # FIX 3: Track persistent IDs across frames
        self.id_mapping = {}  # Map temporary IDs to persistent IDs
        self.next_persistent_id = 1

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

    def _get_persistent_id(self, track_id: int) -> int:
        """
        FIX 3: Map temporary track IDs to persistent IDs
        """
        if track_id not in self.id_mapping:
            self.id_mapping[track_id] = self.next_persistent_id
            self.next_persistent_id += 1
        return self.id_mapping[track_id]

    def _stabilize_classification(self, track_id: int, new_classification: str, confidence: float) -> Tuple[str, float]:
        """
        FIX 1: Stabilize classifier to prevent oscillation between 'ok' and 'help'
        Uses a voting window to smooth rapid changes
        """
        if track_id not in self.person_states:
            return new_classification, confidence

        state = self.person_states[track_id]

        # Initialize classification history if needed
        if 'classification_history' not in state:
            state['classification_history'] = deque(maxlen=self.classifier_stability_window)

        # Add new classification to history
        state['classification_history'].append({
            'class': new_classification,
            'confidence': confidence
        })

        # If we don't have enough history yet, return current
        if len(state['classification_history']) < 3:
            return new_classification, confidence

        # Count votes in the window
        votes = {'need_help': 0, 'fine': 0}
        total_confidence = {'need_help': 0.0, 'fine': 0.0}

        for entry in state['classification_history']:
            votes[entry['class']] += 1
            total_confidence[entry['class']] += entry['confidence']

        # Determine stable classification (requires majority)
        if votes['need_help'] > votes['fine']:
            stable_class = 'need_help'
            avg_confidence = total_confidence['need_help'] / max(votes['need_help'], 1)
        else:
            stable_class = 'fine'
            avg_confidence = total_confidence['fine'] / max(votes['fine'], 1)

        # Only change if there's strong consensus
        threshold_votes = len(state['classification_history']) * 0.6
        if votes[stable_class] >= threshold_votes:
            return stable_class, avg_confidence
        else:
            # Keep previous stable state
            return state.get('last_stable_class', 'fine'), state.get('last_stable_confidence', 0.5)

    def _calculate_danger_value(self, person_state: Dict, frame: np.ndarray, bbox: Tuple) -> float:
        """
        FIX 2: Calculate danger value based on multiple factors
        """
        persistent_id = person_state.get('persistent_id', 0)

        # Initialize danger value for new person
        if persistent_id not in self.danger_values:
            self.danger_values[persistent_id] = {
                'current': 0.0,
                'history': deque(maxlen=30),
                'gpt_analyzed': False,
                'gpt_result': None,
                'location': None,
                'weather': None
            }

        danger_data = self.danger_values[persistent_id]

        # Base danger from fall detection
        base_danger = 0.0
        if person_state.get('classification', {}).get('class') == 'need_help':
            base_danger = 0.5

        # Add danger from static duration
        static_frames = person_state.get('static_frames', 0)
        if static_frames > 0:
            # Increase danger for prolonged static state
            duration_danger = min(static_frames / 180.0, 0.3)  # Max 0.3 after 3 seconds at 60fps
            base_danger += duration_danger

        # Add danger from fall history
        fall_history = person_state.get('fall_history', deque())
        if len(fall_history) > 0:
            fall_ratio = sum(fall_history) / len(fall_history)
            base_danger += fall_ratio * 0.2

        # Update danger history
        danger_data['history'].append(base_danger)

        # Calculate smoothed danger value
        if len(danger_data['history']) > 0:
            danger_data['current'] = np.mean(danger_data['history'])
        else:
            danger_data['current'] = base_danger

        return danger_data['current']

    def _trigger_emergency_analysis(self, frame: np.ndarray, bbox: Tuple, persistent_id: int):
        """
        FIX 2: Trigger GPT analysis when danger > 1.0
        """
        if persistent_id not in self.danger_values:
            return

        danger_data = self.danger_values[persistent_id]

        # Only analyze once per incident
        if danger_data['gpt_analyzed']:
            return

        try:
            # Extract person region
            x1, y1, x2, y2 = bbox
            person_region = frame[y1:y2, x1:x2]

            # Save temporary image for analysis
            temp_path = Path("temp_analysis.jpg")
            cv2.imwrite(str(temp_path), person_region)

            # Get GPT analysis
            logger.info(f"Triggering GPT analysis for person {persistent_id}")
            gpt_result = analyze_image(str(temp_path))
            danger_data['gpt_result'] = gpt_result

            # Get location
            try:
                lat, lon = getLoc()
                danger_data['location'] = {'latitude': lat, 'longitude': lon}
            except:
                logger.warning("Could not get location")

            # Get weather
            try:
                if danger_data['location']:
                    temp, weather_code, time = get_weather(
                        danger_data['location']['latitude'],
                        danger_data['location']['longitude']
                    )
                    danger_data['weather'] = {
                        'temperature': temp,
                        'code': weather_code,
                        'time': time
                    }
            except:
                logger.warning("Could not get weather")

            danger_data['gpt_analyzed'] = True

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

        except Exception as e:
            logger.error(f"Emergency analysis failed: {e}")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame with all fixes integrated.
        """
        self.frame_count += 1
        results = {
            'frame_id': self.frame_count,
            'detections': [],
            'alerts': [],
            'danger_info': {},  # FIX 2: Add danger info
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

                    # FIX 2: Check danger level
                    persistent_id = person_result.get('persistent_id')
                    danger_value = person_result.get('danger_value', 0.0)

                    if danger_value > 1.0 and persistent_id:
                        self._trigger_emergency_analysis(frame, person_result['bbox'], persistent_id)

                        # Add emergency info to results
                        if persistent_id in self.danger_values:
                            danger_data = self.danger_values[persistent_id]
                            if danger_data['gpt_analyzed']:
                                results['danger_info'][persistent_id] = {
                                    'gpt_analysis': danger_data['gpt_result'],
                                    'location': danger_data['location'],
                                    'weather': danger_data['weather'],
                                    'danger_value': danger_value
                                }

            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['detections'] += len(results['detections'])
            results['stats'] = self.stats.copy()

        except Exception as e:
            logger.error(f"Frame processing error: {e}")

        return results

    def _detect_persons(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons in frame using YOLO."""
        detections = []

        try:
            results = self.yolo_model(frame, imgsz=320, verbose=False)

            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

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
        """Track detected persons using SORT tracker."""
        if not detections:
            return []

        det_array = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            det_array.append([x1, y1, x2, y2, 0])

        tracks = self.tracker.update(det_array)
        return tracks

    def _process_person(self, frame: np.ndarray, track: np.ndarray) -> Optional[Dict]:
        """Process individual tracked person with all fixes."""
        x1, y1, x2, y2, track_id, cls = map(int, track)

        # FIX 3: Get persistent ID
        persistent_id = self._get_persistent_id(track_id)

        # Initialize person state if new
        if track_id not in self.person_states:
            self.person_states[track_id] = {
                'persistent_id': persistent_id,  # FIX 3
                'fall_history': deque(maxlen=30),
                'last_classification': None,
                'last_classification_frame': 0,
                'last_stable_class': 'fine',  # FIX 1
                'last_stable_confidence': 0.5,  # FIX 1
                'fall_start_frame': None,
                'alert_sent': False,
                'static_frames': 0,
                'last_position': None
            }

        state = self.person_states[track_id]

        # Check if person is static
        current_position = ((x1 + x2) / 2, (y1 + y2) / 2)
        if state['last_position']:
            movement = np.sqrt(
                (current_position[0] - state['last_position'][0]) ** 2 +
                (current_position[1] - state['last_position'][1]) ** 2
            )
            if movement < 10:  # Threshold for static detection
                state['static_frames'] += 1
            else:
                state['static_frames'] = 0
        state['last_position'] = current_position

        # Extract person bounding box (CRITICAL FIX from original)
        person_crop = self._extract_person_region(frame, (x1, y1, x2, y2))

        if person_crop is None:
            return None

        # Classify if needed
        classification = None
        if self.classifier and (self.frame_count - state['last_classification_frame']) >= self.classifier_interval:
            raw_classification = self._classify_person(person_crop)

            # FIX 1: Stabilize classification
            stable_class, stable_confidence = self._stabilize_classification(
                track_id,
                raw_classification['class'],
                raw_classification['confidence']
            )

            classification = {
                'class': stable_class,
                'confidence': stable_confidence,
                'raw_class': raw_classification['class']
            }

            state['last_classification'] = classification
            state['last_stable_class'] = stable_class
            state['last_stable_confidence'] = stable_confidence
            state['last_classification_frame'] = self.frame_count
            self.stats['classifications'] += 1
        else:
            classification = state['last_classification']

        # Update fall history
        is_fall = self._is_fall_posture(person_crop, classification)
        state['fall_history'].append(is_fall)

        # FIX 2: Calculate danger value
        state['classification'] = classification
        danger_value = self._calculate_danger_value(state, frame, (x1, y1, x2, y2))

        # Check for sustained fall
        alert_triggered = False
        if self._check_sustained_fall(state):
            if not state['alert_sent']:
                alert_triggered = True
                state['alert_sent'] = True
                self.stats['alerts'] += 1
                logger.warning(f"ALERT: Person {persistent_id} has fallen and needs help!")
        else:
            state['alert_sent'] = False

        return {
            'track_id': track_id,
            'persistent_id': persistent_id,  # FIX 3
            'bbox': (x1, y1, x2, y2),
            'classification': classification,
            'is_falling': is_fall,
            'fall_duration': sum(state['fall_history']),
            'static_duration': state['static_frames'],
            'danger_value': danger_value,  # FIX 2
            'alert_triggered': alert_triggered
        }

    def _extract_person_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract person region from frame."""
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
        """Classify person pose using the trained classifier."""
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
        """Extract pose features from person region."""
        try:
            # Run pose detection on person crop
            results = self.pose_model(person_crop, verbose=False)

            if results and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()
                if keypoints.shape[0] > 0:
                    # Flatten keypoints to feature vector
                    features = keypoints[0].flatten()
                    return features

            return None

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def _is_fall_posture(self, person_crop: np.ndarray, classification: Optional[Dict]) -> bool:
        """Determine if person is in fall posture."""
        if classification and classification.get('class') == 'need_help':
            return True
        return False

    def _check_sustained_fall(self, state: Dict) -> bool:
        """Check if fall has been sustained long enough to trigger alert."""
        if len(state['fall_history']) < self.fall_confirmation_frames:
            return False

        recent_falls = list(state['fall_history'])[-self.fall_confirmation_frames:]
        fall_ratio = sum(recent_falls) / len(recent_falls)

        return fall_ratio > 0.7

    def _create_alert(self, person_result: Dict) -> Dict:
        """Create alert dictionary with all relevant information."""
        return {
            'timestamp': time.time(),
            'track_id': person_result['track_id'],
            'persistent_id': person_result['persistent_id'],
            'bbox': person_result['bbox'],
            'danger_value': person_result.get('danger_value', 0.0),
            'message': f"Person {person_result['persistent_id']} has fallen and may need help",
            'confidence': person_result.get('classification', {}).get('confidence', 0.0)
        }

    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw enhanced detection results on frame."""
        # Draw danger info panel if available
        if results.get('danger_info'):
            self._draw_danger_panel(frame, results['danger_info'])

        for detection in results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            persistent_id = detection['persistent_id']  # FIX 3: Use persistent ID
            danger_value = detection.get('danger_value', 0.0)

            # Choose color based on danger level
            if danger_value > 1.0:
                color = (0, 0, 255)  # Red for high danger
                label = f"DANGER: Person {persistent_id}"
            elif detection.get('alert_triggered'):
                color = (0, 100, 255)  # Orange-red for alert
                label = f"ALERT: Person {persistent_id}"
            elif detection.get('is_falling'):
                color = (0, 165, 255)  # Orange for falling
                label = f"Falling: Person {persistent_id}"
            else:
                color = (0, 255, 0)  # Green for normal
                label = f"Person {persistent_id}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add detailed label
            if detection.get('classification'):
                cls_info = detection['classification']
                label += f" [{cls_info['class']}: {cls_info['confidence']:.2f}]"

            # Add danger value
            label += f" Danger: {danger_value:.2f}"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw static duration if significant
            if detection.get('static_duration', 0) > 30:  # More than 0.5 seconds at 60fps
                static_text = f"Static: {detection['static_duration'] / 60:.1f}s"
                cv2.putText(frame, static_text, (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw statistics
        stats_text = f"Frame: {results['frame_id']} | Detections: {len(results['detections'])} | Alerts: {len(results['alerts'])}"
        cv2.putText(frame, stats_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def _draw_danger_panel(self, frame: np.ndarray, danger_info: Dict):
        """
        FIX 2: Draw danger information panel in top-right corner
        """
        if not danger_info:
            return

        # Get frame dimensions
        h, w = frame.shape[:2]

        # Panel dimensions
        panel_width = 300
        panel_height = 200
        margin = 10

        # Panel position (top-right)
        panel_x = w - panel_width - margin
        panel_y = margin

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw border
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (0, 0, 255), 2)

        # Title
        cv2.putText(frame, "EMERGENCY ANALYSIS",
                    (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        y_offset = panel_y + 50

        # Display info for each person with high danger
        for person_id, info in danger_info.items():
            cv2.putText(frame, f"Person {person_id}:",
                        (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

            # GPT Analysis
            if info.get('gpt_analysis'):
                gpt = info['gpt_analysis'].get('gpt_analysis', {})
                status = gpt.get('status', ['unknown'])
                cv2.putText(frame, f"Status: {', '.join(status)}",
                            (panel_x + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15

            # Location
            if info.get('location'):
                loc = info['location']
                cv2.putText(frame, f"Location: {loc['latitude']:.4f}, {loc['longitude']:.4f}",
                            (panel_x + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15

            # Weather
            if info.get('weather'):
                weather = info['weather']
                cv2.putText(frame, f"Weather: {weather['temperature']}°C, Code: {weather['code']}",
                            (panel_x + 20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 15

            # Danger value
            cv2.putText(frame, f"Danger Level: {info['danger_value']:.2f}",
                        (panel_x + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            y_offset += 25

    def process_video(self, video_path: str, output_path: Optional[str] = None, show: bool = False):
        """Process entire video file with all enhancements."""
        cap = cv2.VideoCapture(video_path)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        else:
            out = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = self.process_frame(frame)

                # Draw results
                annotated_frame = self.draw_results(frame, results)

                # Write output
                if out:
                    out.write(annotated_frame)

                # Display if requested
                if show:
                    cv2.imshow('Fall Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            if out:
                out.release()
            if show:
                cv2.destroyAllWindows()

        logger.info(f"Video processing complete. Stats: {self.stats}")


# For backward compatibility
FallDetector = UnifiedFallDetector