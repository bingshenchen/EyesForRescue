# src/core/detection/enhanced_fall_detector.py

import cv2
import numpy as np
import logging
from collections import defaultdict, deque
import time

from .improved_yolo_detector import ImprovedYOLODetector
from ..analysis.smart_pose_analyzer import SmartPoseAnalyzer, PostureType

logger = logging.getLogger(__name__)


class EnhancedFallDetector:
    """
    Enhanced fall detector with improved person detection and false positive reduction
    """

    def __init__(self, settings):
        self.settings = settings

        # Initialize components
        self.yolo_detector = ImprovedYOLODetector(
            settings.YOLO_MODEL_PATH,
            settings
        )
        self.pose_analyzer = SmartPoseAnalyzer()

        # Tracking and state management
        self.person_states = {}
        self.alert_cooldowns = {}
        self.detection_history = defaultdict(lambda: deque(maxlen=30))

        # Alert parameters
        self.fall_confirmation_frames = 15  # Frames to confirm fall
        self.alert_cooldown_time = 30.0  # Seconds between alerts

    def process_frame(self, frame, frame_idx):
        """
        Process single frame with enhanced detection and false positive filtering
        """
        results = {
            'detections': [],
            'alerts': [],
            'debug_info': {
                'total_detections': 0,
                'valid_detections': 0,
                'filtered_detections': 0,
                'active_alerts': 0
            }
        }

        try:
            # Step 1: Multi-scale person detection
            detections = self.yolo_detector.detect_with_multi_scale(frame)
            results['debug_info']['total_detections'] = len(detections)

            if not detections:
                logger.debug(f"Frame {frame_idx}: No persons detected")
                return results

            # Step 2: Pose analysis and context understanding
            valid_detections = []

            for detection in detections:
                enhanced_detection = self.analyze_detection_context(
                    frame, detection, frame_idx
                )

                if enhanced_detection:
                    valid_detections.append(enhanced_detection)

            results['detections'] = valid_detections
            results['debug_info']['valid_detections'] = len(valid_detections)

            # Step 3: Fall state tracking and alert generation
            alerts = self.process_fall_alerts(valid_detections, frame_idx)
            results['alerts'] = alerts
            results['debug_info']['active_alerts'] = len(alerts)

            return results

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return results

    def analyze_detection_context(self, frame, detection, frame_idx):
        """
        Analyze detection context to reduce false positives
        """
        try:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            # Extract person region
            person_region = frame[y1:y2, x1:x2]
            if person_region.size == 0:
                return None

            # Get pose keypoints (simplified - you'd use actual pose model here)
            pose_keypoints = self.extract_pose_keypoints(person_region)

            # Analyze person context
            context_analysis = self.pose_analyzer.analyze_person_context(
                frame, bbox, pose_keypoints
            )

            # Create enhanced detection result
            enhanced_detection = {
                'bbox': bbox,
                'confidence': detection['confidence'],
                'context': context_analysis,
                'pose_keypoints': pose_keypoints,
                'frame_idx': frame_idx,
                'person_id': self.generate_person_id(bbox)  # Simple ID based on position
            }

            # Filter based on context analysis
            if self.should_filter_detection(context_analysis):
                logger.debug(f"Detection filtered: {context_analysis['posture_type']}")
                return None

            return enhanced_detection

        except Exception as e:
            logger.debug(f"Context analysis error: {e}")
            return None

    def should_filter_detection(self, context_analysis):
        """
        Determine if detection should be filtered as false positive
        """
        posture_type = context_analysis.get('posture_type', PostureType.UNKNOWN)
        furniture_detected = context_analysis.get('furniture_detected', False)
        confidence = context_analysis.get('confidence', 0.0)

        # Filter obvious non-emergencies
        if posture_type == PostureType.SITTING and furniture_detected:
            return True

        if posture_type == PostureType.LYING_ON_FURNITURE and furniture_detected:
            # Additional check for intentional lying
            transition = context_analysis.get('transition_analysis', {})
            if transition.get('transition_type') == 'gradual_intentional':
                return True

        if posture_type == PostureType.STANDING:
            return True  # No need to track standing people for falls

        # Keep detections with low confidence for further analysis
        if confidence < 0.3:
            return True

        return False

    def extract_pose_keypoints(self, person_region):
        """
        Extract pose keypoints from person region
        Note: This is a placeholder - you'd use your actual pose model
        """
        try:
            # Resize to standard size
            resized = cv2.resize(person_region, (224, 224))

            # Placeholder: return dummy keypoints
            # In real implementation, use your pose model here
            dummy_keypoints = np.random.rand(34) * 224
            return dummy_keypoints.astype(np.float32)

        except Exception as e:
            logger.debug(f"Pose extraction error: {e}")
            return np.zeros(34, dtype=np.float32)

    def generate_person_id(self, bbox):
        """
        Generate consistent person ID based on bounding box location
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Simple spatial binning for ID consistency
        grid_x = int(center_x // 50)
        grid_y = int(center_y // 50)

        return f"person_{grid_x}_{grid_y}"

    def process_fall_alerts(self, detections, frame_idx):
        """
        Process detections to generate fall alerts with confirmation
        """
        alerts = []
        current_time = time.time()

        for detection in detections:
            person_id = detection['person_id']
            context = detection['context']
            posture_type = context.get('posture_type', PostureType.UNKNOWN)

            # Track detection history
            self.detection_history[person_id].append({
                'frame_idx': frame_idx,
                'posture_type': posture_type,
                'context': context
            })

            # Check for fall patterns
            if self.is_fall_confirmed(person_id, detection):
                # Check cooldown
                last_alert_time = self.alert_cooldowns.get(person_id, 0)
                if current_time - last_alert_time > self.alert_cooldown_time:
                    alert = {
                        'person_id': person_id,
                        'bbox': detection['bbox'],
                        'alert_type': 'fall_detected',
                        'confidence': context.get('confidence', 0.0),
                        'frame_idx': frame_idx,
                        'context_summary': self.generate_context_summary(context),
                        'timestamp': current_time
                    }

                    alerts.append(alert)
                    self.alert_cooldowns[person_id] = current_time

                    logger.warning(f"FALL ALERT: {person_id} at frame {frame_idx}")

        return alerts

    def is_fall_confirmed(self, person_id, detection):
        """
        Confirm if detection represents a confirmed fall
        """
        history = self.detection_history[person_id]

        if len(history) < self.fall_confirmation_frames:
            return False

        # Analyze recent detection history
        recent_history = list(history)[-self.fall_confirmation_frames:]

        # Count frames with fall indicators
        fall_frames = 0
        for record in recent_history:
            posture_type = record['context'].get('posture_type', PostureType.UNKNOWN)
            furniture_detected = record['context'].get('furniture_detected', False)

            # Consider as fall if:
            # 1. Posture is FALLEN
            # 2. Lying without furniture support
            if (posture_type == PostureType.FALLEN or
                    (posture_type == PostureType.LYING_ON_FURNITURE and not furniture_detected)):
                fall_frames += 1

        # Require majority of frames to indicate fall
        fall_ratio = fall_frames / len(recent_history)
        return fall_ratio > 0.6

    def generate_context_summary(self, context):
        """
        Generate human-readable context summary for alerts
        """
        posture_type = context.get('posture_type', PostureType.UNKNOWN)
        furniture_detected = context.get('furniture_detected', False)
        confidence = context.get('confidence', 0.0)

        summary = f"Posture: {posture_type.value}"

        if furniture_detected:
            summary += " (furniture nearby)"
        else:
            summary += " (on ground)"

        summary += f" | Confidence: {confidence:.2f}"

        transition = context.get('transition_analysis', {})
        if transition:
            summary += f" | Transition: {transition.get('transition_type', 'unknown')}"

        return summary

    def get_detection_stats(self):
        """
        Get detection statistics for monitoring
        """
        return {
            'active_persons': len(self.detection_history),
            'total_alerts_sent': len(self.alert_cooldowns),
            'persons_on_cooldown': sum(
                1 for t in self.alert_cooldowns.values()
                if time.time() - t < self.alert_cooldown_time
            )
        }