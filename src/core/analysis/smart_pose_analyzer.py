# src/core/analysis/smart_pose_analyzer.py

import cv2
import numpy as np
import logging
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class PostureType(Enum):
    STANDING = "standing"
    SITTING = "sitting"
    LYING_ON_FURNITURE = "lying_on_furniture"
    FALLEN = "fallen"
    UNKNOWN = "unknown"


class SmartPoseAnalyzer:
    """
    Advanced pose analyzer that distinguishes between intentional lying and falls
    """

    def __init__(self):
        # Historical pose tracking
        self.pose_history = {}
        self.transition_history = {}

        # Furniture/object detection parameters
        self.furniture_colors = [
            ([100, 50, 50], [130, 255, 255]),  # Chair colors in HSV
            ([20, 50, 50], [30, 255, 255]),  # Wood colors
            ([0, 0, 100], [180, 30, 255])  # Light colors (white/gray furniture)
        ]

    def analyze_person_context(self, frame, person_bbox, pose_keypoints):
        """
        Analyze person's context to determine if lying position is intentional
        """
        x1, y1, x2, y2 = person_bbox
        person_region = frame[y1:y2, x1:x2]

        context_analysis = {
            'furniture_detected': False,
            'ground_contact': True,
            'posture_type': PostureType.UNKNOWN,
            'confidence': 0.0,
            'transition_analysis': None
        }

        # 1. Furniture Detection
        furniture_detected = self.detect_furniture_support(person_region)
        context_analysis['furniture_detected'] = furniture_detected

        # 2. Pose Classification
        posture_type, confidence = self.classify_posture(pose_keypoints, furniture_detected)
        context_analysis['posture_type'] = posture_type
        context_analysis['confidence'] = confidence

        # 3. Transition Analysis
        transition_analysis = self.analyze_pose_transition(person_bbox, posture_type)
        context_analysis['transition_analysis'] = transition_analysis

        # 4. Ground Contact Analysis
        ground_contact = self.analyze_ground_contact(person_region, pose_keypoints)
        context_analysis['ground_contact'] = ground_contact

        return context_analysis

    def detect_furniture_support(self, person_region):
        """
        Detect if person is supported by furniture (chair, bench, etc.)
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)

            furniture_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

            # Check for furniture colors
            for lower, upper in self.furniture_colors:
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)
                furniture_mask = cv2.bitwise_or(furniture_mask, mask)

            # Look for structured furniture shapes (horizontal lines for chairs/benches)
            edges = cv2.Canny(furniture_mask, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                    minLineLength=30, maxLineGap=10)

            if lines is not None:
                horizontal_lines = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 15 or angle > 165:  # Nearly horizontal
                        horizontal_lines += 1

                # Furniture typically has horizontal support structures
                if horizontal_lines >= 2:
                    logger.debug(f"Furniture detected: {horizontal_lines} horizontal lines")
                    return True

            # Check furniture mask coverage
            furniture_ratio = np.sum(furniture_mask > 0) / furniture_mask.size
            if furniture_ratio > 0.15:  # 15% furniture color coverage
                logger.debug(f"Furniture detected by color: {furniture_ratio:.2f} coverage")
                return True

            return False

        except Exception as e:
            logger.debug(f"Furniture detection error: {e}")
            return False

    def classify_posture(self, pose_keypoints, furniture_support):
        """
        Classify person's posture based on keypoints and context
        """
        try:
            if pose_keypoints is None or len(pose_keypoints) < 34:
                return PostureType.UNKNOWN, 0.0

            # Reshape keypoints (17 points * 2 coordinates)
            keypoints = pose_keypoints[:34].reshape(17, 2)

            # Key body points
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_knee = keypoints[13]
            right_knee = keypoints[14]

            # Calculate body orientation
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2

            # Vertical body axis
            if np.linalg.norm(shoulder_center - hip_center) < 10:
                return PostureType.UNKNOWN, 0.0

            body_angle = np.arctan2(
                hip_center[1] - shoulder_center[1],
                hip_center[0] - shoulder_center[0]
            ) * 180 / np.pi

            # Normalize angle to [0, 180]
            body_angle = abs(body_angle)
            if body_angle > 90:
                body_angle = 180 - body_angle

            # Knee position analysis
            knee_height_diff = abs(left_knee[1] - right_knee[1])
            avg_knee_y = (left_knee[1] + right_knee[1]) / 2
            hip_y = hip_center[1]

            # Posture classification logic
            if body_angle < 25:  # Nearly vertical
                if furniture_support:
                    return PostureType.SITTING, 0.8
                else:
                    return PostureType.STANDING, 0.9

            elif body_angle > 60:  # Nearly horizontal
                if furniture_support:
                    return PostureType.LYING_ON_FURNITURE, 0.85
                else:
                    # Additional checks for intentional vs fall
                    if self.is_controlled_lying(keypoints):
                        return PostureType.LYING_ON_FURNITURE, 0.7
                    else:
                        return PostureType.FALLEN, 0.8
            else:  # Intermediate angle
                if furniture_support and body_angle > 35:
                    return PostureType.SITTING, 0.6
                else:
                    return PostureType.UNKNOWN, 0.3

        except Exception as e:
            logger.debug(f"Posture classification error: {e}")
            return PostureType.UNKNOWN, 0.0

    def is_controlled_lying(self, keypoints):
        """
        Determine if lying position appears controlled/intentional
        """
        try:
            # Check head position relative to body
            nose = keypoints[0]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]

            shoulder_center = (left_shoulder + right_shoulder) / 2

            # In controlled lying, head is usually aligned or slightly elevated
            head_relative_y = nose[1] - shoulder_center[1]

            # Check limb positioning
            left_elbow = keypoints[7]
            right_elbow = keypoints[8]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]

            # Controlled lying often has arms in relaxed positions
            arms_extended = (
                    np.linalg.norm(left_wrist - left_elbow) > 30 and
                    np.linalg.norm(right_wrist - right_elbow) > 30
            )

            # Legs positioning
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]

            legs_positioned = (
                    abs(left_knee[1] - right_knee[1]) < 50 and  # Knees at similar height
                    abs(left_ankle[1] - right_ankle[1]) < 50  # Ankles at similar height
            )

            # Controlled lying indicators
            controlled_indicators = 0
            if head_relative_y <= 20:  # Head not lower than shoulders
                controlled_indicators += 1
            if arms_extended:
                controlled_indicators += 1
            if legs_positioned:
                controlled_indicators += 1

            return controlled_indicators >= 2

        except Exception as e:
            logger.debug(f"Controlled lying analysis error: {e}")
            return False

    def analyze_pose_transition(self, person_bbox, current_posture):
        """
        Analyze how the person transitioned to current pose
        """
        bbox_key = f"{person_bbox[0]}_{person_bbox[1]}"  # Simple spatial key

        if bbox_key not in self.transition_history:
            self.transition_history[bbox_key] = deque(maxlen=10)

        history = self.transition_history[bbox_key]
        history.append(current_posture)

        if len(history) < 3:
            return {"transition_type": "unknown", "confidence": 0.0}

        # Analyze transition pattern
        recent_poses = list(history)[-5:]  # Last 5 poses

        # Check for abrupt transitions (potential fall)
        if (PostureType.STANDING in recent_poses[:-2] and
                current_posture == PostureType.FALLEN):
            return {"transition_type": "abrupt_fall", "confidence": 0.9}

        # Check for gradual transitions (intentional)
        if (PostureType.STANDING in recent_poses and
                PostureType.SITTING in recent_poses and
                current_posture == PostureType.LYING_ON_FURNITURE):
            return {"transition_type": "gradual_intentional", "confidence": 0.8}

        return {"transition_type": "stable", "confidence": 0.5}

    def analyze_ground_contact(self, person_region, pose_keypoints):
        """
        Analyze if person is in direct contact with ground
        """
        try:
            # Simple ground detection based on image bottom region
            height, width = person_region.shape[:2]
            bottom_region = person_region[int(height * 0.8):, :]

            # Look for ground/floor patterns (usually uniform colors/textures)
            gray_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
            ground_uniformity = 1.0 - (np.std(gray_bottom) / 255.0)

            # High uniformity suggests ground contact
            return ground_uniformity > 0.6

        except Exception as e:
            logger.debug(f"Ground contact analysis error: {e}")
            return True  # Default assumption