# src/core/detection/improved_yolo_detector.py

import cv2
import numpy as np
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class ImprovedYOLODetector:
    """
    Enhanced YOLO detector with multi-scale detection and confidence adjustment
    """

    def __init__(self, model_path, settings):
        self.model = YOLO(str(model_path))
        self.settings = settings

        # IMPROVED: Multi-scale detection parameters
        self.detection_scales = [320, 416, 640]  # Multiple input sizes
        self.min_confidence = 0.25  # Lower for initial detection
        self.nms_threshold = 0.45  # Non-maximum suppression

        # Context-aware confidence adjustment
        self.base_confidence = settings.CONFIDENCE_THRESHOLD
        self.scene_adaptation = True

    def detect_with_multi_scale(self, frame):
        """
        Perform multi-scale detection to improve person detection reliability
        """
        all_detections = []

        for scale in self.detection_scales:
            try:
                # Run detection at current scale
                results = self.model(frame,
                                     imgsz=scale,
                                     conf=self.min_confidence,
                                     iou=self.nms_threshold,
                                     verbose=False)

                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()

                    # Filter for person class (class 0 in COCO)
                    person_mask = classes == 0

                    if np.any(person_mask):
                        person_boxes = boxes[person_mask]
                        person_confs = confidences[person_mask]

                        # Add scale information
                        for box, conf in zip(person_boxes, person_confs):
                            all_detections.append({
                                'bbox': box,
                                'confidence': conf,
                                'scale': scale,
                                'class': 'person'
                            })

            except Exception as e:
                logger.warning(f"Detection failed at scale {scale}: {e}")
                continue

        # Merge and filter detections
        merged_detections = self.merge_multi_scale_detections(all_detections)

        return merged_detections

    def merge_multi_scale_detections(self, detections):
        """
        Merge detections from multiple scales using weighted NMS
        """
        if not detections:
            return []

        # Convert to numpy arrays
        boxes = np.array([det['bbox'] for det in detections])
        scores = np.array([det['confidence'] for det in detections])

        # Weighted NMS - give higher weight to detections from optimal scales
        scale_weights = {320: 1.0, 416: 1.2, 640: 1.1}

        weighted_scores = []
        for det in detections:
            weight = scale_weights.get(det['scale'], 1.0)
            weighted_scores.append(det['confidence'] * weight)

        weighted_scores = np.array(weighted_scores)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            weighted_scores.tolist(),
            self.min_confidence,
            self.nms_threshold
        )

        final_detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                det = detections[i]
                # Apply final confidence threshold
                if det['confidence'] >= self.base_confidence:
                    final_detections.append(det)

        return final_detections

    def adaptive_confidence_adjustment(self, frame, detections):
        """
        Adjust confidence thresholds based on scene characteristics
        """
        if not self.scene_adaptation or not detections:
            return detections

        # Analyze scene characteristics
        scene_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        scene_contrast = np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Adjust confidence based on scene quality
        confidence_adjustment = 1.0

        # Low light adjustment
        if scene_brightness < 80:
            confidence_adjustment *= 0.85
            logger.debug("Low light detected, reducing confidence threshold")

        # Low contrast adjustment
        if scene_contrast < 30:
            confidence_adjustment *= 0.9
            logger.debug("Low contrast detected, reducing confidence threshold")

        # Apply adjustment
        adjusted_detections = []
        adjusted_threshold = self.base_confidence * confidence_adjustment

        for det in detections:
            if det['confidence'] >= adjusted_threshold:
                adjusted_detections.append(det)

        return adjusted_detections