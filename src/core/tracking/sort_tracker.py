# src/core/tracking/sort_tracker.py

import numpy as np
class SORT:
    def __init__(self, max_miss=5, min_hits=3, iou_threshold=0.3):
        self.trackers = []
        self.track_id_count = 0
        self.max_miss = max_miss
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        updated_tracks = []

        for detection in detections:
            x1, y1, x2, y2 = detection[:4]
            cls = detection[4] if len(detection) > 4 else 0
            matched = False

            for tracker in self.trackers:
                if self.iou(tracker['bbox'], detection[:4]) > self.iou_threshold:
                    tracker['bbox'] = detection[:4]
                    tracker['hits'] += 1
                    tracker['misses'] = 0
                    tracker['cls'] = cls
                    tracker['age'] = self.frame_count
                    updated_tracks.append(tracker)
                    matched = True
                    break

            if not matched:
                new_tracker = {
                    'id': self.track_id_count,
                    'bbox': detection[:4],
                    'hits': 1,
                    'misses': 0,
                    'cls': cls,
                    'age': self.frame_count
                }
                self.track_id_count += 1
                updated_tracks.append(new_tracker)

        for tracker in self.trackers:
            if tracker not in updated_tracks:
                tracker['misses'] += 1
                if tracker['misses'] < self.max_miss:
                    updated_tracks.append(tracker)

        self.trackers = [t for t in updated_tracks if t['misses'] < self.max_miss]

        return [[*tracker['bbox'], tracker['id'], tracker['cls']] for tracker in self.trackers
                if tracker['hits'] >= self.min_hits]

    @staticmethod
    def iou(bbox1, bbox2):

        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x4 - x3) * (y4 - y3)
        union_area = bbox1_area + bbox2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0