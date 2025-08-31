#!/usr/bin/env python
"""Quick performance test to verify improvements."""

import time
import cv2
from pathlib import Path
from config.settings import get_settings
from src.core.detection.optimized_fall_detector import OptimizedFallDetector


def test_performance():
    """Test detection performance with fixed bbox extraction."""

    settings = get_settings()

    # Initialize detector
    detector = OptimizedFallDetector(settings)

    # Load test video
    test_video = settings.TEST_VIDEO_PATH
    cap = cv2.VideoCapture(str(test_video))

    frame_count = 0
    correct_detections = 0
    start_time = time.time()

    while frame_count < 100:  # Test first 100 frames
        ret, frame = cap.read()
        if not ret:
            break

        # Create dummy detections for testing
        h, w = frame.shape[:2]
        detections = [
            {'bbox': (100, 100, 200, 300), 'track_id': 1},  # Standing person
            {'bbox': (300, 200, 450, 250), 'track_id': 2},  # Fallen person
        ]

        # Process with fixed detector
        results = detector.smart_fall_detection(frame, detections, frame_count)

        # Count detections
        for result in results:
            if result.get('needs_help'):
                correct_detections += 1

        frame_count += 1

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    print(f"\n📊 Performance Test Results:")
    print(f"   Frames processed: {frame_count}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Detections: {correct_detections}")

    cap.release()
    detector.cleanup()

    return fps > 10  # Should achieve at least 10 FPS


if __name__ == "__main__":
    success = test_performance()
    if success:
        print("\n✅ Performance test PASSED!")
    else:
        print("\n❌ Performance test FAILED - FPS too low")