#!/usr/bin/env python
"""Quick performance test to verify improvements."""

import time
import cv2
from pathlib import Path
from config.settings import get_settings
# Changed: Import from fall_detector instead of optimized_fall_detector
from src.core.detection.fall_detector import UnifiedFallDetector


def test_performance():
    """Test detection performance with fixed bbox extraction."""

    settings = get_settings()

    # Initialize detector
    detector = UnifiedFallDetector(settings)

    # Load test video
    test_video = settings.TEST_VIDEO_PATH

    if not Path(test_video).exists():
        print(f"⚠️  Test video not found at: {test_video}")
        print("   Using first available video in benchmark folder...")
        benchmark_dir = Path(r"C:\Users\Bingshen\Videos\AI Train\movies6\benchmark")
        videos = list(benchmark_dir.glob("*.mp4"))
        if videos:
            test_video = str(videos[0])
            print(f"   Found: {test_video}")
        else:
            print("❌ No test videos found!")
            return False

    cap = cv2.VideoCapture(str(test_video))

    if not cap.isOpened():
        print(f"❌ Cannot open video: {test_video}")
        return False

    frame_count = 0
    correct_detections = 0
    start_time = time.time()

    while frame_count < 100:  # Test first 100 frames
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with UnifiedFallDetector
        results = detector.process_frame(frame)

        # Count fall detections from results
        if 'alerts' in results:
            for alert in results['alerts']:
                correct_detections += 1

        frame_count += 1

        # Show progress
        if frame_count % 20 == 0:
            print(f"   Processed {frame_count} frames...")

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"\n📊 Performance Test Results:")
    print(f"   Frames processed: {frame_count}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Fall alerts: {correct_detections}")
    print(f"   Processing time: {elapsed_time:.2f}s")

    cap.release()
    detector.cleanup()

    return fps > 10  # Should achieve at least 10 FPS


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QUICK PERFORMANCE TEST")
    print("=" * 60)

    try:
        success = test_performance()
        if success:
            print("\n✅ Performance test PASSED!")
        else:
            print("\n❌ Performance test FAILED - FPS too low")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()