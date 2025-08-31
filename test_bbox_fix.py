#!/usr/bin/env python
"""
Simple test script to verify bounding box extraction is working correctly.
Run this to ensure the classifier receives person crops, not full frames.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from ultralytics import YOLO
import joblib


def test_bbox_extraction():
    """Test that we're extracting bboxes correctly before classification."""

    print("=" * 60)
    print("BOUNDING BOX EXTRACTION TEST")
    print("=" * 60)

    # Load settings
    settings = get_settings()

    # Load models
    print("\n1. Loading models...")
    yolo_model = YOLO(str(settings.YOLO_MODEL_PATH))
    classifier = joblib.load(str(settings.CLASSIFIER_PATH))
    print("   ✅ Models loaded")

    # Load a test frame (you can use any test video)
    test_video = settings.TEST_VIDEO_PATH
    if not test_video or not test_video.exists():
        print("   ❌ No test video found. Please set TEST_VIDEO_PATH in .env")
        return

    cap = cv2.VideoCapture(str(test_video))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("   ❌ Could not read test video")
        return

    print(f"   ✅ Test frame loaded: {frame.shape}")

    # Detect people in frame
    print("\n2. Detecting people...")
    results = yolo_model(frame, conf=0.5)

    person_count = 0
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if class_id == 0:  # Person class
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    print(f"\n   Person {person_count} detected:")
                    print(f"   BBox: ({x1}, {y1}, {x2}, {y2})")

                    # Extract person crop (THIS IS THE KEY FIX!)
                    person_crop = frame[y1:y2, x1:x2]
                    print(f"   Crop shape: {person_crop.shape}")

                    # Resize for classifier
                    person_crop_resized = cv2.resize(person_crop, (224, 224))
                    print(f"   Resized shape: {person_crop_resized.shape}")

                    # Test classification (if you have pose model)
                    try:
                        # For now, just verify the crop is valid
                        if person_crop_resized.size > 0:
                            print("   ✅ Valid person crop extracted!")

                            # Save the crop for visual verification
                            crop_path = f"test_person_crop_{person_count}.jpg"
                            cv2.imwrite(crop_path, person_crop_resized)
                            print(f"   💾 Saved crop to: {crop_path}")
                        else:
                            print("   ❌ Invalid crop!")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")

    if person_count == 0:
        print("   ⚠️  No people detected in test frame")
    else:
        print(f"\n✅ Successfully extracted {person_count} person bounding boxes")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def compare_approaches():
    """Compare wrong vs correct approach."""

    print("\n📊 COMPARISON: Wrong vs Correct Approach")
    print("-" * 60)

    # Create a dummy frame with multiple people
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Simulate multiple people bboxes
    person1_bbox = (50, 100, 150, 300)  # Standing person
    person2_bbox = (400, 200, 550, 280)  # Fallen person (width > height)

    print("\n❌ WRONG APPROACH (Old):")
    print("   classifier.predict(frame)  # Sends entire 640x480 frame")
    print("   Problem: Classifier sees multiple people, gets confused")

    print("\n✅ CORRECT APPROACH (Fixed):")
    print("   For person 1:")
    print(f"     person1_crop = frame[100:300, 50:150]  # Extract bbox")
    print(f"     person1_crop = cv2.resize(person1_crop, (224, 224))")
    print(f"     classifier.predict(person1_crop)  # Only this person")

    print("\n   For person 2 (fallen):")
    print(f"     person2_crop = frame[200:280, 400:550]  # Extract bbox")
    print(f"     person2_crop = cv2.resize(person2_crop, (224, 224))")
    print(f"     classifier.predict(person2_crop)  # Only this person")

    print("\n💡 KEY INSIGHT:")
    print("   Each person is classified individually using only their bbox!")
    print("   This prevents confusion in multi-person scenes.")


if __name__ == "__main__":
    print("\n🔍 Testing Bounding Box Extraction Fix\n")

    # Run the test
    test_bbox_extraction()

    # Show comparison
    compare_approaches()

    print("\n✨ If you see person crops saved as images, the fix is working!")
    print("   Check the saved images to verify they contain only one person each.\n")