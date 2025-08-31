# src/core/utils/video_processor_fixed.py
"""
Fixed video processor that adds classifier integration to the existing video_processor.py
This is a minimal change version that maintains full compatibility.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
from threading import Event
from pathlib import Path
import joblib

# Import settings configuration
from config.settings import get_settings
from src.core.analysis.danger_calculator import calculate_danger

# Try to import the original update_danger_score
try:
    from src.core.utils.video_processor import update_danger_score
except ImportError:
    def update_danger_score(score, danger_label):
        """Fallback danger score update."""
        if danger_label and hasattr(danger_label, 'winfo_exists') and danger_label.winfo_exists():
            try:
                danger_label.config(text=f"Danger Score: {score:.2f}")
            except:
                pass


def process_video(video_path, danger_label, canvas, root, stop_event: Event, output_video_path=None):
    """
    Enhanced process_video with classifier integration.
    Maintains exact same interface as original video_processor.py

    FIXES: Classifier now receives person bounding box instead of full frame.
    """
    print("Starting enhanced video processing with classifier...")

    # Get configuration settings
    config = get_settings()

    # Load YOLO model
    model_path = config.YOLO_MODEL_PATH
    if not model_path.exists():
        print(f"Error: YOLO model not found at {model_path}")
        return

    local_model = YOLO(str(model_path))
    print(f"✅ YOLO model loaded: {model_path}")

    # Load classifier if available
    classifier = None
    pose_model = None
    classifier_enabled = False

    if config.CLASSIFIER_PATH.exists():
        try:
            classifier = joblib.load(str(config.CLASSIFIER_PATH))
            print(f"✅ Classifier loaded: {config.CLASSIFIER_PATH}")
            classifier_enabled = True

            # Load pose model for feature extraction
            if config.POSE_MODEL_PATH.exists():
                pose_model = YOLO(str(config.POSE_MODEL_PATH))
                print(f"✅ Pose model loaded: {config.POSE_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Classifier loading failed: {e}")
            classifier_enabled = False

    # Open video capture
    if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
        cap = cv2.VideoCapture(int(video_path))
        is_camera = True
        print(f"📹 Opening camera index: {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        is_camera = False
        print(f"📼 Opening video file: {video_path}")

    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        return []

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if not is_camera else 30

    # Calculate display size
    max_display_width = 800
    max_display_height = 600
    aspect_ratio = original_width / original_height

    if aspect_ratio > max_display_width / max_display_height:
        display_width = max_display_width
        display_height = int(max_display_width / aspect_ratio)
    else:
        display_height = max_display_height
        display_width = int(max_display_height * aspect_ratio)

    # Update canvas
    canvas.config(width=display_width, height=display_height)

    # Setup video writer if needed
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (display_width, display_height))

    # Processing variables
    results_list = []
    after_id = None
    falling_duration = 0
    frame_count = 0
    detection_history = []
    classifier_calls = 0

    # Classes from config
    classes = config.CLASSES
    class_colors = {
        "person": (0, 255, 0),
        "falling_person": (0, 0, 255),
        "sitting_person": (255, 255, 0),
        "lying_person": (255, 0, 255)
    }

    def classify_person(person_crop):
        """
        Classify if person needs help using the cropped region.
        """
        if not classifier_enabled or classifier is None or pose_model is None:
            return None

        try:
            # Extract pose features
            results = pose_model(person_crop, verbose=False, imgsz=224)

            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy().flatten()

                # Ensure correct dimension
                if len(keypoints) >= 34:
                    features = keypoints[:34]
                else:
                    features = np.pad(keypoints, (0, 34 - len(keypoints)), 'constant')

                # Classify
                prediction = classifier.predict([features])[0]
                confidence = 1.0

                if hasattr(classifier, 'predict_proba'):
                    probs = classifier.predict_proba([features])[0]
                    confidence = float(max(probs))

                # 0 = need help, 1 = fine
                return {
                    'needs_help': (prediction == 0),
                    'confidence': confidence
                }
        except Exception as e:
            print(f"Classification error: {e}")

        return None

    def read_frame():
        """Process single frame with classifier integration."""
        nonlocal results_list, after_id, falling_duration, frame_count, detection_history, classifier_calls

        # Check stop event
        if stop_event.is_set():
            print("Stopping video processing...")
            cleanup_resources()
            return

        # Read frame
        ret, frame = cap.read()
        if not ret:
            cleanup_resources()
            return

        # Resize for display
        frame = cv2.resize(frame, (display_width, display_height))

        # YOLO detection
        confidence_threshold = config.CONFIDENCE_THRESHOLD
        results = local_model.predict(frame, conf=confidence_threshold, stream=True, verbose=False)

        frame_results = []
        has_falling_person = False
        detected_objects = []

        # Process detections
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    # Get class name
                    try:
                        class_name = classes[class_id] if class_id < len(classes) else "Unknown"
                    except IndexError:
                        class_name = "Unknown"

                    # CRITICAL FIX: Extract person bounding box for classifier
                    needs_help = False
                    help_confidence = 0.0

                    if classifier_enabled and class_name in ["falling_person", "lying_person"]:
                        # Extract person region
                        person_crop = frame[y1:y2, x1:x2]

                        if person_crop.size > 0:
                            # Resize to standard size
                            person_crop = cv2.resize(person_crop, (224, 224))

                            # Classify the person region
                            classification = classify_person(person_crop)

                            if classification:
                                needs_help = classification['needs_help']
                                help_confidence = classification['confidence']
                                classifier_calls += 1

                                if frame_count % 30 == 0:  # Log every 30 frames
                                    print(f"Classifier: {'HELP' if needs_help else 'OK'} ({help_confidence:.2f})")

                    # Get color
                    color = class_colors.get(class_name, (255, 255, 255))
                    if needs_help:
                        color = (0, 0, 255)  # Red for confirmed need help

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Label
                    label = f"{class_name}: {confidence:.2f}"
                    if classifier_enabled and help_confidence > 0:
                        label += f" [{'HELP' if needs_help else 'OK'}]"

                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Check for fall
                    if class_name == "falling_person" or needs_help:
                        has_falling_person = True
                        fall_detected = 1
                    else:
                        fall_detected = 0

                    frame_results.append(fall_detected)
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'needs_help': needs_help
                    })

        # Update fall duration
        if has_falling_person:
            falling_duration += 1
        else:
            falling_duration = max(0, falling_duration - 1)

        # Store results
        detection_history.append(has_falling_person)
        if len(detection_history) > 30:
            detection_history.pop(0)

        results_list.append(frame_results or [0])

        # Draw info overlay
        info_text = [
            f"Frame: {frame_count}",
            f"Detections: {len(detected_objects)}",
            f"Fall Duration: {falling_duration}",
        ]

        if classifier_enabled:
            info_text.append(f"Classifier Calls: {classifier_calls}")

        y_pos = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20

        # Calculate danger score
        danger_score = 0.0
        if has_falling_person:
            danger_score = min(1.0, 0.5 + (falling_duration / 60))  # Max out at 2 seconds

        # Update danger display
        try:
            update_danger_score(danger_score, danger_label)
        except:
            pass  # Ignore display errors

        # Save frame if recording
        if out:
            out.write(frame)

        # Display frame
        show_frame_on_canvas(frame, canvas, root)

        frame_count += 1

        # Schedule next frame
        delay = 33 if is_camera else 10
        after_id = root.after(delay, read_frame)

    def cleanup_resources():
        """Clean up resources."""
        print(f"Cleanup: Processed {frame_count} frames, {classifier_calls} classifications")
        cap.release()
        if out:
            out.release()
        if after_id:
            try:
                root.after_cancel(after_id)
            except:
                pass
        cv2.destroyAllWindows()

    def show_frame_on_canvas(frame, canvas, root):
        """Display frame on canvas."""
        if not canvas.winfo_exists():
            return

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # Keep reference
            canvas.imgtk = imgtk
            canvas.current_image = img

            # Display
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            root.update_idletasks()

        except Exception as e:
            # Don't crash on display errors
            pass

    # Start processing
    print(f"Starting {'camera' if is_camera else 'video'} processing...")
    print(f"Classifier: {'ENABLED' if classifier_enabled else 'DISABLED'}")
    after_id = root.after(10, read_frame)


# For testing
if __name__ == "__main__":
    import sys

    # Simple test without GUI
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = 0

    print(f"Testing with: {video_path}")

    # Create dummy GUI elements
    root = tk.Tk()
    canvas = tk.Canvas(root, width=800, height=600)
    canvas.pack()
    danger_label = tk.Label(root, text="Danger Score: 0.00")
    danger_label.pack()

    stop_event = Event()

    # Process video
    process_video(video_path, danger_label, canvas, root, stop_event)

    root.mainloop()