# src/core/utils/video_processor.py

import os
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk
from threading import Event

# Import settings configuration
from config.settings import get_settings
from src.core.analysis.danger_calculator import calculate_danger


def process_video(video_path, danger_label, canvas, root, stop_event: Event, output_video_path=None):
    """
    Process video for fall detection using YOLO model with configuration from settings.
    Enhanced for separate window display with proper scaling and scrolling.

    Args:
        video_path: Path to input video file or camera index
        danger_label: Tkinter label widget to display danger score
        canvas: Tkinter canvas widget to display video frames
        root: Tkinter root window
        stop_event: Threading event for stopping video processing
        output_video_path: Optional path to save processed video
    """
    print("Starting video processing...")

    # Get configuration settings
    config = get_settings()

    # Load YOLO model from configuration
    model_path = config.YOLO_MODEL_PATH
    if not model_path.exists():
        print(f"Error: YOLO model not found at {model_path}")
        return

    local_model = YOLO(str(model_path))
    print(f"Video thread started with model: {model_path}")

    # Open video capture (handle both file paths and camera indices)
    if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
        cap = cv2.VideoCapture(int(video_path))
        is_camera = True
        print(f"Opening camera index: {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        is_camera = False
        print(f"Opening video file: {video_path}")

    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_path}")
        update_canvas_with_error(canvas, f"Cannot open video source: {video_path}")
        return []

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if not is_camera else 30

    print(f"Video properties: {original_width}x{original_height} @ {fps} FPS")

    # Calculate display size (maintain aspect ratio)
    max_display_width = 800
    max_display_height = 600

    aspect_ratio = original_width / original_height
    if aspect_ratio > max_display_width / max_display_height:
        display_width = max_display_width
        display_height = int(max_display_width / aspect_ratio)
    else:
        display_height = max_display_height
        display_width = int(max_display_height * aspect_ratio)

    # Update canvas size
    canvas.config(width=display_width, height=display_height)
    canvas.configure(scrollregion=(0, 0, display_width, display_height))

    # Setup video writer if output path is specified
    out = None
    if output_video_path and isinstance(output_video_path, str) and not is_camera:
        output_dir = config.PROCESSED_VIDEOS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        if not os.path.dirname(output_video_path):
            output_video_path = output_dir / os.path.basename(output_video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        try:
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (display_width, display_height))
            print(f"Video will be saved to: {output_video_path}")
        except cv2.error as e:
            print(f"Error initializing VideoWriter: {e}")

    # Initialize processing variables
    results_list = []
    after_id = None
    falling_duration = 0
    frame_count = 0
    detection_history = []

    # Get detection classes from configuration
    classes = config.CLASSES
    class_colors = {
        "person": (0, 255, 0),
        "falling_person": (0, 0, 255),
        "sitting_person": (255, 255, 0),
        "lying_person": (255, 0, 255)
    }

    def read_frame():
        """Process single frame for fall detection with enhanced display."""
        nonlocal results_list, after_id, falling_duration, frame_count, detection_history

        # Check stop event
        if stop_event.is_set():
            print("Stopping video processing...")
            cleanup_resources()
            return

        # Read frame from video
        ret, frame = cap.read()
        if not ret:
            if not is_camera:
                print("Video ended or no more frames.")
            else:
                print("Camera disconnected or error reading frame.")
            cleanup_resources()
            return

        # Resize frame for processing and display
        frame = cv2.resize(frame, (display_width, display_height))

        # Perform YOLO detection with confidence threshold from config
        confidence_threshold = config.CONFIDENCE_THRESHOLD
        results = local_model.predict(frame, conf=confidence_threshold, stream=True, verbose=False)

        frame_results = []
        has_falling_person = False
        detected_objects = []

        # Process detection results
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    # Map class ID to class name
                    try:
                        class_name = classes[class_id] if class_id < len(classes) else "Unknown"
                    except IndexError:
                        class_name = "Unknown"
                        print(f"Warning: class_id {class_id} is out of range.")

                    # Get color for class
                    color = class_colors.get(class_name, (255, 255, 255))

                    # Draw enhanced bounding box
                    draw_enhanced_bbox(frame, x1, y1, x2, y2, class_name, confidence, color)

                    # Check for falling person detection
                    if class_name == "falling_person":
                        fall_detected = 1
                        has_falling_person = True
                    else:
                        fall_detected = 0

                    frame_results.append(fall_detected)
                    detected_objects.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2)
                    })

                    # Debug output
                    if config.DEBUG_MODE:
                        print(f"Frame {frame_count}: Detected {class_name} with confidence {confidence:.2f}")

        # Update falling duration and detection history
        if has_falling_person:
            falling_duration += 1
        else:
            falling_duration = max(0, falling_duration - 1)

        # Keep detection history for trend analysis
        detection_history.append(has_falling_person)
        if len(detection_history) > 30:  # Keep last 30 frames
            detection_history.pop(0)

        # Store frame results
        results_list.append(frame_results or [0])

        # Draw information overlay
        draw_information_overlay(frame, frame_count, len(detected_objects),
                                 falling_duration, detection_history)

        # Create analysis result for danger calculation
        analysis_result = create_analysis_result(frame_results, has_falling_person, detected_objects)

        # Calculate danger score
        try:
            danger_score = calculate_danger(analysis_result, falling_duration)
            update_danger_score(danger_score, danger_label)
        except Exception as e:
            print(f"Error calculating danger score: {e}")
            danger_score = 0
            update_danger_score(danger_score, danger_label)

        # Save frame to output video if enabled
        if out:
            out.write(frame)

        # Display frame on canvas with enhanced features
        show_frame_on_canvas(frame, canvas, root)

        frame_count += 1

        # Schedule next frame processing (adjust delay for camera vs video)
        delay = 33 if is_camera else 10  # ~30 FPS for camera, faster for video
        after_id = root.after(delay, read_frame)

    def draw_enhanced_bbox(frame, x1, y1, x2, y2, class_name, confidence, color):
        """Draw enhanced bounding box with better visibility."""
        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f'{class_name}: {confidence:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        # Background rectangle for text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0] + 10, y1), color, -1)

        # White text on colored background
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_information_overlay(frame, frame_num, num_detections, fall_duration, history):
        """Draw information overlay on frame."""
        height, width = frame.shape[:2]

        # Semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Frame information
        info_texts = [
            f"Frame: {frame_num}",
            f"Detections: {num_detections}",
            f"Fall Duration: {fall_duration}",
            f"Detection Rate: {sum(history)}/{len(history)}" if history else "Detection Rate: 0/0"
        ]

        for i, text in enumerate(info_texts):
            y_pos = 30 + i * 20
            cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

    def cleanup_resources():
        """Clean up video capture and writer resources."""
        cap.release()
        if out:
            out.release()
            print(f"Video saved to: {output_video_path}")
        if after_id:
            try:
                root.after_cancel(after_id)
            except:
                pass  # Window might be destroyed already
        cv2.destroyAllWindows()

    def create_analysis_result(frame_results, has_falling_person, detected_objects):
        """Create enhanced analysis result for danger calculation."""
        return {
            'gpt_analysis': {
                'onePerson': 'true' if any(
                    obj['class'] in ['person', 'falling_person', 'sitting_person', 'lying_person']
                    for obj in detected_objects) else 'false',
                'faceToTheGround': 'true' if has_falling_person else 'false',
                'possible_age': 'adults',
                'gender': 'unknown',
                'status': ['fall'] if has_falling_person else ['walk'],
                'environment': 'indoor',
                'lighting': 'bright',
                'time_of_day': 'day'
            },
            'detection_count': len(detected_objects),
            'frame_number': frame_count
        }

    # Start frame processing
    print(f"Starting {'camera' if is_camera else 'video'} processing:")
    print(f"- Original size: {original_width}x{original_height}")
    print(f"- Display size: {display_width}x{display_height}")
    print(f"- Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"- Detection classes: {', '.join(classes)}")

    after_id = root.after(10, read_frame)


def show_frame_on_canvas(frame, canvas, root):
    """
    Enhanced frame display with proper scaling and error handling.

    Args:
        frame: OpenCV frame (BGR format)
        canvas: Tkinter canvas widget
        root: Tkinter root window
    """
    if not canvas.winfo_exists():
        return

    try:
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)

        # Store original image for screenshot functionality
        canvas.current_image = img

        # Convert to PhotoImage for Tkinter
        imgtk = ImageTk.PhotoImage(image=img)

        # Keep reference to avoid garbage collection
        canvas.imgtk = imgtk

        # Clear canvas and display new image
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        # Update canvas scroll region
        canvas.configure(scrollregion=canvas.bbox("all"))

        root.update_idletasks()

    except Exception as e:
        print(f"Error displaying frame on canvas: {e}")
        update_canvas_with_error(canvas, f"Display error: {str(e)}")


def update_canvas_with_error(canvas, error_message):
    """Display error message on canvas."""
    try:
        if canvas.winfo_exists():
            canvas.delete("all")
            canvas.create_text(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                               text=error_message, fill="red", font=("Arial", 12))
    except:
        pass


def update_danger_score(score, danger_label):
    """
    Enhanced danger score update with better visual feedback.

    Args:
        score: Calculated danger score (float)
        danger_label: Tkinter label widget to update
    """
    if not danger_label.winfo_exists():
        return

    try:
        # Get danger thresholds from configuration
        config = get_settings()
        danger_threshold = config.DANGER_SETTINGS.get('threshold', 5)

        # Update label text with better formatting
        danger_label.config(text=f"Danger Score: {score:.2f}")

        # Enhanced color coding with more granular levels
        if score < danger_threshold * 0.3:  # Very low danger
            danger_label.config(fg="#27ae60", bg=danger_label.master['bg'])  # Green
        elif score < danger_threshold * 0.6:  # Low danger
            danger_label.config(fg="#f39c12", bg=danger_label.master['bg'])  # Orange
        elif score < danger_threshold:  # Medium danger
            danger_label.config(fg="#e67e22", bg=danger_label.master['bg'])  # Dark orange
        else:  # High danger
            danger_label.config(fg="#e74c3c", bg=danger_label.master['bg'])  # Red
            # Add blinking effect for high danger
            if score > danger_threshold * 1.5:
                danger_label.config(bg="#ffebee" if int(score * 10) % 2 else danger_label.master['bg'])

    except Exception as e:
        print(f"Error updating danger score label: {e}")


def get_video_info(video_path):
    """
    Enhanced video information retrieval.

    Args:
        video_path: Path to video file or camera index

    Returns:
        Dictionary with video information
    """
    try:
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
            cap = cv2.VideoCapture(int(video_path))
            is_camera = True
        else:
            cap = cv2.VideoCapture(video_path)
            is_camera = False

        if not cap.isOpened():
            return None

        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)) if not is_camera else 30,
            'is_camera': is_camera
        }

        if not is_camera:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info.update({
                'frame_count': frame_count,
                'duration': frame_count / info['fps'] if info['fps'] > 0 else 0
            })

        cap.release()
        return info

    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def analyze_person_state(class_name, bbox, frame, duration):
    """
    Enhanced person state analysis with context awareness.
    """
    # Basic classification from YOLO
    is_non_standing = class_name in ["falling_person", "sitting_person", "lying_person"]

    if not is_non_standing:
        return "normal", 0.0

    # Context-aware analysis
    danger_score = 0.0

    # 1. Duration analysis
    if class_name == "falling_person":
        if duration > 50:  # Sustained falling state
            danger_score += 0.6
        else:
            danger_score += 0.3

    # 2. Environment context
    if class_name == "lying_person":
        # Check if on furniture (low danger) vs ground (higher danger)
        if is_on_furniture(bbox, frame):
            danger_score += 0.1  # Lying on chair/bed
        else:
            danger_score += 0.4  # Lying on ground

    # 3. Movement analysis
    if duration > 100 and class_name != "person":  # Long time in non-standing position
        danger_score += 0.3

    return class_name, min(danger_score, 1.0)


def is_on_furniture(bbox, frame):
    """
    Simple heuristic to detect if person is on furniture.
    """
    x1, y1, x2, y2 = bbox
    person_bottom = y2
    frame_height = frame.shape[0]

    # If person's bottom is not near ground level, likely on furniture
    ground_ratio = person_bottom / frame_height
    return ground_ratio < 0.8  # Person not near bottom of frame


def draw_enhanced_information_overlay(frame, frame_num, detections, fall_duration, history):
    """
    Enhanced information display with better classification breakdown.
    """
    # Count different types of detections
    detection_counts = {
        'person': 0,
        'sitting_person': 0,
        'lying_person': 0,
        'falling_person': 0
    }

    for detection in detections:
        class_name = detection.get('class_name', 'unknown')
        if class_name in detection_counts:
            detection_counts[class_name] += 1

    # Calculate danger-specific detection rate
    danger_detections = sum(history[-30:]) if len(history) >= 30 else sum(history)
    total_recent_frames = min(30, len(history))

    info_texts = [
        f"Frame: {frame_num}",
        f"People: {detection_counts['person']}",
        f"Sitting: {detection_counts['sitting_person']}",
        f"Lying: {detection_counts['lying_person']}",
        f"Falling: {detection_counts['falling_person']}",
        f"Danger Rate: {danger_detections}/{total_recent_frames}",
        f"Fall Duration: {fall_duration}"
    ]

    # Draw with better formatting
    y_start = 30
    for i, text in enumerate(info_texts):
        color = (0, 0, 255) if 'Falling' in text or 'Danger' in text else (255, 255, 255)
        cv2.putText(frame, text, (10, y_start + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)