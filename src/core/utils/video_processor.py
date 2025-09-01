# src/core/utils/video_processor.py
"""
Modified video processor with:
1. Fixed bounding box extraction for classifier
2. Person tracking with consistent IDs
3. Danger value calculation
4. GPT analyzer integration when danger > 1.0
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
import time
from collections import defaultdict
from datetime import datetime

# Import settings configuration
from config.settings import get_settings

# Import tracking and analysis modules
from src.core.tracking.sort_tracker import SORT
from src.core.analysis.gpt_analyzer import analyze_image
from src.core.analysis.location_service import getLoc, get_address
from src.core.analysis.weather_service import get_weather
from src.core.utils.cache_manager import DetectionCache


def update_danger_score(score, danger_label):
    """Update danger score display."""
    if danger_label and hasattr(danger_label, 'winfo_exists') and danger_label.winfo_exists():
        try:
            color = "red" if score > 0.7 else "orange" if score > 0.3 else "green"
            danger_label.config(text=f"Danger Score: {score:.2f}", fg=color)
        except:
            pass


def process_video(video_path, danger_label, canvas, root, stop_event: Event, output_video_path=None):
    """
    Enhanced video processor with all requested features.
    """
    print(f"Starting enhanced video processing...")
    print(f"Video path: {video_path}")
    print(f"Video exists: {Path(video_path).exists() if video_path != 0 else 'Camera'}")

    # Get configuration settings
    config = get_settings()

    # Load YOLO model
    model_path = config.YOLO_MODEL_PATH
    if not model_path.exists():
        print(f"Error: YOLO model not found at {model_path}")
        return

    local_model = YOLO(str(model_path))
    print(f"✅ YOLO model loaded: {model_path}")

    # Initialize tracker for consistent person IDs
    tracker = SORT(max_miss=5, min_hits=3, iou_threshold=0.3)

    # Initialize cache for faster repeated processing
    cache_manager = DetectionCache(cache_dir=config.CACHE_DIR)
    cache_key = None
    cached_detections = {}
    use_cache = False

    # Check if we can use cache for this video
    if isinstance(video_path, str) and video_path != "0":
        cache_key = f"{Path(video_path).stem}_{model_path.stem}"
        if cache_manager.cache_exists(video_path, cache_key):
            print("📦 Loading cached detections for faster processing...")
            cached_detections = cache_manager.load_detections(video_path, cache_key)
            if cached_detections:
                use_cache = True
                print(f"✅ Cache loaded: {len(cached_detections)} frames")
            else:
                print("⚠️ Cache empty or corrupted, will rebuild")
                cached_detections = {}
        else:
            print("📝 First run - will build cache for next time")
            cached_detections = {}

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

    # Person tracking data structure
    person_tracking = defaultdict(lambda: {
        'id': None,
        'static_frames': 0,
        'falling_frames': 0,
        'needs_help': False,
        'danger_value': 0.0,
        'last_bbox': None,
        'last_position': None,
        'last_seen': 0,
        'alert_triggered': False,
        'analysis_data': None
    })

    # Alert overlay data (persistent display in top-right)
    alert_overlay = {
        'active': False,
        'person_id': None,
        'danger_value': 0.0,
        'location': None,
        'weather': None,
        'gpt_analysis': None,
        'timestamp': None
    }

    # Helper function for showing frames on canvas
    def show_frame_on_canvas(frame):
        """Display frame on canvas."""
        if not canvas or not hasattr(canvas, 'winfo_exists'):
            return

        try:
            if not canvas.winfo_exists():
                return

            # Check if frame is valid
            if frame is None or frame.size == 0:
                return

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
            print(f"Display error: {e}")

    # Open video with better error handling
    cap = None
    try:
        if isinstance(video_path, str):
            # Convert to Path and check if exists
            video_file = Path(video_path)
            if video_file.exists():
                video_path = str(video_file)
            elif video_path != "0":  # Not camera
                print(f"ERROR: Video file not found: {video_path}")
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"File not found", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                show_frame_on_canvas(error_frame)
                return

        # Handle camera input
        if video_path == "0" or video_path == 0:
            video_path = 0

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"ERROR: Cannot open video: {video_path}")
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Cannot open video", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            show_frame_on_canvas(error_frame)
            return

        # Test read one frame
        ret, test_frame = cap.read()
        if not ret:
            print(f"ERROR: Cannot read from video: {video_path}")
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Cannot read video", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            show_frame_on_canvas(error_frame)
            cap.release()
            return

        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    except Exception as e:
        print(f"ERROR opening video: {e}")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Error: {str(e)[:50]}", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        show_frame_on_canvas(error_frame)
        if cap:
            cap.release()
        return

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Display dimensions
    display_width = min(width, 800)
    display_height = int(height * display_width / width) if width > 0 else 480

    print(f"Video opened: {width}x{height} @ {fps}fps")
    print(f"Display size: {display_width}x{display_height}")

    # Output video writer
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (display_width, display_height))

    # Get class names
    classes = local_model.names if hasattr(local_model, 'names') else {}

    # Frame counter and timing
    frame_count = 0
    classifier_calls = 0
    after_id = None

    def classify_person_bbox(person_crop):
        """Classify extracted person bounding box."""
        if not classifier_enabled or classifier is None or pose_model is None:
            return None

        try:
            # Resize person crop to standard size
            person_crop_resized = cv2.resize(person_crop, (224, 224))

            # Extract pose features
            results = pose_model(person_crop_resized, verbose=False, imgsz=224)

            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy().flatten()

                # Ensure correct dimension (34 features for 17 keypoints x 2 coordinates)
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

    def calculate_danger_value(person_data):
        """
        Calculate danger value for a person.
        Increases when person is static AND needs help.
        """
        danger = 0.0

        # Base danger from falling detection
        if person_data['falling_frames'] > 0:
            danger += 0.3

        # Increase danger if static for long time
        if person_data['static_frames'] > 30:  # 1 second at 30fps
            danger += min(0.4, person_data['static_frames'] / 150)  # Max 0.4 at 5 seconds

        # Major increase if needs help (from classifier)
        if person_data['needs_help']:
            danger += 0.5

            # Additional danger for prolonged need
            if person_data['static_frames'] > 60:  # 2 seconds
                danger += 0.3

        return min(2.0, danger)  # Cap at 2.0

    def trigger_alert_analysis(person_id, person_data, frame):
        """
        Trigger GPT analysis and services when danger > 1.0
        """
        try:
            print(f"\n🚨 ALERT TRIGGERED for Person {person_id}")
            print(f"   Danger Value: {person_data['danger_value']:.2f}")

            # Get location data
            try:
                lat, lon = getLoc()
                address = get_address(lat, lon)
                location_data = {
                    'lat': lat,
                    'lon': lon,
                    'address': address,
                    'city': address.split(',')[1] if ',' in address else 'Unknown',
                    'country': address.split(',')[-1] if ',' in address else 'Unknown'
                }
            except Exception as e:
                print(f"Location service error: {e}")
                location_data = {
                    'lat': 0,
                    'lon': 0,
                    'address': 'Unknown',
                    'city': 'Unknown',
                    'country': 'Unknown'
                }

            # Get weather data
            try:
                temperature, weather_code, time = get_weather(
                    location_data.get('lat', 0),
                    location_data.get('lon', 0)
                )
                weather_data = {
                    'temp': temperature,
                    'code': weather_code,
                    'time': time,
                    'description': 'Clear' if weather_code <= 3 else 'Cloudy' if weather_code <= 50 else 'Bad weather'
                }
            except Exception as e:
                print(f"Weather service error: {e}")
                weather_data = {
                    'temp': 'N/A',
                    'code': 0,
                    'description': 'Unknown'
                }

            # Get GPT analysis - extract person region from frame
            try:
                bbox = person_data.get('last_bbox', None)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    person_crop = frame[y1:y2, x1:x2]
                    gpt_result = analyze_image(person_crop)
                    gpt_response = gpt_result.get('gpt_analysis', {})

                    # Extract detailed information from GPT response
                    status_list = gpt_response.get('status', [])
                    environment = gpt_response.get('environment', 'unknown')
                    age = gpt_response.get('possible_age', 'unknown')
                    face_down = gpt_response.get('faceToTheGround', 'false')

                    # Create detailed summary
                    summary = f"Person {age}, {environment} environment"
                    if status_list:
                        summary += f", {', '.join(status_list)}"
                    if face_down == 'true':
                        summary += ", face down"
                else:
                    gpt_response = {'status': ['fall detected'], 'environment': 'unknown'}
                    summary = "Fall detected, analyzing..."
            except Exception as e:
                print(f"GPT analysis error: {e}")
                gpt_response = {'status': ['analysis failed'], 'environment': 'unknown'}
                summary = "Analysis in progress..."

            # Update alert overlay
            alert_overlay.update({
                'active': True,
                'person_id': person_id,
                'danger_value': person_data['danger_value'],
                'location': f"{location_data.get('city', 'Unknown')}, {location_data.get('country', 'Unknown')}",
                'weather': f"{weather_data.get('description', 'Unknown')}, {weather_data.get('temp', 'N/A')}°C",
                'gpt_analysis': summary,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })

            # Mark alert as triggered
            person_data['alert_triggered'] = True
            person_data['analysis_data'] = {
                'location': location_data,
                'weather': weather_data,
                'gpt': gpt_response
            }

            print(f"   Location: {alert_overlay['location']}")
            print(f"   Weather: {alert_overlay['weather']}")
            print(f"   Analysis: {alert_overlay['gpt_analysis']}")

        except Exception as e:
            print(f"Alert analysis error: {e}")

    def draw_alert_overlay(frame):
        """Draw persistent alert information in top-right corner."""
        if not alert_overlay['active']:
            return frame

        # Create semi-transparent overlay box
        overlay_height = 180
        overlay_width = 350
        overlay = frame.copy()

        # Top-right corner position
        x1 = frame.shape[1] - overlay_width - 10
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = y1 + overlay_height

        # Draw background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Unified font settings for overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.6
        text_scale = 0.4

        # Draw title
        cv2.putText(frame, "ALERT ACTIVE", (x1 + 10, y1 + 20),
                    font, title_scale, (0, 0, 255), 2)

        # Draw alert details
        y_pos = y1 + 45
        line_height = 18

        # Person ID
        cv2.putText(frame, f"Person ID: {alert_overlay['person_id']}",
                    (x1 + 10, y_pos),
                    font, text_scale, (255, 255, 255), 1)
        y_pos += line_height

        # Danger value
        cv2.putText(frame, f"Danger: {alert_overlay['danger_value']:.2f}",
                    (x1 + 10, y_pos),
                    font, text_scale, (0, 255, 255), 1)
        y_pos += line_height

        # Location
        location_text = alert_overlay['location'] if alert_overlay['location'] else 'Unknown'
        # Truncate long location names
        if len(location_text) > 40:
            location_text = location_text[:37] + "..."
        cv2.putText(frame, f"Location: {location_text}",
                    (x1 + 10, y_pos),
                    font, text_scale, (255, 255, 255), 1)
        y_pos += line_height

        # Weather
        weather_text = alert_overlay['weather'] if alert_overlay['weather'] else 'Unknown'
        cv2.putText(frame, f"Weather: {weather_text}",
                    (x1 + 10, y_pos),
                    font, text_scale, (255, 255, 255), 1)
        y_pos += line_height

        # GPT Analysis
        gpt_text = alert_overlay['gpt_analysis'] if alert_overlay['gpt_analysis'] else 'Analyzing...'
        # Split long analysis into two lines if needed
        if len(gpt_text) > 45:
            line1 = gpt_text[:45]
            line2 = gpt_text[45:90] if len(gpt_text) > 45 else ""
            cv2.putText(frame, f"Analysis: {line1}",
                        (x1 + 10, y_pos),
                        font, text_scale, (255, 255, 0), 1)
            y_pos += line_height
            if line2:
                cv2.putText(frame, f"  {line2}",
                            (x1 + 10, y_pos),
                            font, text_scale, (255, 255, 0), 1)
                y_pos += line_height
        else:
            cv2.putText(frame, f"Analysis: {gpt_text}",
                        (x1 + 10, y_pos),
                        font, text_scale, (255, 255, 0), 1)
            y_pos += line_height

        # Timestamp
        cv2.putText(frame, f"Time: {alert_overlay['timestamp']}",
                    (x1 + 10, y_pos),
                    font, text_scale, (200, 200, 200), 1)

        return frame

    def read_frame():
        """Process single frame with all enhancements."""
        nonlocal frame_count, after_id, classifier_calls

        try:
            # Check stop event
            if stop_event.is_set():
                print("Stopping video processing...")
                cleanup_resources()
                return

            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"End of video or read error at frame {frame_count}")
                cleanup_resources()
                return

            # Debug: Print every 30 frames
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...")

            # Resize for display
            frame = cv2.resize(frame, (display_width, display_height))

            # YOLO detection - with caching support
            detections = []
            falling_detections = []

            # Check if we have cached detections for this frame
            if use_cache and frame_count in cached_detections:
                # Use cached detections
                cache_data = cached_detections[frame_count]
                detections = cache_data.get('detections', [])
                falling_detections = cache_data.get('falling_detections', [])

                # Debug every 100 frames
                if frame_count % 100 == 0:
                    print(f"   Using cached frame {frame_count}")
            else:
                # Run YOLO detection
                confidence_threshold = config.CONFIDENCE_THRESHOLD
                results = local_model.predict(frame, conf=confidence_threshold, stream=True, verbose=False)

                # Process YOLO results
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

                            # Check if it's a person-related detection
                            if class_name in ["person", "falling_person", "lying_person"]:
                                detections.append([x1, y1, x2, y2, confidence])

                                if class_name in ["falling_person", "lying_person"]:
                                    falling_detections.append((x1, y1, x2, y2))

                # Save to cache for next time
                if not use_cache and cache_key:
                    cached_detections[frame_count] = {
                        'detections': detections,
                        'falling_detections': falling_detections
                    }

            # Convert to numpy array for tracker
            if detections:
                detections_np = np.array(detections)
            else:
                detections_np = np.empty((0, 5))

            # Update tracker to get consistent IDs
            tracked_objects = tracker.update(detections_np)

            # Process each tracked person
            max_danger = 0.0

            for track in tracked_objects:
                # SORT returns: [x1, y1, x2, y2, track_id, cls] (6 values)
                if len(track) < 5:
                    continue

                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                # cls is at index 5 if present, but we don't need it

                # Update person tracking data
                person = person_tracking[track_id]
                person['id'] = track_id
                person['last_seen'] = frame_count
                person['last_bbox'] = (x1, y1, x2, y2)  # Store bbox for GPT analysis

                # Check if person is falling
                is_falling = any(
                    x1 >= fx1 - 10 and y1 >= fy1 - 10 and
                    x2 <= fx2 + 10 and y2 <= fy2 + 10
                    for fx1, fy1, fx2, fy2 in falling_detections
                )

                if is_falling:
                    person['falling_frames'] += 1
                else:
                    person['falling_frames'] = max(0, person['falling_frames'] - 1)

                # Check if person is static
                if person['last_position'] is not None:
                    prev_x, prev_y = person['last_position']
                    curr_x = (x1 + x2) // 2
                    curr_y = (y1 + y2) // 2

                    movement = abs(curr_x - prev_x) + abs(curr_y - prev_y)

                    if movement < 10:  # Threshold for "static"
                        person['static_frames'] += 1
                    else:
                        person['static_frames'] = max(0, person['static_frames'] - 2)

                    person['last_position'] = (curr_x, curr_y)
                else:
                    person['last_position'] = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Classify person if they appear to be falling/static
                if classifier_enabled and (person['falling_frames'] > 5 or person['static_frames'] > 30):
                    # CRITICAL FIX: Extract person bounding box for classifier
                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size > 0:
                        classification = classify_person_bbox(person_crop)
                        classifier_calls += 1

                        if classification:
                            person['needs_help'] = classification['needs_help']

                # Calculate danger value
                person['danger_value'] = calculate_danger_value(person)
                max_danger = max(max_danger, person['danger_value'])

                # Trigger alert if danger > 1.0 and not already triggered
                if person['danger_value'] > 1.0 and not person['alert_triggered']:
                    trigger_alert_analysis(track_id, person, frame)

                # Reset alert if danger drops
                if person['danger_value'] < 0.5 and person['alert_triggered']:
                    person['alert_triggered'] = False

                # Draw bounding box with ID
                color = (0, 0, 255) if person['danger_value'] > 0.7 else \
                    (0, 165, 255) if person['danger_value'] > 0.3 else \
                        (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Unified font settings
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1

                # Draw person ID
                id_text = f"ID:{track_id}"
                cv2.putText(frame, id_text, (x1, y1 - 5),
                            font, font_scale, color, thickness)

                # Draw status on separate lines
                y_offset = y1 + 15

                if person['falling_frames'] > 0:
                    cv2.putText(frame, "FALLING", (x1, y_offset),
                                font, font_scale, (0, 0, 255), thickness)
                    y_offset += 15

                if person['needs_help']:
                    cv2.putText(frame, "NEEDS HELP", (x1, y_offset),
                                font, font_scale, (255, 0, 0), thickness)
                    y_offset += 15

                if person['danger_value'] > 0:
                    danger_text = f"Danger: {person['danger_value']:.2f}"
                    cv2.putText(frame, danger_text, (x1, y_offset),
                                font, font_scale, (0, 165, 255), thickness)
                    y_offset += 15

                if person['static_frames'] > 30:
                    static_text = f"Static: {person['static_frames'] / 30:.1f}s"
                    cv2.putText(frame, static_text, (x1, y_offset),
                                font, font_scale, (255, 255, 0), thickness)

            # Clean up old tracked persons
            current_ids = set()
            for t in tracked_objects:
                if len(t) >= 5:
                    current_ids.add(int(t[4]))  # track_id is at index 4

            for pid in list(person_tracking.keys()):
                if pid not in current_ids:
                    if frame_count - person_tracking[pid]['last_seen'] > 90:  # 3 seconds
                        del person_tracking[pid]

            # Draw info overlay
            info_text = [
                f"Frame: {frame_count}",
                f"Tracked Persons: {len(current_ids)}",
                f"Max Danger: {max_danger:.2f}",
            ]

            if classifier_enabled:
                info_text.append(f"Classifications: {classifier_calls}")

            # Unified font for info overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            info_scale = 0.4
            y_pos = 20
            for text in info_text:
                cv2.putText(frame, text, (10, y_pos),
                            font, info_scale, (255, 255, 255), 1)
                y_pos += 15

            # Draw alert overlay if active
            frame = draw_alert_overlay(frame)

            # Update danger display
            try:
                update_danger_score(max_danger, danger_label)
            except:
                pass

            # Save frame if recording
            if out:
                out.write(frame)

            # Display frame
            show_frame_on_canvas(frame)

            frame_count += 1

            # Schedule next frame
            delay = 33 if video_path == 0 else 10  # Camera vs video file
            after_id = root.after(delay, read_frame)

        except Exception as e:
            print(f"Error in read_frame: {e}")
            import traceback
            traceback.print_exc()
            cleanup_resources()

    def cleanup_resources():
        """Clean up resources."""
        print(f"\nCleanup Summary:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Classifications: {classifier_calls}")
        print(f"  Persons tracked: {len(person_tracking)}")

        # Save cache if we built new detections
        if not use_cache and cache_key and cached_detections:
            print(f"💾 Saving cache for faster next run...")
            try:
                cache_manager.save_detections(video_path, cache_key, cached_detections)
                print(f"✅ Cache saved: {len(cached_detections)} frames")
            except Exception as e:
                print(f"⚠️ Failed to save cache: {e}")

        # Print final person states
        for pid, data in person_tracking.items():
            if data['danger_value'] > 0:
                print(f"  Person {pid}: Danger={data['danger_value']:.2f}, Alert={data['alert_triggered']}")

        cap.release()
        if out:
            out.release()
        if after_id:
            try:
                root.after_cancel(after_id)
            except:
                pass
        cv2.destroyAllWindows()

    # Initialize canvas with a loading frame
    if display_width > 0 and display_height > 0:
        initial_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        cv2.putText(initial_frame, "Starting video...", (display_width // 4, display_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        show_frame_on_canvas(initial_frame)

    # Start processing
    print(f"Starting {'camera' if video_path == 0 else 'video'} processing...")
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height} -> {display_width}x{display_height}")
    print(f"FPS: {fps}")
    print(f"Classifier: {'ENABLED' if classifier_enabled else 'DISABLED'}")
    print(f"Cache: {'ENABLED (using cached)' if use_cache else 'ENABLED (building)' if cache_key else 'DISABLED'}")
    print(f"Alert System: ACTIVE (triggers at danger > 1.0)")

    # Start the frame reading loop
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
    root.title("Fall Detection System - Enhanced")

    canvas = tk.Canvas(root, width=800, height=600, bg='black')
    canvas.pack()

    danger_label = tk.Label(root, text="Danger Score: 0.00", font=("Arial", 14, "bold"))
    danger_label.pack()

    stop_event = Event()

    # Process video
    process_video(video_path, danger_label, canvas, root, stop_event)

    root.mainloop()