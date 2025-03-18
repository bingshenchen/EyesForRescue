import math
import os
from pathlib import Path
from tkinter import messagebox, filedialog
import tkinter as tk
import cv2
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO
from tabulate import tabulate

RED = (0, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)

CORRECT = "✅"
WRONG = "❌"
MISSED = "⚠️"

tracks_history = {}
fallen_tracker = {}

next_track_id = 0
load_dotenv()
CONFIDENCE_THRESHOLD_DETECTION = 0.4
classifier_location = Path(os.getenv('PROJECT_ROOT')) / "assets" / "classifier" / "classifier.pkl"
load_dotenv()

model_pose = os.getenv('MODEL_POSE')
model_base = os.getenv('MODEL_BASE')
model_file = os.getenv('MODEL_FILE')
pose_model_file = os.getenv('MODEL_POSE')
if not model_base or not model_file:
    raise EnvironmentError("MODEL_BASE and MODEL_FILE environment variables must be set.")
model_path = os.path.join(model_base, model_file)
pose_model_path = os.path.join(model_base, pose_model_file)

model = YOLO(model_path)
pose_model = YOLO(pose_model_path)

GROUND_TRUTHS = [int(x) for x in os.getenv('GROUND_TRUTHS').split(',')]
detection_results = []
FRAME_TOLERANCE = 40  # How many frames the prediction can be off to be considered correct

columns = ["frame_idx", "xmin", "ymin", "width", "height", "area", "aspect_ratio", "fall_detected", "bbox_y_center",
           "alert_triggered"]
if classifier_location.is_file():
    print("Loading classifier from file")
    clf = joblib.load(classifier_location)
else:
    print("Classifier file not found")


def get_yolo_detections(frame):
    results = model.track(frame, verbose=False, persist=True, show=False)

    # Uncomment to show the image with bounding boxes
    # results = model(image, show=True)
    detections = {}

    # Iterate over each result (frame) in the results list
    for r in results:
        boxes = r.boxes  # Access the Boxes object
        for box in range(len(boxes)):
            if boxes.id is None:
                continue
            # Extract box properties
            xyxy = boxes.xyxy[box].cpu().numpy().astype(int)  # Convert to numpy and integer
            confidence = boxes.conf[box].item()  # Confidence score
            cls = int(boxes.cls[box].item())  # Class ID

            if cls != 0:
                continue

            if confidence < CONFIDENCE_THRESHOLD_DETECTION:
                continue

            track_id = boxes.id[box].item()

            xmin, ymin, xmax, ymax = xyxy  # xyxy format
            width = xmax - xmin
            height = ymax - ymin
            detections["xmin"] = xmin
            detections["ymin"] = ymin
            detections["width"] = width
            detections["height"] = height
            detections["confidence"] = confidence
            detections["track_id"] = track_id

    return detections


def has_position_changed(xmin, ymin, width, height, track_id):
    global tracks_history
    if track_id not in tracks_history:
        return True

    last_10_frames = min(10, len(tracks_history[track_id]))
    last_positions = tracks_history[track_id][-last_10_frames:]

    for i, last_position in last_positions.iterrows():
        last_xmin = last_position.get("xmin", None)
        last_ymin = last_position.get("ymin", None)
        last_width = last_position.get("width", None)
        last_height = last_position.get("height", None)

        if None in (last_xmin, last_ymin, last_width, last_height):
            return True

        x_trigger = abs(xmin - last_xmin) > 20
        y_trigger = abs(ymin - last_ymin) > 20
        width_trigger = abs(width - last_width) > 20
        height_trigger = abs(height - last_height) > 20

        if x_trigger or y_trigger or width_trigger or height_trigger:
            return True

    return False


def is_fall_detected(track_id):
    global tracks_history
    global fallen_tracker

    if track_id not in tracks_history:
        raise ValueError(f"Track ID {track_id} not found in tracks history.")

    track_data = tracks_history[track_id]
    frames_needed = 5  # Number of recent frames to analyze

    if len(track_data) < frames_needed:
        return False

    tracking_data_last_x_frames = track_data[-frames_needed:]

    aspect_ratios = tracking_data_last_x_frames["aspect_ratio"]
    areas = tracking_data_last_x_frames["area"]
    y_positions = tracking_data_last_x_frames["bbox_y_center"]

    avg_area_change = areas.diff().abs().mean()
    avg_y_position_change = y_positions.diff().mean()

    area_threshold = 2500  # original 100
    vertical_movement_threshold = 5  # original 10

    area_change_trigger = avg_area_change > area_threshold
    y_position_trigger = avg_y_position_change > vertical_movement_threshold

    fall_criteria_met = (
            area_change_trigger
            and y_position_trigger
    )

    track_data.loc[tracking_data_last_x_frames.index, "fall_detected"] = fall_criteria_met

    if fall_criteria_met:
        # Ensure tracker is initialized with "fall_detected"
        if track_id not in fallen_tracker:
            fallen_tracker[track_id] = {"static_frames": 0, "fall_detected": True}
        else:
            fallen_tracker[track_id]["fall_detected"] = True

    return fall_criteria_met


def extract_features_from_frame_and_return_label(frame):
    """
    Extract pose keypoints or other features from the given frame using the YOLO pose model.

    :param frame: The video frame to extract features from.
    :return: Extracted features (e.g., keypoints).
    """
    # Get pose features using YOLO model
    results = pose_model(frame, verbose=False)  # Perform pose detection

    # Assuming you want to extract the pose keypoints (this depends on the model output)
    keypoints = results[0].keypoints  # First element in results, and get the keypoints

    # Check if keypoints are present
    if keypoints is not None:
        # Extract keypoint coordinates (xy)
        keypoints_xy = keypoints.xy.cpu().numpy()  # Convert tensor to numpy array

        # Flatten keypoints (x, y) values
        features = keypoints_xy.flatten()  # Flatten the coordinates into a 1D array
    else:
        features = np.zeros(34)  # If no keypoints detected, use a zero vector (adjust size if necessary)
    if features.size == 34:
        features = features.reshape(1, -1)
        numerical_label = clf.predict(features)[0]  # Get the numerical prediction

        # Map numerical label to human-readable label
        label_map = {0: 'fine', 1: 'needshelp'}
        label = label_map.get(numerical_label, 'unknown')  # Default to 'unknown' if mapping fails

        # print(f"Numerical label: {numerical_label}, Decoded label: {label}")
        return label
    label = "fine"
    return label


def process_videos(video_paths, detection_results, seconds_til_alert=5):
    global tracks_history
    global fallen_tracker

    for idx, video_path in enumerate(video_paths):
        has_already_alerted = False
        video_name = os.path.basename(video_path)
        print(f"Processing video: {video_name}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            detections = get_yolo_detections(frame)

            if detections is None:
                continue

            xmin, ymin, width, height, confidence, track_id = (
                detections.get("xmin", None),
                detections.get("ymin", None),
                detections.get("width", None),
                detections.get("height", None),
                detections.get("confidence", None),
                detections.get("track_id", None),
            )

            if None in (xmin, ymin, width, height, confidence, track_id):
                continue

            area = abs(width * height)
            aspect_ratio = abs(width / height)
            bbox_y_center = ymin + height / 2

            if track_id not in tracks_history:
                new_track = pd.DataFrame(
                    data=[[frame_idx, xmin, ymin, width, height, area, aspect_ratio, False, bbox_y_center, False]],
                    columns=columns
                )
                tracks_history[track_id] = new_track
            else:
                new_track = pd.DataFrame(
                    data=[[frame_idx, xmin, ymin, width, height, area, aspect_ratio, False, bbox_y_center, False]],
                    columns=columns
                )
                tracks_history[track_id] = pd.concat([tracks_history[track_id], new_track], ignore_index=True)

            # Detect fall
            has_fallen = is_fall_detected(track_id)

            color = GREEN

            static_for_seconds = 0

            if track_id in fallen_tracker:
                if not tracks_history[track_id]["alert_triggered"].any():
                    fallen_tracker[track_id]["fall_detected"] = fallen_tracker[track_id].get("fall_detected",
                                                                                             False) or has_fallen

                    # Check if person is static
                    is_static = not has_position_changed(xmin, ymin, width, height, track_id)
                    if is_static:
                        fallen_tracker[track_id]["static_frames"] += 1
                    else:
                        fallen_tracker[track_id]["static_frames"] = 0

                    if fallen_tracker[track_id]["fall_detected"]:
                        color = YELLOW

                    if fallen_tracker[track_id]["fall_detected"] and is_static:
                        color = ORANGE
                        static_for_seconds = fallen_tracker[track_id]["static_frames"] / fps

                    static_frames = fallen_tracker[track_id]["static_frames"]

                    # Trigger alert regardless of delay between fall and static state
                    if fallen_tracker[track_id]["fall_detected"] and static_frames >= fps * seconds_til_alert:
                        label = extract_features_from_frame_and_return_label(frame)
                        print(f"Label: {label}")
                        if label == "needshelp":
                            has_already_alerted = trigger_alert(frame, track_id, seconds_til_alert, frame_idx, idx)
                            print(f"Alert on frame {frame_idx}")
                            # Reset state after triggering alert
                            fallen_tracker[track_id]["fall_detected"] = False
                            fallen_tracker[track_id]["static_frames"] = 0
                            tracks_history[track_id]["alert_triggered"] = True

                            color = RED

                else:
                    color = RED

            # Draw bounding box and information
            frame = draw_info_on_frame(frame, track_id, xmin, ymin, width, height, color, static_for_seconds)

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        if not has_already_alerted:
            detection_results[idx]["actual"] = -1

    cv2.destroyAllWindows()


def trigger_alert(frame, track_id, seconds_til_alert, frame_idx, idx) -> bool:
    label = extract_features_from_frame_and_return_label(frame)
    if label == "needshelp":
        print(
            f"Alert triggered for track ID {track_id}: Person has been static for {seconds_til_alert} seconds after falling."
        )
        detection_results[idx]["actual"] = frame_idx
        return True
    return False


def draw_info_on_frame(frame, track_id, xmin, ymin, width, height, color, static_for_seconds):
    static_seconds_text = f"Static for {static_for_seconds:.2f} seconds" if static_for_seconds > 0 else ""
    cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), color, thickness=2)
    cv2.putText(frame, f"ID: {track_id} {static_seconds_text}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                2)
    return frame


def show_performance_table(detection_results):
    df = pd.DataFrame(detection_results)

    df['correct'] = df.apply(
        lambda row: CORRECT if abs(row['ground_truth'] - row['actual']) <= FRAME_TOLERANCE else "",
        axis=1
    )
    df['wrong'] = df.apply(
        lambda row: WRONG if row['actual'] != -1 and abs(row['ground_truth'] - row['actual']) > FRAME_TOLERANCE else "",
        axis=1
    )
    df['missed'] = df.apply(
        lambda row: MISSED if row['actual'] == -1 and row['ground_truth'] != -1 else "",
        axis=1
    )

    # Reorder columns for display
    df = df[['video', 'ground_truth', 'actual', 'correct', 'wrong', 'missed']]

    # Generate a nicely formatted table
    print(tabulate(df, headers='keys', tablefmt='grid'))


def run_fall_detection(video_paths, ground_truths):
    global detection_results

    # Initialize detection results
    detection_results = [{"video": os.path.basename(path), "ground_truth": gt} for path, gt in
                         zip(video_paths, ground_truths)]

    # Run processing
    process_videos(video_paths, detection_results, seconds_til_alert=1)

    # Show the performance table
    show_performance_table(detection_results)


def gui_main():
    video_paths = []
    ground_truths = []

    def load_videos():
        nonlocal video_paths
        files = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if files:
            video_paths = list(files)
            messagebox.showinfo("Videos Loaded", f"Loaded {len(files)} video(s).")

    def enter_ground_truths():
        nonlocal ground_truths
        if not video_paths:
            messagebox.showerror("Error", "No videos loaded. Please load videos first.")
            return

        ground_truth_input = tk.simpledialog.askstring(
            "Ground Truth Frames",
            f"Enter ground truth frames for {len(video_paths)} videos, separated by commas:"
        )
        if ground_truth_input:
            try:
                frames = list(map(int, ground_truth_input.split(',')))
                if len(frames) != len(video_paths):
                    raise ValueError("Number of frames does not match number of videos.")
                ground_truths = frames
                messagebox.showinfo("Ground Truths Entered", f"Ground truths for {len(frames)} videos saved.")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")

    def run_detection():
        if not video_paths or not ground_truths:
            messagebox.showerror("Error", "Please load videos and ground truths first.")
            return

        try:
            run_fall_detection(video_paths, ground_truths)
            messagebox.showinfo("Processing Complete", "Results have been printed to the terminal.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {e}")

    # Tkinter GUI
    root = tk.Tk()
    root.title("Fall Detection Tool")

    # Buttons
    load_videos_button = tk.Button(root, text="Load Videos", command=load_videos)
    load_videos_button.pack(pady=10)

    enter_ground_truths_button = tk.Button(root, text="Enter Ground Truth Frames", command=enter_ground_truths)
    enter_ground_truths_button.pack(pady=10)

    run_detection_button = tk.Button(root, text="Run Detection", command=run_detection)
    run_detection_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    gui_main()

# if __name__ == "__main__":
#     video_paths = os.getenv("VIDEO_PATHS").split(',')
#     formatted_video_paths = "\n".join(video_paths)
#
#     video_names = [os.path.basename(video_path) for video_path in video_paths]
#     formatted_video_names = "\n".join(video_names)
#     print(f"Running fall detection on videos:\n{formatted_video_names}")
#
#     for idx, video_name in enumerate(video_names):
#         detection_results.append({"video": video_name, "ground_truth": GROUND_TRUTHS[idx]})
#
#     process_videos(video_paths, seconds_til_alert=1)
#     show_performance_table()
