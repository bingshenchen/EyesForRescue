# src/train/evaluation/fall_detection.py

import os
import cv2
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any

from ultralytics import YOLO
from tabulate import tabulate

# Import centralized configuration
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Color constants
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)

# Result indicators
CORRECT = "✅"
WRONG = "❌"
MISSED = "⚠️"

# Global variables
tracks_history = {}
fallen_tracker = {}
next_track_id = 0
detection_results = []

# Frame tolerance for evaluation
FRAME_TOLERANCE = 40

# Detection confidence threshold
CONFIDENCE_THRESHOLD_DETECTION = settings.CONFIDENCE_THRESHOLD


class FallDetectionEvaluator:
    """Centralized fall detection evaluator with configuration integration."""

    def __init__(self):
        """Initialize the evaluator with settings."""
        self.settings = settings
        self.models = self._load_models()
        self.reset_tracking_data()

    def _load_models(self) -> Dict[str, Any]:
        """Load required models based on configuration."""
        models = {}

        # Load YOLO detection model
        if self.settings.YOLO_MODEL_PATH.exists():
            models['detection'] = YOLO(str(self.settings.YOLO_MODEL_PATH))
            logger.info(f"Detection model loaded: {self.settings.YOLO_MODEL_PATH}")
        else:
            raise FileNotFoundError(f"Detection model not found: {self.settings.YOLO_MODEL_PATH}")

        # Load YOLO pose model
        if self.settings.POSE_MODEL_PATH.exists():
            models['pose'] = YOLO(str(self.settings.POSE_MODEL_PATH))
            logger.info(f"Pose model loaded: {self.settings.POSE_MODEL_PATH}")
        else:
            logger.warning(f"Pose model not found: {self.settings.POSE_MODEL_PATH}")
            models['pose'] = None

        # Load classifier
        if self.settings.CLASSIFIER_PATH.exists():
            models['classifier'] = joblib.load(str(self.settings.CLASSIFIER_PATH))
            logger.info(f"Classifier loaded: {self.settings.CLASSIFIER_PATH}")
        else:
            logger.warning(f"Classifier not found: {self.settings.CLASSIFIER_PATH}")
            models['classifier'] = None

        return models

    def reset_tracking_data(self):
        """Reset tracking data for new evaluation."""
        global tracks_history, fallen_tracker, detection_results
        tracks_history = {}
        fallen_tracker = {}
        detection_results = []

    def get_yolo_detections(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get YOLO detections from frame."""
        try:
            results = self.models['detection'].track(
                frame, verbose=False, persist=True, show=False
            )

            detections = {}
            for r in results:
                boxes = r.boxes
                if boxes is None or boxes.id is None:
                    continue

                for box_idx in range(len(boxes)):
                    xyxy = boxes.xyxy[box_idx].cpu().numpy().astype(int)
                    confidence = boxes.conf[box_idx].item()
                    cls = int(boxes.cls[box_idx].item())

                    # Only process person class (class 0)
                    if cls != 0 or confidence < CONFIDENCE_THRESHOLD_DETECTION:
                        continue

                    track_id = boxes.id[box_idx].item()
                    xmin, ymin, xmax, ymax = xyxy
                    width = xmax - xmin
                    height = ymax - ymin

                    detections.update({
                        "xmin": xmin, "ymin": ymin,
                        "width": width, "height": height,
                        "confidence": confidence, "track_id": track_id
                    })
                    break  # Only process first valid detection

            return detections if detections else None

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return None

    def has_position_changed(self, xmin: int, ymin: int, width: int, height: int,
                             track_id: int, threshold: int = 20) -> bool:
        """Check if position has changed significantly."""
        if track_id not in tracks_history:
            return True

        last_frames = min(10, len(tracks_history[track_id]))
        if last_frames == 0:
            return True

        last_positions = tracks_history[track_id][-last_frames:]

        for _, last_position in last_positions.iterrows():
            last_values = [
                last_position.get("xmin"), last_position.get("ymin"),
                last_position.get("width"), last_position.get("height")
            ]

            if None in last_values:
                return True

            current_values = [xmin, ymin, width, height]
            changes = [abs(curr - last) > threshold
                       for curr, last in zip(current_values, last_values)]

            if any(changes):
                return True

        return False

    def is_fall_detected(self, track_id: int, area_threshold: float = 2500,
                         y_threshold: float = 5) -> bool:
        """Detect fall based on tracking history."""
        if track_id not in tracks_history:
            return False

        track_data = tracks_history[track_id]
        frames_needed = 5

        if len(track_data) < frames_needed:
            return False

        recent_data = track_data[-frames_needed:]
        aspect_ratios = recent_data["aspect_ratio"]
        areas = recent_data["area"]
        y_positions = recent_data["bbox_y_center"]

        avg_area_change = areas.diff().abs().mean()
        avg_y_position_change = y_positions.diff().mean()

        area_change_trigger = avg_area_change > area_threshold
        y_position_trigger = avg_y_position_change > y_threshold

        fall_detected = area_change_trigger and y_position_trigger

        # Update tracking data
        track_data.loc[recent_data.index, "fall_detected"] = fall_detected

        if fall_detected and track_id not in fallen_tracker:
            fallen_tracker[track_id] = {"static_frames": 0, "fall_detected": True}
        elif fall_detected:
            fallen_tracker[track_id]["fall_detected"] = True

        return fall_detected

    def extract_features_from_frame(self, frame: np.ndarray) -> str:
        """Extract features and classify using pose model and classifier."""
        if self.models['pose'] is None or self.models['classifier'] is None:
            return "fine"

        try:
            results = self.models['pose'](frame, verbose=False)
            keypoints = results[0].keypoints

            if keypoints is not None:
                features = keypoints.xy.cpu().numpy().flatten()
            else:
                features = np.zeros(34)

            # Ensure correct feature size
            if features.size == 34:
                features = features.reshape(1, -1)
                prediction = self.models['classifier'].predict(features)[0]
                label_map = {0: 'fine', 1: 'needshelp'}
                return label_map.get(prediction, 'fine')

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")

        return "fine"

    def process_video(self, video_path: str, video_idx: int,
                      seconds_til_alert: int = 5) -> bool:
        """Process a single video for fall detection evaluation."""
        video_name = Path(video_path).name
        logger.info(f"Processing video: {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return False

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0
        has_alerted = False

        # Define columns for tracking data
        columns = ["frame_idx", "xmin", "ymin", "width", "height", "area",
                   "aspect_ratio", "fall_detected", "bbox_y_center", "alert_triggered"]

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                detections = self.get_yolo_detections(frame)

                if detections is None:
                    continue

                # Extract detection data
                xmin, ymin = detections["xmin"], detections["ymin"]
                width, height = detections["width"], detections["height"]
                confidence = detections["confidence"]
                track_id = detections["track_id"]

                # Calculate derived values
                area = abs(width * height)
                aspect_ratio = abs(width / height) if height != 0 else 1.0
                bbox_y_center = ymin + height / 2

                # Update tracking history
                if track_id not in tracks_history:
                    tracks_history[track_id] = pd.DataFrame(columns=columns)

                new_data = pd.DataFrame([[
                    frame_idx, xmin, ymin, width, height, area,
                    aspect_ratio, False, bbox_y_center, False
                ]], columns=columns)

                tracks_history[track_id] = pd.concat([
                    tracks_history[track_id], new_data
                ], ignore_index=True)

                # Detect fall
                has_fallen = self.is_fall_detected(track_id)
                color = GREEN
                static_for_seconds = 0

                # Process fall detection logic
                if track_id in fallen_tracker:
                    if not tracks_history[track_id]["alert_triggered"].any():
                        fallen_tracker[track_id]["fall_detected"] = (
                                fallen_tracker[track_id].get("fall_detected", False) or has_fallen
                        )

                        is_static = not self.has_position_changed(xmin, ymin, width, height, track_id)

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

                        # Trigger alert
                        if (fallen_tracker[track_id]["fall_detected"] and
                                static_frames >= fps * seconds_til_alert):

                            label = self.extract_features_from_frame(frame)
                            if label == "needshelp":
                                has_alerted = self.trigger_alert(track_id, seconds_til_alert,
                                                                 frame_idx, video_idx)
                                if has_alerted:
                                    fallen_tracker[track_id]["fall_detected"] = False
                                    fallen_tracker[track_id]["static_frames"] = 0
                                    tracks_history[track_id]["alert_triggered"] = True
                                    color = RED
                    else:
                        color = RED

                # Draw visualization
                frame = self.draw_info_on_frame(frame, track_id, xmin, ymin,
                                                width, height, color, static_for_seconds)

                # Display frame (optional)
                if self.settings.DEBUG_MODE:
                    cv2.imshow("Fall Detection Evaluation", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()

        if not has_alerted:
            detection_results[video_idx]["actual"] = -1

        return True

    def trigger_alert(self, track_id: int, seconds_til_alert: int,
                      frame_idx: int, video_idx: int) -> bool:
        """Trigger alert and update results."""
        logger.info(f"Alert triggered for track {track_id} at frame {frame_idx}")
        detection_results[video_idx]["actual"] = frame_idx
        return True

    def draw_info_on_frame(self, frame: np.ndarray, track_id: int,
                           xmin: int, ymin: int, width: int, height: int,
                           color: tuple, static_seconds: float) -> np.ndarray:
        """Draw tracking information on frame."""
        static_text = f"Static: {static_seconds:.1f}s" if static_seconds > 0 else ""

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), color, 2)

        # Draw label
        label = f"ID: {track_id} {static_text}"
        cv2.putText(frame, label, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def show_performance_table(self):
        """Display performance evaluation table."""
        df = pd.DataFrame(detection_results)

        # Calculate evaluation metrics
        df['correct'] = df.apply(
            lambda row: CORRECT if abs(row['ground_truth'] - row['actual']) <= FRAME_TOLERANCE
            else "", axis=1
        )
        df['wrong'] = df.apply(
            lambda row: WRONG if (row['actual'] != -1 and
                                  abs(row['ground_truth'] - row['actual']) > FRAME_TOLERANCE)
            else "", axis=1
        )
        df['missed'] = df.apply(
            lambda row: MISSED if (row['actual'] == -1 and row['ground_truth'] != -1)
            else "", axis=1
        )

        # Reorder columns
        df = df[['video', 'ground_truth', 'actual', 'correct', 'wrong', 'missed']]

        # Display table
        print("\n" + "=" * 60)
        print("FALL DETECTION EVALUATION RESULTS")
        print("=" * 60)
        print(tabulate(df, headers='keys', tablefmt='grid'))

        # Calculate summary statistics
        total_videos = len(df)
        correct_detections = len(df[df['correct'] == CORRECT])
        wrong_detections = len(df[df['wrong'] == WRONG])
        missed_detections = len(df[df['missed'] == MISSED])

        accuracy = correct_detections / total_videos if total_videos > 0 else 0
        precision = correct_detections / (correct_detections + wrong_detections) if (
                                                                                                correct_detections + wrong_detections) > 0 else 0
        recall = correct_detections / (correct_detections + missed_detections) if (
                                                                                              correct_detections + missed_detections) > 0 else 0

        print(f"\nSUMMARY STATISTICS:")
        print(f"Total Videos: {total_videos}")
        print(f"Correct Detections: {correct_detections}")
        print(f"Wrong Detections: {wrong_detections}")
        print(f"Missed Detections: {missed_detections}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print("=" * 60)

        # Save results
        self.save_evaluation_results(df)

    def save_evaluation_results(self, results_df: pd.DataFrame):
        """Save evaluation results to file."""
        try:
            output_dir = self.settings.EVALUATION_RESULTS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save CSV
            csv_path = output_dir / f"fall_detection_evaluation_{timestamp}.csv"
            results_df.to_csv(csv_path, index=False)

            # Save detailed report
            report_path = output_dir / f"fall_detection_report_{timestamp}.txt"
            with open(report_path, 'w') as f:
                f.write("Fall Detection Evaluation Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Evaluation completed at: {datetime.now()}\n")
                f.write(f"Configuration used:\n")
                f.write(f"  Detection Model: {self.settings.YOLO_MODEL_PATH}\n")
                f.write(f"  Pose Model: {self.settings.POSE_MODEL_PATH}\n")
                f.write(f"  Classifier: {self.settings.CLASSIFIER_PATH}\n")
                f.write(f"  Confidence Threshold: {self.settings.CONFIDENCE_THRESHOLD}\n")
                f.write(f"  Frame Tolerance: {FRAME_TOLERANCE}\n\n")
                f.write("Results:\n")
                f.write(tabulate(results_df, headers='keys', tablefmt='grid'))

            logger.info(f"Evaluation results saved to: {csv_path}")
            logger.info(f"Evaluation report saved to: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")


def run_fall_detection_evaluation(video_paths: List[str], ground_truths: List[int]):
    """Run fall detection evaluation on multiple videos."""
    global detection_results

    # Initialize evaluator
    evaluator = FallDetectionEvaluator()

    # Initialize detection results
    detection_results = [
        {"video": Path(path).name, "ground_truth": gt}
        for path, gt in zip(video_paths, ground_truths)
    ]

    # Process each video
    for idx, video_path in enumerate(video_paths):
        evaluator.reset_tracking_data()
        success = evaluator.process_video(video_path, idx, seconds_til_alert=1)

        if not success:
            logger.warning(f"Failed to process video: {video_path}")

    # Show results
    evaluator.show_performance_table()


class FallDetectionGUI:
    """GUI for fall detection evaluation."""

    def __init__(self):
        self.settings = settings
        self.video_paths = []
        self.ground_truths = []

    def create_gui(self):
        """Create the evaluation GUI."""
        root = tk.Tk()
        root.title("Fall Detection Evaluation Tool")
        root.geometry("600x400")

        # Title
        title_label = tk.Label(root, text="Fall Detection Evaluation",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # Configuration info
        config_frame = tk.Frame(root, relief=tk.RIDGE, bd=1)
        config_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(config_frame, text="Configuration:",
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=5)

        config_text = (
            f"Detection Model: {self.settings.YOLO_MODEL_PATH.name}\n"
            f"Pose Model: {self.settings.POSE_MODEL_PATH.name}\n"
            f"Classifier: {self.settings.CLASSIFIER_PATH.name}\n"
            f"Confidence: {self.settings.CONFIDENCE_THRESHOLD}"
        )

        tk.Label(config_frame, text=config_text, justify=tk.LEFT,
                 font=("Arial", 9)).pack(anchor=tk.W, padx=20, pady=5)

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack(pady=30)

        tk.Button(button_frame, text="Load Videos",
                  command=self.load_videos, font=("Arial", 12),
                  bg="#3498db", fg="white", padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(button_frame, text="Set Ground Truths",
                  command=self.set_ground_truths, font=("Arial", 12),
                  bg="#f39c12", fg="white", padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        tk.Button(button_frame, text="Start Evaluation",
                  command=self.start_evaluation, font=("Arial", 12),
                  bg="#27ae60", fg="white", padx=20, pady=10).pack(side=tk.LEFT, padx=10)

        # Status
        self.status_label = tk.Label(root, text="Ready",
                                     font=("Arial", 10), fg="#7f8c8d")
        self.status_label.pack(pady=20)

        return root

    def load_videos(self):
        """Load video files for evaluation."""
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )

        if files:
            self.video_paths = list(files)
            self.status_label.config(text=f"Loaded {len(files)} video(s)")
            messagebox.showinfo("Videos Loaded", f"Loaded {len(files)} video(s)")

    def set_ground_truths(self):
        """Set ground truth frames for evaluation."""
        if not self.video_paths:
            messagebox.showerror("Error", "Please load videos first")
            return

        ground_truth_input = simpledialog.askstring(
            "Ground Truth Frames",
            f"Enter ground truth frames for {len(self.video_paths)} videos\n"
            "(comma-separated):"
        )

        if ground_truth_input:
            try:
                frames = [int(x.strip()) for x in ground_truth_input.split(',')]
                if len(frames) != len(self.video_paths):
                    raise ValueError("Number of frames must match number of videos")

                self.ground_truths = frames
                self.status_label.config(text=f"Ground truths set for {len(frames)} videos")
                messagebox.showinfo("Success", "Ground truths set successfully")

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")

    def start_evaluation(self):
        """Start the evaluation process."""
        if not self.video_paths or not self.ground_truths:
            messagebox.showerror("Error", "Please load videos and set ground truths first")
            return

        try:
            self.status_label.config(text="Running evaluation...")
            run_fall_detection_evaluation(self.video_paths, self.ground_truths)
            self.status_label.config(text="Evaluation completed")
            messagebox.showinfo("Complete", "Evaluation completed. Check console for results.")

        except Exception as e:
            error_msg = f"Evaluation failed: {e}"
            logger.error(error_msg)
            self.status_label.config(text="Evaluation failed")
            messagebox.showerror("Error", error_msg)


def main():
    """Main function to run the evaluation."""
    logger.info("Starting Fall Detection Evaluation")
    logger.info(f"Using configuration: {settings.PROJECT_ROOT}")

    # Create and run GUI
    gui = FallDetectionGUI()
    root = gui.create_gui()
    root.mainloop()


if __name__ == "__main__":
    main()