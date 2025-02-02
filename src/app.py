import os
import cv2
import psutil
import threading
import tkinter as tk

from src.classifier_test.stuff import process_images
from src.gui.gui_analyze import AnalyzeReportGUI
from src.gui.gui_extract_frame import update_main_frame_for_extract_frames
from src.gui.gui_generate_labels import update_main_frame_for_generate_labels
from src.gui.gui_run_alert_test import update_main_frame_for_alert
from src.gui.gui_start_camera import update_main_frame_for_camera_analysis
from src.gui.gui_train_model import update_main_frame_for_train_model
from src.gui.gui_open_video_file import update_main_frame_for_fall_detection_video
from src.gui.gui_menubar import setup_menubar

from src.models.train_model import train_yolo_model
from src.utils.extract_frames import save_video_frames
from src.utils.video_processing import process_video
from src.utils.generate_labels_g import create_labels_using_yolo

# Add a global stop_event to be used for stopping video processing
stop_event = threading.Event()


def release_camera():
    """Release any open camera resources."""
    print("Releasing camera resources...")
    try:
        cap = cv2.VideoCapture(0)  # Assuming camera index 0
        if cap.isOpened():
            cap.release()
            print("Camera released.")
    except Exception as e:
        print(f"Error releasing camera: {e}")


def kill_all_background_tasks():
    """Kill all processes started by this application and release camera."""
    release_camera()  # Release the camera before killing processes
    stop_event.set()
    stop_event.clear()
    current_process = psutil.Process(os.getpid())
    for child in current_process.children(recursive=True):
        print(f"Killing process: {child.pid}")
        child.kill()


# Function placeholders for menu actions
def extract_frames():
    """Update main frame to allow extracting frames from video files."""
    update_main_frame_for_extract_frames(main_frame, extract_frames_callback)


def extract_frames_callback(video_paths, output_dir, frame_interval):
    """Callback function to handle frame extraction."""
    kill_all_background_tasks()
    for video_path in video_paths:
        threading.Thread(target=save_video_frames, args=(video_path, output_dir, frame_interval)).start()


def generate_labels():
    """Update main frame to allow generating labels for images."""
    update_main_frame_for_generate_labels(main_frame, generate_labels_callback)


def generate_labels_callback(image_dir, model_path):
    """Callback function to handle YOLO label generation."""
    kill_all_background_tasks()
    threading.Thread(target=create_labels_using_yolo, args=(image_dir, model_path)).start()


def train_model():
    """Update main frame to allow training YOLO model."""
    update_main_frame_for_train_model(main_frame, train_model_callback)


def train_model_callback(model_path, data_path, output_dir, run_name, epochs, imgsz, batch):
    """Callback function to handle YOLO model training."""
    kill_all_background_tasks()
    threading.Thread(target=train_yolo_model,
                     args=(model_path, data_path, output_dir, run_name, epochs, imgsz, batch)).start()


def analyze_report():
    analyze_report_callback()


def analyze_report_callback():
    """Callback function to display the Analyze Report GUI."""
    kill_all_background_tasks()  # Stop any ongoing processes

    # Ensure main_frame is valid and clear its contents
    if main_frame.winfo_exists():
        for widget in main_frame.winfo_children():
            widget.destroy()

    # Initialize the AnalyzeReportGUI on the cleared main_frame
    AnalyzeReportGUI(main_frame)


def open_video_file():
    """Open a file dialog to select a video file and set up the interface for analysis."""
    # Stop any ongoing camera processing
    update_main_frame_for_fall_detection_video(main_frame, lambda *args: process_video(*args, stop_event), root)


def open_video_file_callback(video_path):
    """Set up the interface for analysis after a video file is selected."""
    kill_all_background_tasks()
    if video_path:
        # Start video processing in a new thread, pass stop_event
        threading.Thread(target=process_video, args=(video_path, None, None, root, stop_event), daemon=True).start()


def start_camera():
    """Update the main frame to allow real-time camera analysis."""
    update_main_frame_for_camera_analysis(main_frame, root)  # Call the function to update the main frame


def start_camera_callback():
    """Callback function for starting real-time camera analysis."""
    kill_all_background_tasks()
    threading.Thread(target=process_video, args=(0, None, None, root, stop_event), daemon=True).start()


def alert_test():
    update_main_frame_for_alert(main_frame, alert_test_callback)


def alert_test_callback(image_dir, frame_skip, fall_duration_threshold):
    kill_all_background_tasks()

    threading.Thread(target=process_images, args=(image_dir, frame_skip, fall_duration_threshold)).start()


# Main application function
if __name__ == "__main__":
    root = tk.Tk()
    root.title("EYES-4-RESCUE")

    # Set the initial size of the window
    root.geometry("800x600")
    root.minsize(600, 400)

    # Create the main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Set up the menubar
    setup_menubar(root, extract_frames, generate_labels, train_model, open_video_file, start_camera, analyze_report,
                  alert_test)

    root.mainloop()
