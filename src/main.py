# src/main.py

import os
import cv2
import psutil
import threading
import tkinter as tk
from tkinter import messagebox

# Import GUI modules with new separate window functionality
from src.gui.gui_analyze import AnalyzeReportGUI
from src.gui.gui_extract_frame import update_main_frame_for_extract_frames
from src.gui.gui_generate_labels import update_main_frame_for_generate_labels
from src.gui.gui_start_camera import update_main_frame_for_camera_analysis
from src.gui.gui_train_model import update_main_frame_for_train_model
from src.gui.gui_open_video_file import update_main_frame_for_fall_detection_video
from src.gui.gui_menubar import setup_menubar

# Import core functionality
from src.train.models.train_model import train_yolo_model
from src.core.utils.video_processor import process_video
from config.settings import get_settings

# Add a global stop_event to be used for stopping video processing
stop_event = threading.Event()


def release_camera():
    """Release any open camera resources."""
    print("Releasing camera resources...")
    try:
        # Try to release multiple camera indices
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                print(f"Camera {i} released.")
    except Exception as e:
        print(f"Error releasing cameras: {e}")


def kill_all_background_tasks():
    """Kill all processes started by this application and release camera."""
    release_camera()
    stop_event.set()

    # Give some time for threads to stop gracefully
    threading.Event().wait(0.5)

    stop_event.clear()

    # Kill child processes
    try:
        current_process = psutil.Process(os.getpid())
        for child in current_process.children(recursive=True):
            print(f"Killing process: {child.pid}")
            child.kill()
    except Exception as e:
        print(f"Error killing processes: {e}")


# Function placeholders for menu actions
def extract_frames():
    """Update main frame to allow extracting frames from video files."""
    update_main_frame_for_extract_frames(main_frame, extract_frames_callback)


def extract_frames_callback(video_paths, output_dir, frame_interval):
    """Callback function to handle frame extraction."""
    kill_all_background_tasks()

    # Import here to avoid circular imports
    from src.core.utils.extract_frames import save_video_frames

    for video_path in video_paths:
        if video_path.strip():  # Only process non-empty paths
            threading.Thread(target=save_video_frames,
                             args=(video_path, output_dir, frame_interval),
                             daemon=True).start()


def generate_labels():
    """Update main frame to allow generating labels for images."""
    update_main_frame_for_generate_labels(main_frame, generate_labels_callback)


def generate_labels_callback(image_dir, model_path):
    """Callback function to handle YOLO label generation."""
    kill_all_background_tasks()

    # Import here to avoid circular imports
    from src.core.utils.generate_labels_g import create_labels_using_yolo

    if image_dir.strip() and model_path.strip():
        threading.Thread(target=create_labels_using_yolo,
                         args=(image_dir, model_path),
                         daemon=True).start()


def train_model():
    """Update main frame to allow training YOLO model."""
    update_main_frame_for_train_model(main_frame, train_model_callback)


def train_model_callback(model_path, data_path, output_dir, epochs, imgsz, batch):
    """Callback function to handle YOLO model training."""
    kill_all_background_tasks()

    if all([model_path.strip(), data_path.strip(), output_dir.strip()]):
        threading.Thread(target=train_yolo_model,
                         args=(model_path, data_path, output_dir, epochs, imgsz, batch),
                         daemon=True).start()


def analyze_report():
    """Display the analyze report GUI."""
    analyze_report_callback()


def analyze_report_callback():
    """Callback function to display the Analyze Report GUI."""
    kill_all_background_tasks()

    # Ensure main_frame is valid and clear its contents
    if main_frame.winfo_exists():
        for widget in main_frame.winfo_children():
            widget.destroy()

        # Initialize the AnalyzeReportGUI on the cleared main_frame
        AnalyzeReportGUI(main_frame)


def open_video_file():
    """
    Open video file selection interface.
    This will now create a separate window for video processing.
    """
    try:
        update_main_frame_for_fall_detection_video(main_frame, process_video, root)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open video interface: {str(e)}")


def start_camera():
    """
    Start camera interface.
    This will now create a separate window for camera processing.
    """
    try:
        update_main_frame_for_camera_analysis(main_frame, root)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start camera interface: {str(e)}")


def on_closing():
    """Handle application closing."""
    try:
        # Stop all video processing
        kill_all_background_tasks()

        # Give threads time to clean up
        threading.Event().wait(0.5)

        # Destroy main window
        root.destroy()

    except Exception as e:
        print(f"Error during application shutdown: {e}")
        # Force exit if needed
        os._exit(0)


def setup_application():
    """Setup application with configuration validation."""
    try:
        # Get and validate configuration
        config = get_settings()

        # Print configuration summary
        print("EyesForRescue Application Starting...")
        print("=" * 40)
        print(f"Project Root: {config.PROJECT_ROOT}")
        print(f"YOLO Model: {config.YOLO_MODEL_PATH}")
        print(f"Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
        print(f"Cache Enabled: {config.CACHE_ENABLED}")
        print("=" * 40)

        # Create necessary directories
        config.create_directories()

        # Validate models (warn but don't stop if missing)
        if not config.validate_models():
            print("⚠️  Warning: Some model files are missing.")
            print("   The application will start but some features may not work.")

        return True

    except Exception as e:
        messagebox.showerror("Configuration Error",
                             f"Failed to setup application:\n{str(e)}\n\n"
                             "Please check your configuration files.")
        return False


def create_welcome_screen():
    """Create welcome screen in main frame."""
    # Clear main frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Welcome title
    title_label = tk.Label(main_frame,
                           text="EYES-4-RESCUE",
                           font=("Arial", 24, "bold"),
                           fg="#2c3e50")
    title_label.pack(pady=50)

    # Subtitle
    subtitle_label = tk.Label(main_frame,
                              text="AI-Powered Fall Detection System",
                              font=("Arial", 14),
                              fg="#7f8c8d")
    subtitle_label.pack(pady=10)

    # Description
    desc_text = """
Welcome to Eyes-4-Rescue, an advanced fall detection system.

Use the menu bar above to access different features:
• Detection: Analyze videos or start live camera monitoring
• Data Preparation: Extract frames and generate labels
• Training: Train custom YOLO models
• Analysis: Generate performance reports

Get started by selecting a feature from the menu!
    """

    desc_label = tk.Label(main_frame,
                          text=desc_text,
                          font=("Arial", 11),
                          fg="#34495e",
                          justify=tk.CENTER)
    desc_label.pack(pady=30, padx=50)

    # Version and status info
    config = get_settings()
    status_frame = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RIDGE, bd=1)
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

    status_text = f"Model: {config.YOLO_MODEL_PATH.name} | Confidence: {config.CONFIDENCE_THRESHOLD} | Cache: {'ON' if config.CACHE_ENABLED else 'OFF'}"
    status_label = tk.Label(status_frame, text=status_text,
                            font=("Arial", 9), bg="#ecf0f1", fg="#7f8c8d")
    status_label.pack(pady=10)


# Main application function
if __name__ == "__main__":
    # Setup application configuration
    if not setup_application():
        exit(1)

    # Create main window
    root = tk.Tk()
    root.title("EYES-4-RESCUE - AI Fall Detection System")

    # Set window properties
    root.geometry("1000x700")
    root.minsize(800, 600)

    # Configure window grid
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    # Create the main frame
    main_frame = tk.Frame(root)
    main_frame.grid(row=0, column=0, sticky="nsew")
    main_frame.rowconfigure(0, weight=1)
    main_frame.columnconfigure(0, weight=1)

    # Set up the menubar
    setup_menubar(root, extract_frames, generate_labels, train_model,
                  open_video_file, start_camera, analyze_report)

    # Create welcome screen
    create_welcome_screen()

    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the application
    try:
        print("Starting EYES-4-RESCUE application...")
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        on_closing()
    except Exception as e:
        print(f"Unexpected error: {e}")
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{str(e)}")
        on_closing()