import threading
import tkinter as tk
from src.utils.video_processing import process_video

# Global stop event for controlling the camera processing
stop_event = threading.Event()


def update_main_frame_for_camera_analysis(main_frame, root):
    """Set up the interface for real-time camera analysis."""
    # Clear all controls in the current interface
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Show danger score label
    danger_label = tk.Label(main_frame, text="Danger Score: 0", font=("Arial", 20), fg="green")
    danger_label.pack(pady=10)

    # Create a canvas for displaying video output
    canvas = tk.Canvas(main_frame)
    canvas.pack()

    # Stop any existing video processing
    stop_event.set()  # Stop any ongoing camera processing
    stop_event.clear()  # Reset the event to allow new processing

    # Start video processing using the camera (camera index 0)
    threading.Thread(target=process_video, args=(0, danger_label, canvas, root, stop_event), daemon=True).start()
