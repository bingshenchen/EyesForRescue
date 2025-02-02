from tkinter import filedialog, Label, Canvas
import threading

# Global stop event for controlling the video processing
stop_event = threading.Event()


def update_main_frame_for_fall_detection_video(main_frame, process_video, root):
    """Open a file dialog to select a video file and display a new interface for analysis."""
    # Clear all controls in the current interface
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Select video file
    video_path = filedialog.askopenfilename()

    if video_path:
        # Show danger score label
        danger_label = Label(main_frame, text="Danger Score: 0", font=("Arial", 20), fg="green")
        danger_label.pack(pady=10)

        # Create a canvas for displaying video output
        canvas = Canvas(main_frame, width=640, height=480, bg='black')
        canvas.pack()

        # Stop any existing video processing
        stop_event.set()  # Stop any ongoing video processing
        stop_event.clear()  # Reset the event to allow new processing

        # Start video processing in a new thread to avoid interface freezing
        threading.Thread(target=process_video, args=(video_path, danger_label, canvas, root, stop_event),
                         daemon=True).start()
