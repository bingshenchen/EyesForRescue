# src/gui/gui_open_video_file.py

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from PIL import Image, ImageTk
import cv2

# Global stop event for controlling the video processing
stop_event = threading.Event()


class FallDetectionVideoWindow:
    """Separate window for fall detection video processing."""

    def __init__(self, parent, process_video_callback):
        self.parent = parent
        self.process_video_callback = process_video_callback
        self.video_window = None
        self.canvas = None
        self.danger_label = None
        self.control_frame = None
        self.is_processing = False

    def create_video_window(self, video_path):
        """Create a new window for video processing."""
        if self.video_window and self.video_window.winfo_exists():
            self.video_window.destroy()

        # Create new top-level window
        self.video_window = tk.Toplevel(self.parent)
        self.video_window.title("Fall Detection - Video Analysis")
        self.video_window.geometry("800x700")
        self.video_window.minsize(600, 500)

        # Make window resizable
        self.video_window.rowconfigure(1, weight=1)
        self.video_window.columnconfigure(0, weight=1)

        # Create header frame
        header_frame = tk.Frame(self.video_window, bg="#2c3e50", height=60)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        header_frame.columnconfigure(1, weight=1)

        # Video file name label
        video_name = video_path.split('/')[-1] if '/' in video_path else video_path.split('\\')[-1]
        file_label = tk.Label(header_frame, text=f"Video: {video_name}",
                              font=("Arial", 12, "bold"), fg="white", bg="#2c3e50")
        file_label.grid(row=0, column=0, padx=20, pady=15, sticky="w")

        # Danger score label in header
        self.danger_label = tk.Label(header_frame, text="Danger Score: 0.00",
                                     font=("Arial", 16, "bold"), fg="green", bg="#2c3e50")
        self.danger_label.grid(row=0, column=1, padx=20, pady=15, sticky="e")

        # Create main content frame
        content_frame = tk.Frame(self.video_window)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.rowconfigure(0, weight=1)
        content_frame.columnconfigure(0, weight=1)

        # Create canvas for video display with scrollbars
        canvas_frame = tk.Frame(content_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # Create canvas with border
        self.canvas = tk.Canvas(canvas_frame, bg='black', highlightthickness=2,
                                highlightbackground="#34495e")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Add scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Create control panel
        self.create_control_panel()

        # Bind window events
        self.video_window.protocol("WM_DELETE_WINDOW", self.on_window_close)

        # Bind canvas resize event
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Start video processing
        self.start_video_processing(video_path)

    def create_control_panel(self):
        """Create control panel with buttons."""
        self.control_frame = tk.Frame(self.video_window, bg="#ecf0f1", height=80)
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.control_frame.grid_propagate(False)

        # Center the buttons
        button_frame = tk.Frame(self.control_frame, bg="#ecf0f1")
        button_frame.pack(expand=True)

        # Stop/Start button
        self.stop_start_btn = tk.Button(button_frame, text="Stop Processing",
                                        command=self.toggle_processing,
                                        font=("Arial", 10, "bold"),
                                        bg="#e74c3c", fg="white",
                                        padx=20, pady=5, cursor="hand2")
        self.stop_start_btn.pack(side=tk.LEFT, padx=10, pady=20)

        # Save screenshot button
        screenshot_btn = tk.Button(button_frame, text="Save Screenshot",
                                   command=self.save_screenshot,
                                   font=("Arial", 10),
                                   bg="#3498db", fg="white",
                                   padx=20, pady=5, cursor="hand2")
        screenshot_btn.pack(side=tk.LEFT, padx=10, pady=20)

        # Close window button
        close_btn = tk.Button(button_frame, text="Close Window",
                              command=self.on_window_close,
                              font=("Arial", 10),
                              bg="#95a5a6", fg="white",
                              padx=20, pady=5, cursor="hand2")
        close_btn.pack(side=tk.LEFT, padx=10, pady=20)

    def on_canvas_configure(self, event):
        """Handle canvas resize event."""
        # Update scroll region when canvas is resized
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def toggle_processing(self):
        """Toggle video processing on/off."""
        if self.is_processing:
            # Stop processing
            stop_event.set()
            self.stop_start_btn.config(text="Start Processing", bg="#27ae60")
            self.is_processing = False
        else:
            # Start processing
            stop_event.clear()
            self.stop_start_btn.config(text="Stop Processing", bg="#e74c3c")
            self.is_processing = True

    def save_screenshot(self):
        """Save current frame as screenshot."""
        try:
            if hasattr(self.canvas, 'current_image'):
                filename = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
                )
                if filename:
                    # Save the current PIL image
                    self.canvas.current_image.save(filename)
                    messagebox.showinfo("Success", f"Screenshot saved to {filename}")
            else:
                messagebox.showwarning("Warning", "No frame available to save")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save screenshot: {str(e)}")

    def start_video_processing(self, video_path):
        """Start video processing in a separate thread."""
        stop_event.clear()
        self.is_processing = True

        # Start video processing thread
        threading.Thread(
            target=self.process_video_callback,
            args=(video_path, self.danger_label, self.canvas, self.video_window, stop_event),
            daemon=True
        ).start()

    def on_window_close(self):
        """Handle window close event."""
        # Stop video processing
        stop_event.set()
        self.is_processing = False

        # Destroy window
        if self.video_window and self.video_window.winfo_exists():
            self.video_window.destroy()

        # Clear stop event for next use
        stop_event.clear()


def update_main_frame_for_fall_detection_video(main_frame, process_video, root):
    """Update the main frame to show video selection interface."""
    # Clear the main frame but keep it for other functions
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Create video selection interface
    create_video_selection_interface(main_frame, process_video, root)


def create_video_selection_interface(main_frame, process_video, root):
    """Create interface for video file selection."""
    # Title
    title_label = tk.Label(main_frame, text="Fall Detection - Video Analysis",
                           font=("Arial", 18, "bold"), fg="#2c3e50")
    title_label.pack(pady=30)

    # Description
    desc_label = tk.Label(main_frame,
                          text="Select a video file to analyze for fall detection.\nA new window will open for video processing.",
                          font=("Arial", 12), fg="#7f8c8d", justify=tk.CENTER)
    desc_label.pack(pady=10)

    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=40)

    # Select video button
    def select_and_open_video():
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )

        if video_path:
            # Create video processing window
            video_window = FallDetectionVideoWindow(root, process_video)
            video_window.create_video_window(video_path)

    select_btn = tk.Button(button_frame, text="📁 Select Video File",
                           command=select_and_open_video,
                           font=("Arial", 14, "bold"),
                           bg="#3498db", fg="white",
                           padx=30, pady=15, cursor="hand2",
                           relief=tk.RAISED, bd=2)
    select_btn.pack(pady=10)

    # Recent files section (placeholder for future enhancement)
    recent_frame = tk.Frame(main_frame)
    recent_frame.pack(pady=30, fill=tk.X, padx=50)

    recent_label = tk.Label(recent_frame, text="Quick Actions:",
                            font=("Arial", 12, "bold"), fg="#2c3e50")
    recent_label.pack(anchor=tk.W)

    # Sample video button (if exists)
    def open_sample_video():
        # You can add a sample video path here
        sample_path = "sample_video.mp4"  # Replace with actual sample video path
        if sample_path:
            video_window = FallDetectionVideoWindow(root, process_video)
            video_window.create_video_window(sample_path)
        else:
            messagebox.showinfo("Info", "No sample video available")

    sample_btn = tk.Button(recent_frame, text="🎬 Open Sample Video",
                           command=open_sample_video,
                           font=("Arial", 10),
                           bg="#95a5a6", fg="white",
                           padx=20, pady=8, cursor="hand2")
    sample_btn.pack(anchor=tk.W, pady=5)

    # Instructions
    instructions_frame = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RIDGE, bd=1)
    instructions_frame.pack(pady=30, padx=50, fill=tk.X)

    instructions_title = tk.Label(instructions_frame, text="Instructions:",
                                  font=("Arial", 11, "bold"), bg="#ecf0f1", fg="#2c3e50")
    instructions_title.pack(anchor=tk.W, padx=15, pady=(10, 5))

    instructions = [
        "1. Click 'Select Video File' to choose a video for analysis",
        "2. A new window will open showing the video with fall detection",
        "3. The danger score will be displayed in real-time",
        "4. Use controls to stop/start processing or save screenshots",
        "5. Close the window when finished"
    ]

    for instruction in instructions:
        inst_label = tk.Label(instructions_frame, text=instruction,
                              font=("Arial", 9), bg="#ecf0f1", fg="#34495e")
        inst_label.pack(anchor=tk.W, padx=25, pady=2)

    tk.Label(instructions_frame, text="", bg="#ecf0f1").pack(pady=5)  # Spacer