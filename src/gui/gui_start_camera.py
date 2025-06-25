# src/gui/gui_start_camera.py

import tkinter as tk
from tkinter import messagebox, ttk
import threading
from PIL import Image, ImageTk
import cv2

# Global stop event for controlling the camera processing
stop_event = threading.Event()


class FallDetectionCameraWindow:
    """Separate window for fall detection camera processing."""

    def __init__(self, parent, process_video_callback):
        self.parent = parent
        self.process_video_callback = process_video_callback
        self.camera_window = None
        self.canvas = None
        self.danger_label = None
        self.control_frame = None
        self.is_processing = False
        self.camera_index = 0

    def create_camera_window(self):
        """Create a new window for camera processing."""
        if self.camera_window and self.camera_window.winfo_exists():
            self.camera_window.destroy()

        # Create new top-level window
        self.camera_window = tk.Toplevel(self.parent)
        self.camera_window.title("Fall Detection - Live Camera Feed")
        self.camera_window.geometry("900x750")
        self.camera_window.minsize(700, 600)

        # Make window resizable
        self.camera_window.rowconfigure(1, weight=1)
        self.camera_window.columnconfigure(0, weight=1)

        # Create header frame
        header_frame = tk.Frame(self.camera_window, bg="#27ae60", height=70)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)
        header_frame.columnconfigure(1, weight=1)

        # Camera status label
        status_label = tk.Label(header_frame, text="🔴 LIVE CAMERA FEED",
                                font=("Arial", 14, "bold"), fg="white", bg="#27ae60")
        status_label.grid(row=0, column=0, padx=20, pady=20, sticky="w")

        # Danger score label in header
        self.danger_label = tk.Label(header_frame, text="Danger Score: 0.00",
                                     font=("Arial", 18, "bold"), fg="white", bg="#27ae60")
        self.danger_label.grid(row=0, column=1, padx=20, pady=20, sticky="e")

        # Create main content frame
        content_frame = tk.Frame(self.camera_window)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content_frame.rowconfigure(0, weight=1)
        content_frame.columnconfigure(0, weight=1)

        # Create canvas for camera display
        canvas_frame = tk.Frame(content_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        # Create canvas with border
        self.canvas = tk.Canvas(canvas_frame, bg='black', highlightthickness=2,
                                highlightbackground="#27ae60")
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
        self.camera_window.protocol("WM_DELETE_WINDOW", self.on_window_close)

        # Bind canvas resize event
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Test camera availability before starting
        if self.test_camera():
            self.start_camera_processing()
        else:
            messagebox.showerror("Camera Error", "No camera found or camera is already in use.")
            self.camera_window.destroy()

    def test_camera(self):
        """Test if camera is available."""
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret
            return False
        except Exception as e:
            print(f"Camera test failed: {e}")
            return False

    def create_control_panel(self):
        """Create control panel with buttons."""
        self.control_frame = tk.Frame(self.camera_window, bg="#ecf0f1", height=90)
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.control_frame.grid_propagate(False)

        # Create two rows of controls
        top_controls = tk.Frame(self.control_frame, bg="#ecf0f1")
        top_controls.pack(pady=10)

        bottom_controls = tk.Frame(self.control_frame, bg="#ecf0f1")
        bottom_controls.pack(pady=5)

        # Top row - main controls
        self.stop_start_btn = tk.Button(top_controls, text="Stop Camera",
                                        command=self.toggle_processing,
                                        font=("Arial", 11, "bold"),
                                        bg="#e74c3c", fg="white",
                                        padx=25, pady=8, cursor="hand2")
        self.stop_start_btn.pack(side=tk.LEFT, padx=10)

        # Camera selection
        camera_frame = tk.Frame(top_controls, bg="#ecf0f1")
        camera_frame.pack(side=tk.LEFT, padx=15)

        tk.Label(camera_frame, text="Camera:", font=("Arial", 9), bg="#ecf0f1").pack(side=tk.LEFT)
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var,
                                    values=["0", "1", "2"], width=5, state="readonly")
        camera_combo.pack(side=tk.LEFT, padx=5)
        camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)

        # Save screenshot button
        screenshot_btn = tk.Button(top_controls, text="📷 Screenshot",
                                   command=self.save_screenshot,
                                   font=("Arial", 10),
                                   bg="#3498db", fg="white",
                                   padx=20, pady=8, cursor="hand2")
        screenshot_btn.pack(side=tk.LEFT, padx=10)

        # Close window button
        close_btn = tk.Button(top_controls, text="✖ Close",
                              command=self.on_window_close,
                              font=("Arial", 10),
                              bg="#95a5a6", fg="white",
                              padx=20, pady=8, cursor="hand2")
        close_btn.pack(side=tk.LEFT, padx=10)

        # Bottom row - status and info
        self.status_label = tk.Label(bottom_controls, text="Camera Status: Starting...",
                                     font=("Arial", 9), fg="#7f8c8d", bg="#ecf0f1")
        self.status_label.pack()

    def on_camera_change(self, event):
        """Handle camera selection change."""
        new_index = int(self.camera_var.get())
        if new_index != self.camera_index:
            self.camera_index = new_index
            if self.is_processing:
                # Restart with new camera
                self.toggle_processing()  # Stop current
                self.camera_window.after(500, self.toggle_processing)  # Start with new camera

    def on_canvas_configure(self, event):
        """Handle canvas resize event."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def toggle_processing(self):
        """Toggle camera processing on/off."""
        if self.is_processing:
            # Stop processing
            stop_event.set()
            self.stop_start_btn.config(text="Start Camera", bg="#27ae60")
            self.status_label.config(text="Camera Status: Stopped")
            self.is_processing = False
        else:
            # Test camera before starting
            if self.test_camera():
                # Start processing
                stop_event.clear()
                self.stop_start_btn.config(text="Stop Camera", bg="#e74c3c")
                self.status_label.config(text="Camera Status: Running")
                self.is_processing = True
                self.start_camera_processing()
            else:
                messagebox.showerror("Camera Error",
                                     f"Cannot access camera {self.camera_index}. Please check if camera is available.")

    def save_screenshot(self):
        """Save current frame as screenshot."""
        try:
            if hasattr(self.canvas, 'current_image'):
                from tkinter import filedialog
                filename = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                    initialfilename=f"camera_screenshot_{self.camera_index}.png"
                )
                if filename:
                    self.canvas.current_image.save(filename)
                    messagebox.showinfo("Success", f"Screenshot saved to {filename}")
            else:
                messagebox.showwarning("Warning", "No frame available to save")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save screenshot: {str(e)}")

    def start_camera_processing(self):
        """Start camera processing in a separate thread."""
        stop_event.clear()
        self.is_processing = True

        # Start camera processing thread
        threading.Thread(
            target=self.process_video_callback,
            args=(self.camera_index, self.danger_label, self.canvas, self.camera_window, stop_event),
            daemon=True
        ).start()

    def on_window_close(self):
        """Handle window close event."""
        # Stop camera processing
        stop_event.set()
        self.is_processing = False

        # Small delay to ensure camera is released
        self.camera_window.after(200, self._destroy_window)

    def _destroy_window(self):
        """Destroy window after ensuring camera is released."""
        if self.camera_window and self.camera_window.winfo_exists():
            self.camera_window.destroy()
        stop_event.clear()


def update_main_frame_for_camera_analysis(main_frame, root):
    """Update the main frame to show camera selection interface."""
    # Clear the main frame but keep it for other functions
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Import the process_video function
    from src.core.utils.video_processor import process_video

    # Create camera selection interface
    create_camera_selection_interface(main_frame, process_video, root)


def create_camera_selection_interface(main_frame, process_video, root):
    """Create interface for camera selection and setup."""
    # Title
    title_label = tk.Label(main_frame, text="Fall Detection - Live Camera Analysis",
                           font=("Arial", 18, "bold"), fg="#27ae60")
    title_label.pack(pady=30)

    # Description
    desc_label = tk.Label(main_frame,
                          text="Start live camera feed for real-time fall detection.\nA new window will open for camera monitoring.",
                          font=("Arial", 12), fg="#7f8c8d", justify=tk.CENTER)
    desc_label.pack(pady=10)

    # Camera setup frame
    setup_frame = tk.LabelFrame(main_frame, text="Camera Setup",
                                font=("Arial", 12, "bold"), fg="#2c3e50",
                                padx=20, pady=20)
    setup_frame.pack(pady=30, padx=50, fill=tk.X)

    # Camera selection
    camera_frame = tk.Frame(setup_frame)
    camera_frame.pack(pady=10)

    tk.Label(camera_frame, text="Select Camera:", font=("Arial", 11)).pack(side=tk.LEFT, padx=10)
    camera_var = tk.StringVar(value="0")
    camera_combo = ttk.Combobox(camera_frame, textvariable=camera_var,
                                values=["0 (Default)", "1 (External)", "2 (USB)"],
                                width=20, state="readonly")
    camera_combo.pack(side=tk.LEFT, padx=10)

    # Test camera button
    def test_camera():
        """Test selected camera."""
        try:
            camera_index = int(camera_var.get().split()[0])
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    messagebox.showinfo("Camera Test", f"Camera {camera_index} is working properly!")
                else:
                    messagebox.showwarning("Camera Test", f"Camera {camera_index} found but not working properly.")
            else:
                messagebox.showerror("Camera Test", f"Cannot access camera {camera_index}.")
        except Exception as e:
            messagebox.showerror("Camera Test", f"Error testing camera: {str(e)}")

    test_btn = tk.Button(camera_frame, text="Test Camera", command=test_camera,
                         font=("Arial", 9), bg="#f39c12", fg="white",
                         padx=15, pady=5, cursor="hand2")
    test_btn.pack(side=tk.LEFT, padx=10)

    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=40)

    # Start camera button
    def start_camera_analysis():
        """Start camera analysis in new window."""
        try:
            camera_index = int(camera_var.get().split()[0])

            # Test camera first
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                messagebox.showerror("Camera Error",
                                     f"Cannot access camera {camera_index}. Please check camera connection.")
                return
            cap.release()

            # Create camera processing window
            camera_window = FallDetectionCameraWindow(root, process_video)
            camera_window.camera_index = camera_index
            camera_window.create_camera_window()

        except ValueError:
            messagebox.showerror("Error", "Invalid camera selection.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")

    start_btn = tk.Button(button_frame, text="📹 Start Live Camera",
                          command=start_camera_analysis,
                          font=("Arial", 14, "bold"),
                          bg="#27ae60", fg="white",
                          padx=30, pady=15, cursor="hand2",
                          relief=tk.RAISED, bd=2)
    start_btn.pack(pady=10)

    # Status and requirements frame
    status_frame = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RIDGE, bd=1)
    status_frame.pack(pady=30, padx=50, fill=tk.X)

    status_title = tk.Label(status_frame, text="Requirements & Tips:",
                            font=("Arial", 11, "bold"), bg="#ecf0f1", fg="#2c3e50")
    status_title.pack(anchor=tk.W, padx=15, pady=(10, 5))

    requirements = [
        "• Ensure camera is connected and not used by other applications",
        "• Good lighting conditions improve detection accuracy",
        "• Position camera to have clear view of the monitored area",
        "• Test camera before starting live analysis",
        "• Close the camera window when finished to free resources"
    ]

    for req in requirements:
        req_label = tk.Label(status_frame, text=req,
                             font=("Arial", 9), bg="#ecf0f1", fg="#34495e")
        req_label.pack(anchor=tk.W, padx=25, pady=2)

    tk.Label(status_frame, text="", bg="#ecf0f1").pack(pady=5)  # Spacer