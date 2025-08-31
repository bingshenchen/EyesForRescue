# src/gui/gui_start_camera.py - Improved version

import tkinter as tk
from tkinter import messagebox, ttk
import threading
from PIL import Image, ImageTk
import cv2
import platform
import time

# Global stop event for controlling the camera processing
stop_event = threading.Event()


class ImprovedCameraDetector:
    """Enhanced camera detection with better error handling."""

    @staticmethod
    def find_available_cameras(max_cameras=10):
        """
        Find all available cameras with detailed information.

        Returns:
            list: List of dictionaries with camera information
        """
        available_cameras = []

        for i in range(max_cameras):
            try:
                # Try different backends based on platform
                if platform.system() == "Windows":
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(i)

                if cap.isOpened():
                    # Set buffer size to avoid lag
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        available_cameras.append({
                            'index': i,
                            'name': f"Camera {i}",
                            'resolution': f"{width}x{height}",
                            'fps': fps,
                            'working': True
                        })

                cap.release()

            except Exception as e:
                print(f"Error testing camera {i}: {str(e)}")

        return available_cameras

    @staticmethod
    def test_camera_advanced(camera_index, timeout=3):
        """
        Advanced camera test with multiple backends and timeout.

        Args:
            camera_index (int): Camera index to test
            timeout (int): Timeout in seconds

        Returns:
            dict: Test results
        """
        backends_to_try = []

        # Choose backends based on platform
        if platform.system() == "Windows":
            backends_to_try = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Any")
            ]
        else:
            backends_to_try = [
                (cv2.CAP_V4L2, "Video4Linux2"),
                (cv2.CAP_ANY, "Any")
            ]

        for backend_id, backend_name in backends_to_try:
            try:
                cap = cv2.VideoCapture(camera_index, backend_id)

                if cap.isOpened():
                    # Set properties for better performance
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Try to read frame with timeout
                    start_time = time.time()
                    frame_captured = False

                    while time.time() - start_time < timeout:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frame_captured = True
                            break
                        time.sleep(0.1)

                    if frame_captured:
                        result = {
                            'success': True,
                            'backend': backend_name,
                            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            'fps': cap.get(cv2.CAP_PROP_FPS)
                        }
                        cap.release()
                        return result

                cap.release()

            except Exception as e:
                continue

        return {
            'success': False,
            'error': 'Camera not accessible with any backend'
        }


class FallDetectionCameraWindow:
    """Separate window for fall detection camera processing with improved camera handling."""

    def __init__(self, parent, process_video_callback):
        self.parent = parent
        self.process_video_callback = process_video_callback
        self.camera_window = None
        self.canvas = None
        self.danger_label = None
        self.control_frame = None
        self.is_processing = False
        self.camera_index = 0
        self.camera_detector = ImprovedCameraDetector()

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
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Test camera availability before starting
        if self.test_camera_improved():
            self.start_camera_processing()
        else:
            messagebox.showerror("Camera Error",
                                 f"Camera {self.camera_index} is not accessible.\n"
                                 f"Please check:\n"
                                 f"• Camera is connected properly\n"
                                 f"• Camera is not being used by another application\n"
                                 f"• Camera drivers are installed\n"
                                 f"• Try a different camera index")
            self.camera_window.destroy()

    def test_camera_improved(self):
        """Enhanced camera test with detailed feedback."""
        result = self.camera_detector.test_camera_advanced(self.camera_index)

        if result['success']:
            print(f"Camera {self.camera_index} test successful:")
            print(f"  Backend: {result['backend']}")
            print(f"  Resolution: {result['width']}x{result['height']}")
            print(f"  FPS: {result['fps']}")
            return True
        else:
            print(f"Camera {self.camera_index} test failed: {result.get('error', 'Unknown error')}")
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

        # Camera selection with available cameras
        camera_frame = tk.Frame(top_controls, bg="#ecf0f1")
        camera_frame.pack(side=tk.LEFT, padx=15)

        tk.Label(camera_frame, text="Camera:", font=("Arial", 9), bg="#ecf0f1").pack(side=tk.LEFT)

        # Get available cameras
        available_cameras = self.camera_detector.find_available_cameras()
        camera_options = [f"{cam['index']} ({cam['resolution']})" for cam in available_cameras]

        if not camera_options:
            camera_options = ["0 (Default)", "1 (External)", "2 (USB)"]

        self.camera_var = tk.StringVar(value=camera_options[0] if camera_options else "0")
        camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var,
                                    values=camera_options, width=15, state="readonly")
        camera_combo.pack(side=tk.LEFT, padx=5)
        camera_combo.bind("<<ComboboxSelected>>", self.on_camera_change)

        # Save screenshot button
        screenshot_btn = tk.Button(top_controls, text="📷 Screenshot",
                                   command=self.save_screenshot,
                                   font=("Arial", 10),
                                   bg="#3498db", fg="white",
                                   padx=20, pady=8, cursor="hand2")
        screenshot_btn.pack(side=tk.LEFT, padx=10)

        # Refresh cameras button
        refresh_btn = tk.Button(top_controls, text="🔄 Refresh",
                                command=self.refresh_cameras,
                                font=("Arial", 10),
                                bg="#f39c12", fg="white",
                                padx=20, pady=8, cursor="hand2")
        refresh_btn.pack(side=tk.LEFT, padx=10)

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

    def refresh_cameras(self):
        """Refresh the camera list."""
        try:
            available_cameras = self.camera_detector.find_available_cameras()
            camera_options = [f"{cam['index']} ({cam['resolution']})" for cam in available_cameras]

            if not camera_options:
                camera_options = ["No cameras found"]
                messagebox.showwarning("No Cameras", "No working cameras found on the system.")

            # Update the combobox
            for widget in self.control_frame.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Frame):
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, ttk.Combobox):
                                    grandchild['values'] = camera_options
                                    if camera_options[0] != "No cameras found":
                                        grandchild.set(camera_options[0])
                                    break

            self.status_label.config(text=f"Found {len(available_cameras)} working cameras")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh cameras: {str(e)}")

    def on_camera_change(self, event):
        """Handle camera selection change."""
        try:
            new_index = int(self.camera_var.get().split()[0])
            if new_index != self.camera_index:
                self.camera_index = new_index
                if self.is_processing:
                    # Restart with new camera
                    self.toggle_processing()  # Stop current
                    self.camera_window.after(500, self.toggle_processing)  # Start with new camera
        except ValueError:
            pass  # Ignore invalid selections

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
            if self.test_camera_improved():
                # Start processing
                stop_event.clear()
                self.stop_start_btn.config(text="Stop Camera", bg="#e74c3c")
                self.status_label.config(text="Camera Status: Running")
                self.is_processing = True
                self.start_camera_processing()
            else:
                messagebox.showerror("Camera Error",
                                     f"Cannot access camera {self.camera_index}.\n"
                                     f"Please try:\n"
                                     f"• Different camera index\n"
                                     f"• Check camera connections\n"
                                     f"• Close other camera applications\n"
                                     f"• Refresh camera list")

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
    """Create interface for camera selection and setup with improved detection."""
    # Title
    title_label = tk.Label(main_frame, text="Fall Detection - Live Camera Analysis",
                           font=("Arial", 18, "bold"), fg="#27ae60")
    title_label.pack(pady=30)

    # Description
    desc_label = tk.Label(main_frame,
                          text="Start live camera feed for real-time fall detection.\nImproved camera detection and error handling.",
                          font=("Arial", 12), fg="#7f8c8d", justify=tk.CENTER)
    desc_label.pack(pady=10)

    # Camera setup frame
    setup_frame = tk.LabelFrame(main_frame, text="Camera Setup",
                                font=("Arial", 12, "bold"), fg="#2c3e50",
                                padx=20, pady=20)
    setup_frame.pack(pady=30, padx=50, fill=tk.X)

    # Camera detection and selection
    camera_frame = tk.Frame(setup_frame)
    camera_frame.pack(pady=10)

    # Scan for cameras button
    def scan_cameras():
        """Scan for available cameras and update the list."""
        try:
            detector = ImprovedCameraDetector()
            available_cameras = detector.find_available_cameras()

            if available_cameras:
                camera_options = []
                for cam in available_cameras:
                    camera_options.append(f"{cam['index']} - {cam['resolution']} @ {cam['fps']:.1f}fps")

                camera_combo['values'] = camera_options
                camera_combo.set(camera_options[0])

                messagebox.showinfo("Cameras Found",
                                    f"Found {len(available_cameras)} working cameras:\n" +
                                    "\n".join([f"Camera {cam['index']}: {cam['resolution']}"
                                               for cam in available_cameras]))
            else:
                messagebox.showwarning("No Cameras", "No working cameras found on the system.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to scan cameras: {str(e)}")

    scan_btn = tk.Button(camera_frame, text="🔍 Scan Cameras", command=scan_cameras,
                         font=("Arial", 11, "bold"), bg="#3498db", fg="white",
                         padx=20, pady=8, cursor="hand2")
    scan_btn.pack(pady=10)

    # Camera selection
    selection_frame = tk.Frame(camera_frame)
    selection_frame.pack(pady=10)

    tk.Label(selection_frame, text="Select Camera:", font=("Arial", 11)).pack(side=tk.LEFT, padx=10)
    camera_var = tk.StringVar(value="0 - Default Camera")
    camera_combo = ttk.Combobox(selection_frame, textvariable=camera_var,
                                values=["0 - Default Camera", "1 - External Camera", "2 - USB Camera"],
                                width=30, state="readonly")
    camera_combo.pack(side=tk.LEFT, padx=10)

    # Advanced test camera button
    def test_camera_advanced():
        """Test selected camera with advanced diagnostics."""
        try:
            camera_index = int(camera_var.get().split()[0])
            detector = ImprovedCameraDetector()
            result = detector.test_camera_advanced(camera_index)

            if result['success']:
                messagebox.showinfo("Camera Test - Success",
                                    f"Camera {camera_index} is working!\n\n"
                                    f"Backend: {result['backend']}\n"
                                    f"Resolution: {result['width']}x{result['height']}\n"
                                    f"FPS: {result['fps']:.1f}")
            else:
                messagebox.showerror("Camera Test - Failed",
                                     f"Camera {camera_index} failed the test!\n\n"
                                     f"Error: {result.get('error', 'Unknown error')}\n\n"
                                     f"Troubleshooting:\n"
                                     f"• Check if camera is connected\n"
                                     f"• Close other camera applications\n"
                                     f"• Try a different camera index\n"
                                     f"• Check camera drivers")

        except ValueError:
            messagebox.showerror("Invalid Selection", "Please select a valid camera.")
        except Exception as e:
            messagebox.showerror("Test Error", f"Error testing camera: {str(e)}")

    test_btn = tk.Button(selection_frame, text="🧪 Test Camera", command=test_camera_advanced,
                         font=("Arial", 10), bg="#f39c12", fg="white",
                         padx=15, pady=5, cursor="hand2")
    test_btn.pack(side=tk.LEFT, padx=10)

    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=40)

    # Start camera button
    def start_camera_analysis():
        """Start camera analysis in new window with improved error handling."""
        try:
            camera_index = int(camera_var.get().split()[0])

            # Advanced camera test before starting
            detector = ImprovedCameraDetector()
            result = detector.test_camera_advanced(camera_index)

            if not result['success']:
                messagebox.showerror("Camera Error",
                                     f"Cannot start camera {camera_index}!\n\n"
                                     f"Error: {result.get('error', 'Unknown error')}\n\n"
                                     f"Please:\n"
                                     f"• Check camera connection\n"
                                     f"• Close other camera applications\n"
                                     f"• Try scanning for cameras first\n"
                                     f"• Select a different camera")
                return

            # Create camera processing window
            camera_window = FallDetectionCameraWindow(root, process_video)
            camera_window.camera_index = camera_index
            camera_window.create_camera_window()

        except ValueError:
            messagebox.showerror("Invalid Selection", "Please select a valid camera.")
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

    status_title = tk.Label(status_frame, text="Requirements & Troubleshooting:",
                            font=("Arial", 11, "bold"), bg="#ecf0f1", fg="#2c3e50")
    status_title.pack(anchor=tk.W, padx=15, pady=(10, 5))

    requirements = [
        "• Scan for cameras first to see available devices",
        "• USB cameras typically appear as Camera 1 or 2",
        "• Close other applications using the camera (Skype, Teams, etc.)",
        "• Check camera drivers and Windows privacy settings",
        "• Try different camera indices if one doesn't work",
        "• Good lighting improves detection accuracy",
        "• Test camera before starting live analysis"
    ]

    for req in requirements:
        req_label = tk.Label(status_frame, text=req,
                             font=("Arial", 9), bg="#ecf0f1", fg="#34495e")
        req_label.pack(anchor=tk.W, padx=25, pady=2)

    # Troubleshooting section
    troubleshoot_title = tk.Label(status_frame, text="Common Issues:",
                                  font=("Arial", 10, "bold"), bg="#ecf0f1", fg="#e74c3c")
    troubleshoot_title.pack(anchor=tk.W, padx=15, pady=(10, 5))

    issues = [
        "• 'Cannot access camera 2' → Try camera 0 or 1, or scan for cameras",
        "• Camera opens but no image → Check camera permissions and drivers",
        "• Application crashes → Close other camera apps and restart",
        "• Poor performance → Reduce resolution or close background apps"
    ]

    for issue in issues:
        issue_label = tk.Label(status_frame, text=issue,
                               font=("Arial", 9), bg="#ecf0f1", fg="#c0392b")
        issue_label.pack(anchor=tk.W, padx=25, pady=2)

    tk.Label(status_frame, text="", bg="#ecf0f1").pack(pady=5)  # Spacer

    # Initialize with camera scan
    scan_cameras()