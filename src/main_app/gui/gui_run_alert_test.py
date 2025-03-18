import tkinter as tk
from tkinter import filedialog


def update_main_frame_for_alert(main_frame, run_alert_test_callback):
    """Update the main frame content for Train Model functionality."""
    # Clear the existing frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    model_frame = tk.Frame(main_frame)
    model_frame.pack(pady=10)

    image_dir = tk.StringVar()

    def select_image_dir():
        image_directory = filedialog.askdirectory()
        if image_directory:
            image_dir.set(image_directory)

    image_dir_label = tk.Label(model_frame, text="Select image directory")
    image_dir_label.pack(side=tk.LEFT, padx=10)

    image_dir_button = tk.Button(model_frame, text="Browse", command=select_image_dir)
    image_dir_button.pack(side=tk.LEFT, padx=10)

    image_dir_display = tk.Label(main_frame, textvariable=image_dir, anchor="w")
    image_dir_display.pack(pady=5)


    params_frame = tk.Frame(main_frame)
    params_frame.pack(pady=10)

    frameskip_label = tk.Label(params_frame, text="Frameskip")
    frameskip_label.pack(side=tk.LEFT, padx=10)
    frameskip_entry = tk.Entry(params_frame)
    frameskip_entry.insert(0, "23")
    frameskip_entry.pack(side=tk.LEFT, padx=10)

    fall_threshold_label = tk.Label(params_frame, text="Seconds to trigger alert")
    fall_threshold_label.pack(side=tk.LEFT, padx=10)
    fall_threshold_entry = tk.Entry(params_frame)
    fall_threshold_entry.insert(0, "1")
    fall_threshold_entry.pack(side=tk.LEFT, padx=10)

    # Confirmation button
    def confirm():
        image_directory = image_dir.get()
        frameskip = int(frameskip_entry.get())
        fall_threshold = int(fall_threshold_entry.get())

        run_alert_test_callback(image_directory, frameskip, fall_threshold)

    confirm_button = tk.Button(main_frame, text="Confirm", command=confirm)
    confirm_button.pack(pady=20)
