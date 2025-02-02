import tkinter as tk
from tkinter import filedialog


def update_main_frame_for_generate_labels(main_frame, generate_labels_callback):
    """Update the main frame content for Generate Labels functionality."""
    # Clear the existing frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Frame for selecting image root directory
    image_frame = tk.Frame(main_frame)
    image_frame.pack(pady=10)

    selected_image_dir = tk.StringVar()

    def select_image_dir():
        image_dir = filedialog.askdirectory(title="Select Image Directory")
        if image_dir:
            selected_image_dir.set(image_dir)

    image_label = tk.Label(image_frame, text="Select Image Root Directory:")
    image_label.pack(side=tk.LEFT, padx=10)

    image_button = tk.Button(image_frame, text="Browse", command=select_image_dir)
    image_button.pack(side=tk.LEFT, padx=10)

    image_display = tk.Label(main_frame, textvariable=selected_image_dir, anchor="w")
    image_display.pack(pady=5)

    # Frame for selecting YOLO model directory
    model_frame = tk.Frame(main_frame)
    model_frame.pack(pady=10)

    selected_model_path = tk.StringVar()

    def select_model_file():
        model_path = filedialog.askopenfilename(filetypes=[("YOLO model file", "*.pt")])
        if model_path:
            selected_model_path.set(model_path)

    model_label = tk.Label(model_frame, text="Select YOLO Model File:")
    model_label.pack(side=tk.LEFT, padx=10)

    model_button = tk.Button(model_frame, text="Browse", command=select_model_file)
    model_button.pack(side=tk.LEFT, padx=10)

    model_display = tk.Label(main_frame, textvariable=selected_model_path, anchor="w")
    model_display.pack(pady=5)

    # Confirmation button
    def confirm():
        generate_labels_callback(selected_image_dir.get(), selected_model_path.get())

    confirm_button = tk.Button(main_frame, text="Confirm", command=confirm)
    confirm_button.pack(pady=20)
