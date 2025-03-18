import tkinter as tk
from tkinter import filedialog


def update_main_frame_for_train_model(main_frame, train_model_callback):
    """Update the main frame content for Train Model functionality."""
    # Clear the existing frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Frame for selecting YOLO model
    model_frame = tk.Frame(main_frame)
    model_frame.pack(pady=10)

    selected_model = tk.StringVar()

    def select_model():
        model_path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if model_path:
            selected_model.set(model_path)

    model_label = tk.Label(model_frame, text="Select YOLO Model:")
    model_label.pack(side=tk.LEFT, padx=10)

    model_button = tk.Button(model_frame, text="Browse", command=select_model)
    model_button.pack(side=tk.LEFT, padx=10)

    model_display = tk.Label(main_frame, textvariable=selected_model, anchor="w")
    model_display.pack(pady=5)

    # Frame for selecting dataset .yaml file
    data_frame = tk.Frame(main_frame)
    data_frame.pack(pady=10)

    selected_data = tk.StringVar()

    def select_data():
        data_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if data_path:
            selected_data.set(data_path)

    data_label = tk.Label(data_frame, text="Select Dataset (.yaml):")
    data_label.pack(side=tk.LEFT, padx=10)

    data_button = tk.Button(data_frame, text="Browse", command=select_data)
    data_button.pack(side=tk.LEFT, padx=10)

    data_display = tk.Label(main_frame, textvariable=selected_data, anchor="w")
    data_display.pack(pady=5)

    # Frame for selecting output directory
    output_frame = tk.Frame(main_frame)
    output_frame.pack(pady=10)

    selected_output = tk.StringVar()

    def select_output_dir():
        output_path = filedialog.askdirectory()
        if output_path:
            selected_output.set(output_path)

    output_label = tk.Label(output_frame, text="Select Output Directory:")
    output_label.pack(side=tk.LEFT, padx=10)

    output_button = tk.Button(output_frame, text="Browse", command=select_output_dir)
    output_button.pack(side=tk.LEFT, padx=10)

    output_display = tk.Label(main_frame, textvariable=selected_output, anchor="w")
    output_display.pack(pady=5)

    # Frame for setting training parameters
    params_frame = tk.Frame(main_frame)
    params_frame.pack(pady=10)

    epochs_label = tk.Label(params_frame, text="Epochs:")
    epochs_label.pack(side=tk.LEFT, padx=10)
    epochs_entry = tk.Entry(params_frame)
    epochs_entry.insert(0, "30")
    epochs_entry.pack(side=tk.LEFT, padx=10)

    imgsz_label = tk.Label(params_frame, text="Image Size (imgsz):")
    imgsz_label.pack(side=tk.LEFT, padx=10)
    imgsz_entry = tk.Entry(params_frame)
    imgsz_entry.insert(0, "640")
    imgsz_entry.pack(side=tk.LEFT, padx=10)

    batch_label = tk.Label(params_frame, text="Batch Size:")
    batch_label.pack(side=tk.LEFT, padx=10)
    batch_entry = tk.Entry(params_frame)
    batch_entry.insert(0, "8")
    batch_entry.pack(side=tk.LEFT, padx=10)

    # Confirmation button
    def confirm():
        model_path = selected_model.get()
        data_path = selected_data.get()
        output_dir = selected_output.get()
        epochs = int(epochs_entry.get())
        imgsz = int(imgsz_entry.get())
        batch = int(batch_entry.get())
        train_model_callback(model_path, data_path, output_dir, epochs, imgsz, batch)

    confirm_button = tk.Button(main_frame, text="Confirm", command=confirm)
    confirm_button.pack(pady=20)
