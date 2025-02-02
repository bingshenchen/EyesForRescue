import tkinter as tk
from tkinter import filedialog


def update_main_frame_for_extract_frames(main_frame, extract_frames_callback):
    """Update the main frame content for Extract Frames functionality."""
    # Clear the existing frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    # Frame for selecting video files
    video_frame = tk.Frame(main_frame)
    video_frame.pack(pady=10)

    selected_files = tk.StringVar()

    def select_videos():
        video_paths = filedialog.askopenfilenames(filetypes=[("Video files", "*.mp4;*.avi")])
        if video_paths:
            selected_files.set("\n".join(video_paths))

    video_label = tk.Label(video_frame, text="Select Videos:")
    video_label.pack(side=tk.LEFT, padx=10)

    video_button = tk.Button(video_frame, text="Browse", command=select_videos)
    video_button.pack(side=tk.LEFT, padx=10)

    video_display = tk.Label(main_frame, textvariable=selected_files, anchor="w")
    video_display.pack(pady=5)

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

    # Frame for selecting frame interval
    interval_frame = tk.Frame(main_frame)
    interval_frame.pack(pady=10)

    interval_label = tk.Label(interval_frame, text="Frame Interval:")
    interval_label.pack(side=tk.LEFT, padx=10)

    interval_entry = tk.Entry(interval_frame)
    interval_entry.insert(0, "5")
    interval_entry.pack(side=tk.LEFT, padx=10)

    # Confirmation button
    def confirm():
        extract_frames_callback(selected_files.get().split("\n"), selected_output.get(), int(interval_entry.get()))

    confirm_button = tk.Button(main_frame, text="Confirm", command=confirm)
    confirm_button.pack(pady=20)
