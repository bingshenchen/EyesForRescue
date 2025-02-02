import tkinter as tk


def setup_menubar(root, extract_frames, generate_labels, train_model, open_video_file, start_camera, analyze_report, alert_test):
    """Set up the menu bar with options for different functionalities."""
    menubar = tk.Menu(root)

    # Detection  Menu
    app_menu = tk.Menu(menubar, tearoff=0)
    app_menu.add_command(label="Fall Detection (Video)", command=open_video_file)
    app_menu.add_command(label="Fall Detection (Camera)", command=start_camera)
    menubar.add_cascade(label="Detection ", menu=app_menu)

    # Data Preparation Menu
    edit_menu = tk.Menu(menubar, tearoff=0)
    edit_menu.add_command(label="Extract Frames", command=extract_frames)
    edit_menu.add_command(label="Auto Generate Labels", command=generate_labels)
    menubar.add_cascade(label="Data Preparation", menu=edit_menu)

    # Training Menu
    training_menu = tk.Menu(menubar, tearoff=0)
    training_menu.add_command(label="Model Training", command=train_model)
    menubar.add_cascade(label="Training", menu=training_menu)

    # Analysis Menu
    analysis_menu = tk.Menu(menubar, tearoff=0)
    analysis_menu.add_command(label="Model Performance Report", command=analyze_report)
    analysis_menu.add_command(label="Alert Test", command=alert_test)
    menubar.add_cascade(label="Analysis", menu=analysis_menu)

    root.config(menu=menubar)
