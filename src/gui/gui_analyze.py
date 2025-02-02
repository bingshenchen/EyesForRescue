import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
from src.utils.generate_report import generate_batch_report


class AnalyzeReportGUI:
    def __init__(self, main_frame):
        self.main_frame = main_frame

        # Clear previous content without destroying the main_frame
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        self.frame = tk.Frame(self.main_frame)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Model paths and video folder variables
        self.new_model_path = tk.StringVar()
        self.video_folder_path = tk.StringVar()
        self.output_folder_path = tk.StringVar()

        # Report Data (to save later)
        self.report_data = None

        # Treeview for displaying results
        self.tree = ttk.Treeview(self.frame, columns=("Video", "Truth", "Found", "Correct",
                                                      "False Positive", "Missed"))
        self.tree.heading("#0", text="Index")
        self.tree.heading("#1", text="Video")
        self.tree.heading("#2", text="Truth")
        self.tree.heading("#3", text="Found")
        self.tree.heading("#4", text="Correct")
        self.tree.heading("#5", text="False")
        self.tree.heading("#6", text="Missed")
        self.tree.column("#0", stretch=tk.NO, width=50)
        self.tree.column("#1", anchor="w")
        self.tree.column("#2", anchor="center")
        self.tree.column("#3", anchor="center")
        self.tree.column("#4", anchor="center")
        self.tree.column("#5", anchor="center")
        self.tree.column("#6", anchor="center")

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Model Selection
        tk.Label(self.frame, text="Select YOLO Model").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.frame, textvariable=self.new_model_path, width=40).grid(row=0, column=1, padx=5, pady=5,
                                                                              sticky="w")
        tk.Button(self.frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5,
                                                                             sticky="w")

        # Video folder selection
        tk.Label(self.frame, text="Select Video Folder(Contains videos, sliced clips, \nand frames with ground truth labels)").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.frame, textvariable=self.video_folder_path, width=40).grid(row=1, column=1, padx=5, pady=5,
                                                                                 sticky="w")
        tk.Button(self.frame, text="Browse", command=self.browse_video_folder).grid(row=1, column=2, padx=5, pady=5,
                                                                                    sticky="w")

        # Output folder selection
        tk.Label(self.frame, text="Select Output Folder").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        tk.Entry(self.frame, textvariable=self.output_folder_path, width=40).grid(row=2, column=1, padx=5, pady=5,
                                                                                  sticky="w")
        tk.Button(self.frame, text="Browse", command=self.browse_output_folder).grid(row=2, column=2, padx=5, pady=5,
                                                                                     sticky="w")

        # Analyze button
        tk.Button(self.frame, text="Analyze", command=self.analyze_videos).grid(row=3, column=1, padx=5, pady=10,
                                                                                sticky="w")

        # Treeview placement
        self.tree.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        # Adjust column widths dynamically
        self.adjust_treeview_columns()

        # Save button
        tk.Button(self.frame, text="Save Report", command=self.save_report).grid(row=5, column=1, padx=5, pady=10,
                                                                                 sticky="w")

        # Configure grid weights for resizing
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

    def browse_model(self):
        file_path = filedialog.askopenfilename(title="Select YOLO Model",
                                               filetypes=[("Model Files", "*.pt"), ("All Files", "*.*")])
        if file_path:
            self.new_model_path.set(file_path)

    def browse_video_folder(self):
        folder_path = filedialog.askdirectory(title="Select Video Folder (Contains videos, sliced clips, and frames with ground truth labels)")
        if folder_path:
            self.video_folder_path.set(folder_path)

    def browse_output_folder(self):
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_folder_path.set(folder_path)

    def analyze_videos(self):
        # Validate input
        if not self.new_model_path.get() or not self.video_folder_path.get():
            messagebox.showerror("Error", "Please select both model and the video folder.")
            return

        if not self.output_folder_path.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return

        # Generate report and display in the tree
        try:
            report = generate_batch_report(self.video_folder_path.get(),
                                           self.new_model_path.get(),
                                           ["person", "falling_person", "sitting_person", "lying_person"],
                                           frame_skip=5)
            if not report:
                messagebox.showwarning("Warning", "No data found in the report.")
                return

            self.report_data = report
            self.display_report(report)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while generating the report: {e}")

    def display_report(self, report_data):
        """Display the report in the Treeview."""
        # Clear existing rows
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Insert new rows
        for idx, data in enumerate(report_data):
            self.tree.insert("", tk.END, text=str(idx),
                             values=(data['videoname'],
                                     data['truth'],
                                     data['found'],
                                     data['correct'],
                                     data['false_positive'],
                                     data['missed']))

    def save_report(self):
        """Save the report to an Excel file."""
        if not self.report_data:
            messagebox.showerror("Error", "No report data to save.")
            return

        # Prompt user to choose the save location
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx"), ("All Files", "*.*")])
        if file_path:
            try:
                # Convert report data to a DataFrame and save to the chosen file path
                df = pd.DataFrame(self.report_data)
                df.to_excel(file_path, index=False)
                messagebox.showinfo("Success", f"Report saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while saving the report: {e}")

    def adjust_treeview_columns(self):
        """Dynamically adjust treeview column widths based on the frame size."""
        self.frame.update_idletasks()  # Ensure frame dimensions are updated
        total_width = self.frame.winfo_width()
        column_width = total_width // len(self.tree['columns'])

        for col in self.tree['columns']:
            self.tree.column(col, width=column_width, stretch=tk.NO)