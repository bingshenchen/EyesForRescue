import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import os

import cv2
import joblib
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load environment variables
load_dotenv()

# Define paths
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT', '.'))
OUTPUT_DIR = PROJECT_ROOT / "assets" / "classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINE_FEATURES_PATH = OUTPUT_DIR / "fine_features.pkl"
FINE_LABELS_PATH = OUTPUT_DIR / "fine_labels.pkl"
NEEDHELP_FEATURES_PATH = OUTPUT_DIR / "needhelp_features.pkl"
NEEDHELP_LABELS_PATH = OUTPUT_DIR / "needhelp_labels.pkl"
CLASSIFIER_PATH = OUTPUT_DIR / "classifier.pkl"

GUI_MINIO_PATH = PROJECT_ROOT / "src" / "gui" / "gui_minio.py"

# Global paths for GUI
fine_path = None
needhelp_path = None

# Load the YOLO pose model
model = YOLO("yolo11n-pose.pt")  # Make sure to use the pose model

# Initialize empty dataframe
pose_df = pd.DataFrame(columns=["image_path", "label", "features"])


def extract_features_from_image(image_path, label):
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (224, 224))  # Resize image using OpenCV
    img = img.astype(np.uint8)  # Ensure it's in the proper format

    # Get pose features using YOLO model
    results = model(img)  # Perform pose detection

    # Assuming you want to extract the pose keypoints (this depends on the model output)
    keypoints = results[0].keypoints  # First element in results, and get the keypoints

    # Check if keypoints are present
    if keypoints is not None:
        # Extract keypoint coordinates (xy)
        keypoints_xy = keypoints.xy.numpy()  # Convert tensor to numpy array

        # Flatten keypoints (x, y) values
        features = keypoints_xy.flatten()  # Flatten the coordinates into a 1D array
    else:
        features = np.zeros(34)  # If no keypoints detected, use a zero vector (adjust size if necessary)

    return features


def extract_features_from_folder(folder, label):
    feature_vectors = []
    labels = []
    for image_path in folder.glob("*.*"):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            features = extract_features_from_image(image_path, label)
        if features is not None:
            feature_vectors.append(features)
            labels.append(label)
            pose_df.loc[len(pose_df)] = [image_path, label, features]
    return feature_vectors, labels


def build_and_train_model(fine_dir, needhelp_dir):
    """
    Extract features from images, train a Random Forest classifier, and evaluate it.
    """
    # Paths to saved features and labels
    fine_features_location = Path("fine_features.pkl")
    fine_labels_location = Path("fine_labels.pkl")
    needhelp_features_location = Path("needhelp_features.pkl")
    needhelp_labels_location = Path("needhelp_labels.pkl")
    classifier_location = Path("classifier.pkl")

    # Load features and labels if files exist
    if fine_labels_location.is_file() and needhelp_labels_location.is_file() and \
       needhelp_features_location.is_file() and classifier_location.is_file():
        print("Loading features and labels from file...")
        fine_features = joblib.load(fine_features_location)
        fine_labels = joblib.load(fine_labels_location)
        needhelp_features = joblib.load(needhelp_features_location)
        needhelp_labels = joblib.load(needhelp_labels_location)
    else:
        print("Extracting features and labels from images...")
        # Extract features for both classes
        fine_features, fine_labels = extract_features_from_folder(Path(fine_dir), "Fine")
        needhelp_features, needhelp_labels = extract_features_from_folder(Path(needhelp_dir), "Need Help")

        # Save features and labels for future use
        joblib.dump(fine_features, fine_features_location)
        joblib.dump(fine_labels, fine_labels_location)
        joblib.dump(needhelp_features, needhelp_features_location)
        joblib.dump(needhelp_labels, needhelp_labels_location)

    # Combine features and labels
    if not fine_features or not needhelp_features:
        raise ValueError("No valid features extracted. Please check your dataset and paths.")

    # Combine features
    max_length = 34  # Adjust based on your feature size
    fine_features_padded = [np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length] for f in fine_features]
    needhelp_features_padded = [np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length] for f in needhelp_features]

    X = np.concatenate([fine_features_padded, needhelp_features_padded], axis=0)
    y = np.array(fine_labels + needhelp_labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=78)

    # Train or load the classifier
    if classifier_location.is_file():
        print("Loading existing classifier...")
        clf = joblib.load(classifier_location)
    else:
        print("Training new classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, classifier_location)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Classification Report:\n", report)


def start_training():
    """Start training the model and display the results in the GUI."""
    if not fine_path or not needhelp_path:
        messagebox.showerror("Error", "Please select both Fine and Need Help paths.")
        return
    try:
        build_and_train_model(fine_path, needhelp_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def select_fine_path():
    """Allow the user to select the fine data path."""
    global fine_path
    path = filedialog.askdirectory(title="Select Fine Data Path")
    if path:
        fine_path = path
        fine_path_label.config(text=f"Fine Path: {fine_path}")
        messagebox.showinfo("Selected Path", f"Fine path set to: {fine_path}")


def select_needhelp_path():
    """Allow the user to select the need-help data path."""
    global needhelp_path
    path = filedialog.askdirectory(title="Select Need Help Data Path")
    if path:
        needhelp_path = path
        needhelp_path_label.config(text=f"Need Help Path: {needhelp_path}")
        messagebox.showinfo("Selected Path", f"Need Help path set to: {needhelp_path}")


def open_minio_gui(bucket_name, prefix, path_type):
    """
    Open the MinIO GUI for selecting files and folders.
    """

    def callback():
        global fine_path, needhelp_path
        try:
            result = subprocess.run(
                ["python", str(GUI_MINIO_PATH), bucket_name, prefix],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                messagebox.showerror("Error", f"Failed to open MinIO GUI: {result.stderr}")
                return

            selected_path = result.stdout.strip()
            if path_type == "fine":
                fine_path = selected_path
                fine_path_label.config(text=f"Fine Path: {fine_path}")
                messagebox.showinfo("Selected Path", f"Fine path set to: {fine_path}")
            elif path_type == "needhelp":
                needhelp_path = selected_path
                needhelp_path_label.config(text=f"Need Help Path: {needhelp_path}")
                messagebox.showinfo("Selected Path", f"Need Help path set to: {needhelp_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            print(f"Error: {e}")

    return callback


# GUI setup
root = tk.Tk()
root.title("Random Forest Trainer")
root.geometry("600x400")

# Fine path selection
tk.Button(root, text="Select Fine Data Path", command=select_fine_path).pack(pady=5)
tk.Button(root, text="Use MinIO", command=open_minio_gui("eyes4rescue-group-13", "fine_needhelp/test/", "fine")).pack(
    pady=5)
fine_path_label = tk.Label(root, text="Fine Path: Not Selected", font=("Helvetica", 10), wraplength=500)
fine_path_label.pack(pady=5)

# Need Help path selection
tk.Button(root, text="Select Need Help Data Path", command=select_needhelp_path).pack(pady=5)
tk.Button(root, text="Use MinIO",
          command=open_minio_gui("eyes4rescue-group-13", "fine_needhelp/test/", "needhelp")).pack(pady=5)
needhelp_path_label = tk.Label(root, text="Need Help Path: Not Selected", font=("Helvetica", 10), wraplength=500)
needhelp_path_label.pack(pady=5)

# Start training
tk.Button(root, text="Start Training", command=start_training).pack(pady=20)

root.mainloop()
