# src/train/classifier/random_forest/gui_RF.py

import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import logging
from pathlib import Path
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd

from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Import centralized configuration
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# Define paths using centralized configuration
PROJECT_ROOT = settings.PROJECT_ROOT
MODELS_DIR = settings.MODELS_DIR
CLASSIFIER_DATASET = settings.CLASSIFIER_DATASET
OUTPUT_DIR = settings.OUTPUT_DIR

# Default paths from settings
DEFAULT_FINE_PATH = settings.TEST_FINE_DIR
DEFAULT_NEEDHELP_PATH = settings.TEST_NEEDHELP_DIR

# MinIO GUI path
GUI_MINIO_PATH = PROJECT_ROOT / "src" / "gui" / "gui_minio.py"

# Global paths for GUI
fine_path = str(DEFAULT_FINE_PATH) if DEFAULT_FINE_PATH.exists() else None
needhelp_path = str(DEFAULT_NEEDHELP_PATH) if DEFAULT_NEEDHELP_PATH.exists() else None

# YOLO pose model - using settings configuration
POSE_MODEL_PATH = settings.POSE_MODEL_PATH

# Initialize empty dataframe
pose_df = pd.DataFrame(columns=["image_path", "labels", "features"])


def setup_directories():
    """
    Create necessary directories based on configuration.
    """
    directories = [
        MODELS_DIR / "classifier",
        OUTPUT_DIR,
        CLASSIFIER_DATASET,
        settings.TRAINING_RUNS_DIR / "random_forest",
        settings.TEMP_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def get_classifier_save_paths():
    """
    Get paths for saving classifier and feature files.

    Returns:
        dict: Dictionary containing all save paths
    """
    rf_dir = MODELS_DIR / "classifier"
    rf_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = {
        'classifier': rf_dir / "rf_classifier.pkl",
        'fine_features': rf_dir / "fine_features.pkl",
        'fine_labels': rf_dir / "fine_labels.pkl",
        'needhelp_features': rf_dir / "needhelp_features.pkl",
        'needhelp_labels': rf_dir / "needhelp_labels.pkl",
        'backup_classifier': rf_dir / f"rf_classifier_backup_{timestamp}.pkl"
    }

    return paths


def load_pose_model():
    """
    Load YOLO pose model from configuration.

    Returns:
        YOLO: Loaded pose model
    """
    try:
        if not POSE_MODEL_PATH.exists():
            logger.warning(f"Pose model not found at {POSE_MODEL_PATH}, downloading...")
            model = YOLO("yolo11n-pose.pt")  # This will download if not present
        else:
            model = YOLO(str(POSE_MODEL_PATH))

        logger.info(f"Pose model loaded from: {POSE_MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Failed to load pose model: {e}")
        raise


def extract_features_from_image(image_path, label, model):
    """
    Extract pose features from an image using YOLO pose model.

    Args:
        image_path (Path): Path to the image
        label (str): Label for the image
        model (YOLO): YOLO pose model

    Returns:
        np.ndarray: Feature vector
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not load image: {image_path}")
            return None

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.uint8)

        # Get pose features using YOLO model
        results = model(img, verbose=False)
        keypoints = results[0].keypoints

        # Check if keypoints are present
        if keypoints is not None:
            # Extract keypoint coordinates (xy)
            keypoints_xy = keypoints.xy.cpu().numpy()
            # Flatten keypoints (x, y) values
            features = keypoints_xy.flatten()
        else:
            features = np.zeros(34)

        return features

    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
        return None


def extract_features_from_folder(folder, label, model):
    """
    Extract features from all images in a folder.

    Args:
        folder (Path): Path to the folder
        label (str): Label for the images
        model (YOLO): YOLO pose model

    Returns:
        tuple: (feature_vectors, labels)
    """
    folder_path = Path(folder)
    feature_vectors = []
    labels = []
    processed_count = 0

    logger.info(f"Processing folder: {folder_path}")

    # Get all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))

    total_images = len(image_files)
    logger.info(f"Found {total_images} images in {folder_path}")

    for image_path in image_files:
        features = extract_features_from_image(image_path, label, model)
        if features is not None:
            feature_vectors.append(features)
            labels.append(label)
            processed_count += 1

            # Log progress every 50 images
            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count}/{total_images} images from {label} folder")

    logger.info(f"Successfully processed {processed_count}/{total_images} images from {label} folder")
    return feature_vectors, labels


def validate_dataset_paths(fine_dir, needhelp_dir):
    """
    Validate that the dataset paths exist and contain images.

    Args:
        fine_dir (str): Path to fine images directory
        needhelp_dir (str): Path to needhelp images directory

    Returns:
        bool: True if paths are valid, False otherwise
    """
    fine_path = Path(fine_dir)
    needhelp_path = Path(needhelp_dir)

    if not fine_path.exists():
        logger.error(f"Fine directory does not exist: {fine_path}")
        return False

    if not needhelp_path.exists():
        logger.error(f"Need help directory does not exist: {needhelp_path}")
        return False

    # Check if directories contain images
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    fine_images = []
    needhelp_images = []

    for ext in image_extensions:
        fine_images.extend(list(fine_path.glob(f"*{ext}")) + list(fine_path.glob(f"*{ext.upper()}")))
        needhelp_images.extend(list(needhelp_path.glob(f"*{ext}")) + list(needhelp_path.glob(f"*{ext.upper()}")))

    if len(fine_images) == 0:
        logger.error(f"No images found in fine directory: {fine_path}")
        return False

    if len(needhelp_images) == 0:
        logger.error(f"No images found in needhelp directory: {needhelp_path}")
        return False

    logger.info(f"Dataset validation successful:")
    logger.info(f"  Fine images: {len(fine_images)}")
    logger.info(f"  Need help images: {len(needhelp_images)}")

    return True


def build_and_train_model(fine_dir, needhelp_dir):
    """
    Extract features from images, train a Random Forest classifier, and evaluate it.

    Args:
        fine_dir (str): Path to fine images directory
        needhelp_dir (str): Path to needhelp images directory

    Returns:
        dict: Training results and metrics
    """
    logger.info("Starting Random Forest training process...")

    # Validate paths
    if not validate_dataset_paths(fine_dir, needhelp_dir):
        raise ValueError("Invalid dataset paths")

    # Setup directories
    setup_directories()

    # Get save paths
    save_paths = get_classifier_save_paths()

    # Load pose model
    logger.info("Loading YOLO pose model...")
    model = load_pose_model()

    # Check if cached features exist
    if (save_paths['fine_features'].exists() and save_paths['needhelp_features'].exists() and
            save_paths['fine_labels'].exists() and save_paths['needhelp_labels'].exists()):

        logger.info("Loading cached features and labels...")
        fine_features = joblib.load(save_paths['fine_features'])
        fine_labels = joblib.load(save_paths['fine_labels'])
        needhelp_features = joblib.load(save_paths['needhelp_features'])
        needhelp_labels = joblib.load(save_paths['needhelp_labels'])

    else:
        logger.info("Extracting features from images...")

        # Extract features for both classes
        fine_features, fine_labels = extract_features_from_folder(Path(fine_dir), "Fine", model)
        needhelp_features, needhelp_labels = extract_features_from_folder(Path(needhelp_dir), "Need Help", model)

        if not fine_features or not needhelp_features:
            raise ValueError("No valid features extracted. Please check your dataset and paths.")

        # Save features and labels for future use
        logger.info("Saving extracted features...")
        joblib.dump(fine_features, save_paths['fine_features'])
        joblib.dump(fine_labels, save_paths['fine_labels'])
        joblib.dump(needhelp_features, save_paths['needhelp_features'])
        joblib.dump(needhelp_labels, save_paths['needhelp_labels'])

    # Prepare features for training
    logger.info("Preparing features for training...")
    max_length = 34  # Standard pose keypoint feature size

    fine_features_padded = [
        np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length]
        for f in fine_features
    ]
    needhelp_features_padded = [
        np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length]
        for f in needhelp_features
    ]

    # Combine features and labels
    X = np.concatenate([fine_features_padded, needhelp_features_padded], axis=0)
    y = np.array(fine_labels + needhelp_labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logger.info(f"Dataset prepared:")
    logger.info(f"  Total samples: {len(X)}")
    logger.info(f"  Feature dimension: {X.shape[1]}")
    logger.info(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    logger.info(f"Data split:")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")

    # Train or load the classifier
    if save_paths['classifier'].exists():
        logger.info("Loading existing classifier...")
        clf = joblib.load(save_paths['classifier'])

        # Check if we should retrain
        retrain = messagebox.askyesno(
            "Existing Model Found",
            "A trained classifier already exists. Do you want to retrain it?"
        )

        if retrain:
            logger.info("Retraining classifier...")
            clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
            clf.fit(X_train, y_train)

            # Save backup of old model
            if save_paths['classifier'].exists():
                joblib.dump(clf, save_paths['backup_classifier'])

            # Save new model
            joblib.dump(clf, save_paths['classifier'])
            logger.info(f"Retrained classifier saved to: {save_paths['classifier']}")

    else:
        logger.info("Training new Random Forest classifier...")
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        clf.fit(X_train, y_train)
        joblib.dump(clf, save_paths['classifier'])
        logger.info(f"New classifier saved to: {save_paths['classifier']}")

    # Evaluate the model
    logger.info("Evaluating model performance...")
    y_pred = clf.predict(X_test)

    # Generate detailed results
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results
    report_str = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    logger.info(f"Classification Report:\n{report_str}")

    # Calculate feature importance
    feature_importance = clf.feature_importances_

    # Prepare results dictionary
    results = {
        'classifier': clf,
        'label_encoder': label_encoder,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance,
        'test_accuracy': report['accuracy'],
        'save_paths': save_paths,
        'dataset_info': {
            'total_samples': len(X),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'fine_samples': len(fine_features),
            'needhelp_samples': len(needhelp_features)
        }
    }

    return results


def save_training_results(results):
    """
    Save training results to files.

    Args:
        results (dict): Training results dictionary
    """
    try:
        results_dir = settings.TRAINING_RUNS_DIR / "random_forest"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save classification report
        report_path = results_dir / f"classification_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write("Random Forest Classifier Training Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training completed at: {datetime.now()}\n\n")

            # Dataset information
            f.write("Dataset Information:\n")
            for key, value in results['dataset_info'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            # Classification report
            f.write("Classification Report:\n")
            for class_name, metrics in results['classification_report'].items():
                if isinstance(metrics, dict):
                    f.write(f"  {class_name}:\n")
                    for metric, value in metrics.items():
                        f.write(f"    {metric}: {value:.4f}\n")
                else:
                    f.write(f"  {class_name}: {metrics:.4f}\n")

        logger.info(f"Training results saved to: {report_path}")

    except Exception as e:
        logger.warning(f"Failed to save training results: {e}")


def start_training():
    """Start training the model and display the results in the GUI."""
    global fine_path, needhelp_path

    if not fine_path or not needhelp_path:
        messagebox.showerror("Error", "Please select both Fine and Need Help paths.")
        return

    try:
        logger.info("Starting Random Forest training...")
        results = build_and_train_model(fine_path, needhelp_path)

        # Save results
        save_training_results(results)

        # Display results in message box
        accuracy = results['test_accuracy']
        dataset_info = results['dataset_info']

        success_message = (
            f"Training completed successfully!\n\n"
            f"Test Accuracy: {accuracy:.4f}\n"
            f"Total Samples: {dataset_info['total_samples']}\n"
            f"Training Samples: {dataset_info['training_samples']}\n"
            f"Test Samples: {dataset_info['test_samples']}\n\n"
            f"Model saved to: {results['save_paths']['classifier']}"
        )

        messagebox.showinfo("Training Complete", success_message)

        # Print detailed report to console
        print("\n" + "=" * 60)
        print("RANDOM FOREST TRAINING RESULTS")
        print("=" * 60)
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Model Location: {results['save_paths']['classifier']}")
        print("=" * 60)

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        messagebox.showerror("Training Error", error_msg)


def select_fine_path():
    """Allow the user to select the fine data path."""
    global fine_path
    path = filedialog.askdirectory(
        title="Select Fine Data Path",
        initialdir=str(CLASSIFIER_DATASET)
    )
    if path:
        fine_path = path
        fine_path_label.config(text=f"Fine Path: {fine_path}")
        logger.info(f"Fine path selected: {fine_path}")
        messagebox.showinfo("Selected Path", f"Fine path set to: {fine_path}")


def select_needhelp_path():
    """Allow the user to select the need-help data path."""
    global needhelp_path
    path = filedialog.askdirectory(
        title="Select Need Help Data Path",
        initialdir=str(CLASSIFIER_DATASET)
    )
    if path:
        needhelp_path = path
        needhelp_path_label.config(text=f"Need Help Path: {needhelp_path}")
        logger.info(f"Need help path selected: {needhelp_path}")
        messagebox.showinfo("Selected Path", f"Need Help path set to: {needhelp_path}")


def open_minio_gui(bucket_name, prefix, path_type):
    """
    Open the MinIO GUI for selecting files and folders.

    Args:
        bucket_name (str): MinIO bucket name
        prefix (str): Prefix path in bucket
        path_type (str): Type of path ('fine' or 'needhelp')
    """

    def callback():
        global fine_path, needhelp_path
        try:
            # Check if MinIO GUI exists
            if not GUI_MINIO_PATH.exists():
                messagebox.showerror("Error", f"MinIO GUI not found at: {GUI_MINIO_PATH}")
                return

            result = subprocess.run(
                ["python", str(GUI_MINIO_PATH), bucket_name, prefix],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"MinIO GUI error: {result.stderr}")
                messagebox.showerror("Error", f"Failed to open MinIO GUI: {result.stderr}")
                return

            selected_path = result.stdout.strip()
            selected_path = os.path.normpath(selected_path)

            if not os.path.exists(selected_path):
                logger.error(f"Selected path does not exist: {selected_path}")
                messagebox.showerror("Error", f"Selected path does not exist: {selected_path}")
                return

            if path_type == "fine":
                fine_path = selected_path
                fine_path_label.config(text=f"Fine Path: {fine_path}")
                logger.info(f"MinIO fine path selected: {fine_path}")
                messagebox.showinfo("Selected Path", f"Fine path set to: {fine_path}")
            elif path_type == "needhelp":
                needhelp_path = selected_path
                needhelp_path_label.config(text=f"Need Help Path: {needhelp_path}")
                logger.info(f"MinIO need help path selected: {needhelp_path}")
                messagebox.showinfo("Selected Path", f"Need Help path set to: {needhelp_path}")

        except subprocess.TimeoutExpired:
            messagebox.showerror("Error", "MinIO GUI operation timed out.")
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    return callback


def create_gui():
    """
    Create the GUI for the Random Forest training application.
    """
    global fine_path_label, needhelp_path_label

    # Create main window
    root = tk.Tk()
    root.title("Random Forest Trainer - Pose Classification")
    root.geometry("800x600")
    root.resizable(True, True)

    # Main title
    title_label = tk.Label(root, text="Random Forest Pose Classifier Trainer",
                           font=("Arial", 16, "bold"), fg="#2c3e50")
    title_label.pack(pady=20)

    # Configuration info
    config_frame = tk.Frame(root, bg="#ecf0f1", relief=tk.RIDGE, bd=1)
    config_frame.pack(fill=tk.X, padx=20, pady=10)

    config_title = tk.Label(config_frame, text="Configuration:",
                            font=("Arial", 12, "bold"), bg="#ecf0f1")
    config_title.pack(anchor=tk.W, padx=10, pady=5)

    config_info = [
        f"Project Root: {PROJECT_ROOT}",
        f"Models Directory: {MODELS_DIR / 'classifier'}",
        f"Pose Model: {POSE_MODEL_PATH}",
        f"Default Fine Path: {DEFAULT_FINE_PATH}",
        f"Default Need Help Path: {DEFAULT_NEEDHELP_PATH}"
    ]

    for info in config_info:
        info_label = tk.Label(config_frame, text=info, font=("Arial", 9),
                              bg="#ecf0f1", fg="#34495e")
        info_label.pack(anchor=tk.W, padx=20, pady=1)

    # Fine path selection
    fine_frame = tk.Frame(root)
    fine_frame.pack(pady=15, fill=tk.X, padx=20)

    tk.Button(fine_frame, text="Select Fine Data Path",
              command=select_fine_path, font=("Arial", 10),
              bg="#27ae60", fg="white", padx=15, pady=5).pack(side=tk.LEFT)

    tk.Button(fine_frame, text="Use MinIO",
              command=open_minio_gui("eyes4rescue-group-13", "fine_needhelp/test/fine/", "fine"),
              font=("Arial", 10), bg="#9b59b6", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=10)

    fine_path_label = tk.Label(root, text=f"Fine Path: {fine_path or 'Not Selected'}",
                               font=("Arial", 10), wraplength=700, anchor="w")
    fine_path_label.pack(pady=5, padx=20, fill=tk.X)

    # Need Help path selection
    needhelp_frame = tk.Frame(root)
    needhelp_frame.pack(pady=15, fill=tk.X, padx=20)

    tk.Button(needhelp_frame, text="Select Need Help Data Path",
              command=select_needhelp_path, font=("Arial", 10),
              bg="#e74c3c", fg="white", padx=15, pady=5).pack(side=tk.LEFT)

    tk.Button(needhelp_frame, text="Use MinIO",
              command=open_minio_gui("eyes4rescue-group-13", "fine_needhelp/test/needhelp/", "needhelp"),
              font=("Arial", 10), bg="#9b59b6", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=10)

    needhelp_path_label = tk.Label(root, text=f"Need Help Path: {needhelp_path or 'Not Selected'}",
                                   font=("Arial", 10), wraplength=700, anchor="w")
    needhelp_path_label.pack(pady=5, padx=20, fill=tk.X)

    # Start training button
    tk.Button(root, text="Start Training", command=start_training,
              font=("Arial", 14, "bold"), bg="#3498db", fg="white",
              padx=30, pady=10, cursor="hand2").pack(pady=30)

    # Model information
    model_info_frame = tk.Frame(root, bg="#f8f9fa", relief=tk.RIDGE, bd=1)
    model_info_frame.pack(fill=tk.X, padx=20, pady=10)

    model_info_title = tk.Label(model_info_frame, text="Model Information:",
                                font=("Arial", 11, "bold"), bg="#f8f9fa")
    model_info_title.pack(anchor=tk.W, padx=10, pady=5)

    model_info = [
        "• Algorithm: Random Forest Classifier",
        "• Feature Extraction: YOLO11n-Pose keypoints",
        "• Feature Dimension: 34 (17 keypoints × 2 coordinates)",
        "• Classes: Fine, Need Help",
        "• Train/Test Split: 70/30 with stratification"
    ]

    for info in model_info:
        info_label = tk.Label(model_info_frame, text=info,
                              font=("Arial", 9), bg="#f8f9fa", fg="#495057")
        info_label.pack(anchor=tk.W, padx=20, pady=1)

    # Instructions
    instructions_frame = tk.Frame(root, bg="#fff3cd", relief=tk.RIDGE, bd=1)
    instructions_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    instructions_title = tk.Label(instructions_frame, text="Instructions:",
                                  font=("Arial", 11, "bold"), bg="#fff3cd")
    instructions_title.pack(anchor=tk.W, padx=10, pady=5)

    instructions = [
        "1. Select 'Fine' data directory containing images of people in normal poses",
        "2. Select 'Need Help' data directory containing images of people needing help",
        "3. Ensure directories contain image files (jpg, png, etc.)",
        "4. Click 'Start Training' to extract features and train the Random Forest",
        "5. Model and features will be cached for future use",
        "6. Training results will be displayed and saved automatically"
    ]

    for instruction in instructions:
        inst_label = tk.Label(instructions_frame, text=instruction,
                              font=("Arial", 9), bg="#fff3cd", fg="#856404")
        inst_label.pack(anchor=tk.W, padx=20, pady=2)

    return root


def main():
    """
    Main function to run the Random Forest training application.
    """
    try:
        # Initialize configuration
        logger.info("Initializing Random Forest Trainer...")
        logger.info(f"Project root: {PROJECT_ROOT}")
        logger.info(f"Using configuration from: {settings}")

        # Setup directories
        setup_directories()

        # Create and run GUI
        root = create_gui()
        root.mainloop()

    except Exception as e:
        error_msg = f"Application failed to start: {e}"
        logger.error(error_msg)
        if 'root' in locals():
            messagebox.showerror("Startup Error", error_msg)
        else:
            print(error_msg)


if __name__ == "__main__":
    main()