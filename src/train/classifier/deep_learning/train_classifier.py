# src/train/classifier/deep_learning/train_classifier.py

import os
import subprocess
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

from keras import layers, models, applications, optimizers
from keras.src.applications.efficientnet_v2 import preprocess_input
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator

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
OUTPUT_DIR = settings.OUTPUT_DIR
CLASSIFIER_DATASET = settings.CLASSIFIER_DATASET

# Default paths from settings
DEFAULT_TRAIN_PATH = settings.TRAINING_CLASSIFIER_PATH
DEFAULT_TEST_PATH = settings.TEST_CLASSIFIER_PATH

# MinIO GUI path
GUI_MINIO_PATH = PROJECT_ROOT / "src" / "gui" / "gui_minio.py"

# Global variables for paths
train_path = str(DEFAULT_TRAIN_PATH) if DEFAULT_TRAIN_PATH.exists() else ""
test_path = str(DEFAULT_TEST_PATH) if DEFAULT_TEST_PATH.exists() else ""


def setup_directories():
    """
    Create necessary directories based on configuration.
    """
    directories = [
        MODELS_DIR / "classifier",
        OUTPUT_DIR,
        CLASSIFIER_DATASET,
        settings.TRAINING_RUNS_DIR,
        settings.TEMP_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {directory}")


def get_model_save_path():
    """
    Get the path to save the trained model.
    """
    model_dir = MODELS_DIR / "classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "final_person_help_classifier.keras"


def get_checkpoint_path():
    """
    Get the path for model checkpoints during training.
    """
    checkpoint_dir = settings.TRAINING_RUNS_DIR / "classifier_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return checkpoint_dir / f"best_model_{timestamp}.keras"


def validate_dataset_paths(train_dir, test_dir):
    """
    Validate that the dataset paths exist and contain data.

    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test data directory

    Returns:
        bool: True if paths are valid, False otherwise
    """
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    if not train_path.exists():
        logger.error(f"Training directory does not exist: {train_path}")
        return False

    if not test_path.exists():
        logger.error(f"Test directory does not exist: {test_path}")
        return False

    # Check if directories contain subdirectories (classes)
    train_subdirs = [d for d in train_path.iterdir() if d.is_dir()]
    test_subdirs = [d for d in test_path.iterdir() if d.is_dir()]

    if len(train_subdirs) < 2:
        logger.error(f"Training directory must contain at least 2 class subdirectories")
        return False

    if len(test_subdirs) < 2:
        logger.error(f"Test directory must contain at least 2 class subdirectories")
        return False

    logger.info(f"Dataset validation successful:")
    logger.info(f"  Training classes: {[d.name for d in train_subdirs]}")
    logger.info(f"  Test classes: {[d.name for d in test_subdirs]}")

    return True


def build_and_train_model(train_path, test_path):
    """
    Build and train a binary classification model using EfficientNetV2M as the base model.

    Args:
        train_path (str): Path to training data directory
        test_path (str): Path to test data directory

    Returns:
        tuple: (history, history_fine) - Training histories
    """
    logger.info("Starting model training process...")

    # Validate paths
    if not validate_dataset_paths(train_path, test_path):
        raise ValueError("Invalid dataset paths")

    # Setup directories
    setup_directories()

    # Get model save paths
    model_save_path = get_model_save_path()
    checkpoint_path = get_checkpoint_path()

    logger.info(f"Model will be saved to: {model_save_path}")
    logger.info(f"Checkpoints will be saved to: {checkpoint_path}")

    # Define data generators with augmentation for training data
    logger.info("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        preprocessing_function=applications.efficientnet_v2.preprocess_input,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(
        preprocessing_function=applications.efficientnet_v2.preprocess_input
    )

    # Create data generators
    try:
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(480, 480),
            batch_size=10,
            class_mode="binary"
        )

        validation_generator = validation_datagen.flow_from_directory(
            test_path,
            target_size=(480, 480),
            batch_size=10,
            class_mode="binary"
        )

        logger.info(f"Training samples: {train_generator.samples}")
        logger.info(f"Validation samples: {validation_generator.samples}")
        logger.info(f"Class indices: {train_generator.class_indices}")

    except Exception as e:
        logger.error(f"Error creating data generators: {e}")
        raise

    # Load the EfficientNetV2M model with pre-trained ImageNet weights
    logger.info("Loading EfficientNetV2M base model...")
    try:
        base_model = applications.EfficientNetV2M(
            include_top=False,
            weights="imagenet",
            input_shape=(480, 480, 3)
        )
        base_model.trainable = False  # Freeze the base model
        logger.info("Base model loaded and frozen successfully")
    except Exception as e:
        logger.error(f"Error loading base model: {e}")
        raise

    # Add custom top layers
    logger.info("Building model architecture...")
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    logger.info("Model compiled successfully")
    logger.info(f"Model summary:\n{model.summary()}")

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-6
        )
    ]

    # Train the top layers
    logger.info("Starting initial training phase...")
    try:
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("Initial training phase completed")
    except Exception as e:
        logger.error(f"Error during initial training: {e}")
        raise

    # Fine-tuning phase
    logger.info("Starting fine-tuning phase...")
    try:
        # Unfreeze the base model for fine-tuning
        base_model.trainable = True

        # Freeze early layers
        for layer in base_model.layers[:300]:
            layer.trainable = False

        # Recompile with lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        # Fine-tune the model
        history_fine = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            epochs=10,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // validation_generator.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        logger.info("Fine-tuning phase completed")
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise

    # Save the final model
    try:
        model.save(str(model_save_path))
        logger.info(f"Final model saved to: {model_save_path}")

        # Also update the settings classifier path if this is the best model
        settings_classifier_path = settings.CLASSIFIER_MODEL_PATH
        if settings_classifier_path.parent != model_save_path.parent:
            model.save(str(settings_classifier_path))
            logger.info(f"Model also saved to settings path: {settings_classifier_path}")

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

    return history, history_fine


def plot_history(history, history_fine, save_plots=True):
    """
    Plot the training and validation accuracy and loss curves.

    Args:
        history: Training history from initial phase
        history_fine: Training history from fine-tuning phase
        save_plots (bool): Whether to save plots to disk
    """
    logger.info("Generating training plots...")

    try:
        # Safely combine history data
        phase1_acc = history.history.get('accuracy', [])
        phase1_val_acc = history.history.get('val_accuracy', [])
        phase1_loss = history.history.get('loss', [])
        phase1_val_loss = history.history.get('val_loss', [])

        phase2_acc = history_fine.history.get('accuracy', [])
        phase2_val_acc = history_fine.history.get('val_accuracy', [])
        phase2_loss = history_fine.history.get('loss', [])
        phase2_val_loss = history_fine.history.get('val_loss', [])

        # Combine phases
        acc = phase1_acc + phase2_acc
        val_acc = phase1_val_acc + phase2_val_acc
        loss = phase1_loss + phase2_loss
        val_loss = phase1_val_loss + phase2_val_loss

        # Ensure all lists have the same length
        min_length = min(len(acc), len(val_acc), len(loss), len(val_loss))
        if min_length == 0:
            logger.warning("No training history data available for plotting")
            return

        acc = acc[:min_length]
        val_acc = val_acc[:min_length]
        loss = loss[:min_length]
        val_loss = val_loss[:min_length]

        epochs = range(1, len(acc) + 1)

        logger.info(f"Plotting {len(epochs)} epochs of training history")

    except Exception as e:
        logger.error(f"Error preparing plot data: {e}")
        return

    plt.figure(figsize=(14, 5))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_plots:
        try:
            plots_dir = settings.TRAINING_RUNS_DIR / "classifier_plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = plots_dir / f"training_history_{timestamp}.png"

            plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
            logger.info(f"Training plots saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to save plots: {e}")

    plt.show()


def start_training():
    """
    Start training the model and display the results in the GUI.
    """
    global train_path, test_path

    if not train_path or not test_path:
        messagebox.showerror("Error", "Please select both training and test data paths.")
        return

    try:
        logger.info("Starting training process...")
        history, history_fine = build_and_train_model(train_path, test_path)

        # Plot results
        plot_history(history, history_fine)

        messagebox.showinfo("Success",
                            f"Training completed successfully!\n"
                            f"Model saved to: {get_model_save_path()}")

    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        messagebox.showerror("Error", error_msg)


def select_train_path():
    """
    Allow the user to select the training data path and update the GUI.
    """
    global train_path
    path = filedialog.askdirectory(
        title="Select Training Data Path",
        initialdir=str(CLASSIFIER_DATASET)
    )
    if path:
        train_path = path
        train_path_label.config(text=f"Training Path: {train_path}")
        logger.info(f"Training path selected: {train_path}")
        messagebox.showinfo("Selected Path", f"Training path set to: {train_path}")


def select_test_path():
    """
    Allow the user to select the test data path and update the GUI.
    """
    global test_path
    path = filedialog.askdirectory(
        title="Select Test Data Path",
        initialdir=str(CLASSIFIER_DATASET)
    )
    if path:
        test_path = path
        test_path_label.config(text=f"Test Path: {test_path}")
        logger.info(f"Test path selected: {test_path}")
        messagebox.showinfo("Selected Path", f"Test path set to: {test_path}")


def open_minio_gui(bucket_name, prefix, path_type):
    """
    Open the MinIO GUI for selecting files and folders.

    Args:
        bucket_name (str): MinIO bucket name
        prefix (str): Prefix path in bucket
        path_type (str): Type of path ('train' or 'test')
    """

    def callback():
        global train_path, test_path
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

            if path_type == "train":
                train_path = selected_path
                logger.info(f"MinIO training path selected: {train_path}")
                train_path_label.config(text=f"Training Path: {train_path}")
                messagebox.showinfo("Selected Path", f"Training path set to: {train_path}")
            elif path_type == "test":
                test_path = selected_path
                logger.info(f"MinIO test path selected: {test_path}")
                test_path_label.config(text=f"Test Path: {test_path}")
                messagebox.showinfo("Selected Path", f"Test path set to: {test_path}")

        except subprocess.TimeoutExpired:
            messagebox.showerror("Error", "MinIO GUI operation timed out.")
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)

    return callback


def create_gui():
    """
    Create the GUI for the classifier training application.
    """
    global train_path_label, test_path_label

    # Create main window
    root = tk.Tk()
    root.title("AI Model Trainer - Deep Learning Classifier")
    root.geometry("700x500")
    root.resizable(True, True)

    # Main title
    title_label = tk.Label(root, text="Deep Learning Classifier Trainer",
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
        f"Models Directory: {MODELS_DIR}",
        f"Default Training Path: {DEFAULT_TRAIN_PATH}",
        f"Default Test Path: {DEFAULT_TEST_PATH}"
    ]

    for info in config_info:
        info_label = tk.Label(config_frame, text=info, font=("Arial", 9),
                              bg="#ecf0f1", fg="#34495e")
        info_label.pack(anchor=tk.W, padx=20, pady=1)

    # Training path selection
    train_frame = tk.Frame(root)
    train_frame.pack(pady=15, fill=tk.X, padx=20)

    tk.Button(train_frame, text="Select Training Data Path",
              command=select_train_path, font=("Arial", 10),
              bg="#3498db", fg="white", padx=15, pady=5).pack(side=tk.LEFT)

    tk.Button(train_frame, text="Use MinIO",
              command=open_minio_gui("eyes4rescue-group-13", "fine_needhelp/train/", "train"),
              font=("Arial", 10), bg="#9b59b6", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=10)

    train_path_label = tk.Label(root, text=f"Training Path: {train_path or 'Not Selected'}",
                                font=("Arial", 10), wraplength=600, anchor="w")
    train_path_label.pack(pady=5, padx=20, fill=tk.X)

    # Test path selection
    test_frame = tk.Frame(root)
    test_frame.pack(pady=15, fill=tk.X, padx=20)

    tk.Button(test_frame, text="Select Test Data Path",
              command=select_test_path, font=("Arial", 10),
              bg="#3498db", fg="white", padx=15, pady=5).pack(side=tk.LEFT)

    tk.Button(test_frame, text="Use MinIO",
              command=open_minio_gui("eyes4rescue-group-13", "fine_needhelp/test/", "test"),
              font=("Arial", 10), bg="#9b59b6", fg="white", padx=15, pady=5).pack(side=tk.LEFT, padx=10)

    test_path_label = tk.Label(root, text=f"Test Path: {test_path or 'Not Selected'}",
                               font=("Arial", 10), wraplength=600, anchor="w")
    test_path_label.pack(pady=5, padx=20, fill=tk.X)

    # Start training button
    tk.Button(root, text="Start Training", command=start_training,
              font=("Arial", 14, "bold"), bg="#27ae60", fg="white",
              padx=30, pady=10, cursor="hand2").pack(pady=30)

    # Instructions
    instructions_frame = tk.Frame(root, bg="#f8f9fa", relief=tk.RIDGE, bd=1)
    instructions_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    instructions_title = tk.Label(instructions_frame, text="Instructions:",
                                  font=("Arial", 11, "bold"), bg="#f8f9fa")
    instructions_title.pack(anchor=tk.W, padx=10, pady=5)

    instructions = [
        "1. Select training and test data directories or use MinIO",
        "2. Ensure directories contain class subdirectories (e.g., 'fine', 'needhelp')",
        "3. Click 'Start Training' to begin the training process",
        "4. Model will be saved automatically upon completion",
        "5. Training plots will be displayed and saved"
    ]

    for instruction in instructions:
        inst_label = tk.Label(instructions_frame, text=instruction,
                              font=("Arial", 9), bg="#f8f9fa", fg="#495057")
        inst_label.pack(anchor=tk.W, padx=20, pady=2)

    return root


def main():
    """
    Main function to run the classifier training application.
    """
    try:
        # Initialize configuration
        logger.info("Initializing Deep Learning Classifier Trainer...")
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