import os
import subprocess
from dotenv import load_dotenv
from keras import layers, models, applications, optimizers
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox

from keras.src.applications.efficientnet_v2 import preprocess_input
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Load environment variables
load_dotenv()

# Define paths
PROJECT_ROOT = os.getenv('PROJECT_ROOT', '.')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')
train_path = os.getenv('TRAINING_CLASSIFIER_PATH', './train')
test_path = os.getenv('TEST_CLASSIFIER_PATH', './test')

GUI_MINIO_PATH = os.path.join(PROJECT_ROOT, "src", "gui", "gui_minio.py")


def build_and_train_model(train_path, test_path):
    """
    Build and train a binary classification model using EfficientNetV2M as the base model.
    """
    # Define data generators with augmentation for training data
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
    train_generator = train_datagen.flow_from_directory(
        train_path, target_size=(480, 480), batch_size=10, class_mode="binary"
    )
    validation_generator = validation_datagen.flow_from_directory(
        test_path, target_size=(480, 480), batch_size=10, class_mode="binary"
    )

    # Load the EfficientNetV2M model with pre-trained ImageNet weights
    base_model = applications.EfficientNetV2M(include_top=False, weights="imagenet", input_shape=(480, 480, 3))
    base_model.trainable = False  # Freeze the base model

    # Add custom top layers
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
        optimizer=optimizers.Adam(learning_rate=3e-4),  # Adjusted learning rate for better optimization
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, "best_model.keras"), save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6)
    ]

    # Train the top layers
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )

    base_model.trainable = True
    for layer in base_model.layers[:300]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Fine-tune the model
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,  # Reduced epochs for fine-tuning
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Save the final model
    model.save(os.path.join(OUTPUT_DIR, "final_person_help_classifier.keras"))

    return history, history_fine



def plot_history(history, history_fine):
    """
    Plot the training and validation accuracy and loss curves.
    """
    # Combine history data
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()



def start_training():
    """
    Start training the model and display the results in the GUI.
    """
    try:
        history = build_and_train_model(train_path, test_path)
        messagebox.showinfo("Success", "Training completed successfully!")
        plot_history(history)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def select_train_path():
    """
    Allow the user to select the training data path and update the GUI.
    """
    global train_path
    path = filedialog.askdirectory(title="Select Training Data Path")
    if path:
        train_path = path
        train_path_label.config(text=f"Training Path: {train_path}")
        messagebox.showinfo("Selected Path", f"Training path set to: {train_path}")


def select_test_path():
    """
    Allow the user to select the test data path and update the GUI.
    """
    global test_path
    path = filedialog.askdirectory(title="Select Test Data Path")
    if path:
        test_path = path
        test_path_label.config(text=f"Test Path: {test_path}")
        messagebox.showinfo("Selected Path", f"Test path set to: {test_path}")


import subprocess


def open_minio_gui(bucket_name, prefix, path_type):
    """
    Open the MinIO GUI for selecting files and folders.
    """

    def callback():
        global train_path, test_path
        try:
            result = subprocess.run(
                ["python", GUI_MINIO_PATH, bucket_name, prefix],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                messagebox.showerror("Error", f"Failed to open MinIO GUI: {result.stderr}")
                return

            selected_path = result.stdout.strip()
            selected_path = os.path.normpath(selected_path)
            if not os.path.exists(selected_path):
                print(f"Selected path does not exist: {selected_path}")
                messagebox.showerror("Error", f"Selected path does not exist: {selected_path}")
                return

            if path_type == "train":
                train_path = selected_path
                print(f"train_path: {train_path}")
                train_path_label.config(text=f"Training Path: {train_path}")
            elif path_type == "test":
                test_path = selected_path
                print(f"test_path: {test_path}")
                test_path_label.config(text=f"Test Path: {test_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            print(f"Error: {e}")

    return callback


# GUI setup
root = tk.Tk()
root.title("AI Model Trainer")
root.geometry("600x400")

tk.Button(root, text="Select Training Data Path", command=select_train_path).pack(pady=5)
tk.Button(root, text="Use MinIO", command=open_minio_gui("eyes4rescue-group-13", "fine_needhelp/", "train")).pack(
    pady=5)
train_path_label = tk.Label(root, text=f"Training Path: {train_path}", font=("Helvetica", 10), wraplength=500)
train_path_label.pack(pady=5)

tk.Button(root, text="Select Test Data Path", command=select_test_path).pack(pady=5)
tk.Button(root, text="Use MinIO", command=open_minio_gui("eyes4rescue-group-13", "fine_needhelp/", "test")).pack(pady=5)
test_path_label = tk.Label(root, text=f"Test Path: {test_path}", font=("Helvetica", 10), wraplength=500)
test_path_label.pack(pady=5)

tk.Button(root, text="Start Training", command=start_training).pack(pady=20)

root.mainloop()
