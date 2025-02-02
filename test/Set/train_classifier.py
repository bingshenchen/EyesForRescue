import os
from dotenv import load_dotenv
from keras import layers, models, applications, optimizers
import matplotlib.pyplot as plt
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Load environment variables
load_dotenv()

# Define paths
PROJECT_ROOT = os.getenv('PROJECT_ROOT', '.')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')
train_path = os.getenv('TRAINING_CLASSIFIER_PATH', './train')
test_path = os.getenv('TEST_CLASSIFIER_PATH', './test')

# Check if paths exist
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training path does not exist: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test path does not exist: {test_path}")


def build_and_train_model():
    """
    Build and train a binary classification model using ResNet50 as the base model.
    """
    # Define data generators with augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=45,  # Increase rotation range
        width_shift_range=0.3,  # More horizontal shifts
        height_shift_range=0.3,  # More vertical shifts
        shear_range=0.3,  # Adjust shearing
        zoom_range=0.4,  # Wider zoom range
        horizontal_flip=True,  # Allow horizontal flipping
        fill_mode='nearest',
        brightness_range=(0.6, 1.4),  # Extend brightness adjustment
        channel_shift_range=70.0  # Larger color shift range
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=0.1,  # Minor zoom
        rotation_range=5  # Small rotation
    )
    # Create training and validation data generators
    train_generator = train_datagen.flow_from_directory(
        train_path, target_size=(224, 224), batch_size=16, class_mode="binary"
    )
    validation_generator = validation_datagen.flow_from_directory(
        test_path, target_size=(224, 224), batch_size=16, class_mode="binary"
    )

    # Use ResNet50 as the base model
    base_model = applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = True

    # Fine-tune the last 10 layers of the base model
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Build the full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Effective pooling for modern architectures
        layers.BatchNormalization(),  # Normalize activations
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),  # Slightly lower dropout
        layers.BatchNormalization(),  # Add another normalization layer
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid")
    ])

    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),  # Use Adam optimizer with a learning rate
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Define callbacks for early stopping and saving the best model
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, "best_model.keras"), save_best_only=True),
        ReduceLROnPlateau(
            monitor="val_loss",  # Monitor the validation loss
            factor=0.5,  # Reduce learning rate by half
            patience=3,  # Wait for 3 epochs with no improvement before reducing
            verbose=1,  # Log the reduction event
            min_lr=1e-6  # Set a lower bound for the learning rate
        )
    ]

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )

    # Save the final model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save(os.path.join(OUTPUT_DIR, "final_person_help_classifier.keras"))

    return history


def plot_history(history):
    """
    Plot the training and validation accuracy and loss curves.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # Build, train, and evaluate the model
    history = build_and_train_model()
    # Plot the training history
    plot_history(history)
