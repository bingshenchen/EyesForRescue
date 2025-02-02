import os
import numpy as np
from PIL import Image
from keras.src.saving import load_model
from sklearn.metrics import classification_report
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths from environment variables
MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH")
FINE_DIR = os.getenv("TEST_FINE_DIR")
NEEDHELP_DIR = os.getenv("TEST_NEEDHELP_DIR")


# Load the trained model
def load_trained_model(model_path):
    """
    Load the trained model from the given path.
    """
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()


# Preprocess images in batch
def preprocess_images_in_batch(directory, target_size=(224, 224)):
    """
    Load and preprocess all images in the given directory.
    """
    image_arrays = []
    file_paths = []

    for file_name in os.listdir(directory):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(directory, file_name)
            try:
                img = Image.open(image_path)
                img = img.resize(target_size)  # Resize to 224x224
                img_array = np.array(img) / 255.0  # Normalize
                image_arrays.append(img_array)
                file_paths.append(image_path)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")

    return np.array(image_arrays), file_paths


# Predict images in batch
def predict_images_in_batch(image_arrays, model, threshold=0.8):
    """
    Perform batch predictions on images using the trained model.
    """
    predictions = model.predict(image_arrays, verbose=0)  # Batch predictions
    labels = [1 if pred[0] > threshold else 0 for pred in predictions]  # Generate labels based on threshold
    return labels


# Evaluate directory in batch
def evaluate_directory_in_batch(directory, model, true_label):
    """
    Predict labels for all images in a directory and compare with the true label.
    """
    # Load and preprocess images in batch
    image_arrays, file_paths = preprocess_images_in_batch(directory)

    # Return empty lists if no valid images
    if len(image_arrays) == 0:
        return [], []

    # Batch predictions
    predictions = predict_images_in_batch(image_arrays, model)

    # Generate true label list
    true_labels = [true_label] * len(predictions)

    return predictions, true_labels


# Main script
if __name__ == "__main__":
    # Load the trained model
    model = load_trained_model(MODEL_PATH)

    # Evaluate the "fine" folder
    fine_predictions, fine_labels = evaluate_directory_in_batch(FINE_DIR, model, 0)

    # Evaluate the "need help" folder
    needhelp_predictions, needhelp_labels = evaluate_directory_in_batch(NEEDHELP_DIR, model, 1)

    # Combine results
    all_predictions = fine_predictions + needhelp_predictions
    all_true_labels = fine_labels + needhelp_labels

    # Generate classification report
    report = classification_report(all_true_labels, all_predictions, target_names=["Fine", "Need Help"])
    print("Classification Report:\n", report)
