import os
import numpy as np
from PIL import Image
from tkinter import Tk, filedialog, messagebox
from keras.src.saving import load_model


# Load the trained model
def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()


# Preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess the image to meet the model's input requirements.
    """
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)  # Resize to 224x224
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Use the model to predict the help status of a person in the image
def predict_help_status(image_path, model):
    """
    Use the classifier to predict if the person in the image needs help.
    """
    processed_image = preprocess_image(image_path)
    if processed_image is None:
        return

    prediction = model.predict(processed_image)[0][0]  # Get the prediction value

    if prediction > 0.8:
        result = f"The person in the image likely NEEDS help. Confidence: {prediction:.2f}"
    else:
        result = f"The person in the image is likely FINE. Confidence: {1 - prediction:.2f}"

    print(f"Prediction for {image_path}: {result}")
    return result


# Test a single image
def test_single_image(model):
    """
    Allow the user to select a single image and perform prediction.
    """
    Tk().withdraw()  # Hide the main window
    image_path = filedialog.askopenfilename(title="Select an Image",
                                            filetypes=[("Image Files", "*.jpg;*.png")])
    if not image_path:
        messagebox.showerror("Error", "No image selected.")
        return

    print(f"Selected image: {image_path}")
    result = predict_help_status(image_path, model)
    if result:
        messagebox.showinfo("Prediction Result", result)


# Test multiple images in a folder
def test_multiple_images(model):
    """
    Allow the user to select a folder and iterate through all images for prediction.
    """
    Tk().withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title="Select a Folder of Images")
    if not folder_path:
        messagebox.showerror("Error", "No folder selected.")
        return

    print(f"Selected folder: {folder_path}")
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):
            image_path = os.path.join(folder_path, file_name)
            result = predict_help_status(image_path, model)
            if result:
                print(result)


# Main program
if __name__ == "__main__":
    # Prompt the user to select the trained model
    model_path = filedialog.askopenfilename(title="Select the Trained Model",
                                            filetypes=[("Keras Model", "*.keras")])
    if not model_path:
        messagebox.showerror("Error", "No model selected.")
        exit()

    model = load_trained_model(model_path)

    # Menu for selecting single image or batch processing
    while True:
        print("\nMenu:")
        print("1. Test a Single Image")
        print("2. Test Multiple Images in a Folder")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            test_single_image(model)
        elif choice == "2":
            test_multiple_images(model)
        elif choice == "3":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
