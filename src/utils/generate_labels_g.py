import os
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv


def create_labels_using_yolo(image_root_dir, model_dir_path):
    """
    Use a YOLO model to create label files for images based on detection results.

    Args:
        image_root_dir (str): Root directory containing image files.
        model_dir_path (str): Path to the trained YOLO model (.pt file).
    """
    # Ensure classes.txt exists, create if not
    classes_file = os.path.join(image_root_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        with open(classes_file, 'w') as f:
            f.write("person\nfalling_person\nsitting_person\nsleeping_person")
        print(f"Created classes.txt file: {classes_file}")

    # Load the YOLO model
    model = YOLO(model_dir_path)

    # Traverse all subdirectories and image files
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                label_file = image_path.replace('.png', '.txt')

                # Check if label file already exists
                if os.path.exists(label_file):
                    print(f"Label file already exists, skipping: {label_file}")
                    continue

                # Read the image
                img = cv2.imread(image_path)

                # Use the YOLO model to predict objects in the image
                results = model.predict(img)

                # If objects are detected, create or update the label file
                if results[0].boxes:
                    with open(label_file, 'w') as f:
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])  # Get class ID (0 = person, 1 = falling_person, etc.)
                            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                            img_height, img_width, _ = img.shape

                            # Convert coordinates to YOLO format (normalized center_x, center_y, width, height)
                            center_x = (x1 + x2) / 2 / img_width
                            center_y = (y1 + y2) / 2 / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height

                            # Write the label in YOLO format
                            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                        print(f"Created/Updated label file: {label_file}")
                else:
                    # If no objects detected, skip writing any label file
                    print(f"No objects detected, skipped: {image_path}")


def main():
    load_dotenv()

    # Define the root directory for image frames and labels
    image_root_directory = os.getenv('TRAINING_IMAGES_PATH')  # Path to your Image_root in .env
    model_path = os.getenv('YOLO_MODEL_PATH')  # Path to your YOLO model in .env

    # Check if the folder exists
    if image_root_directory is None or model_path is None:
        print("Error: YOLO_MODEL_PATH or TRAINING_IMAGES_PATH not found in environment variables")
        return

    # Call the function to create label files using the YOLO model
    create_labels_using_yolo(image_root_directory, model_path)


if __name__ == "__main__":
    main()
