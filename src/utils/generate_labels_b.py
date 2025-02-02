import os
from dotenv import load_dotenv
import logging


def create_labels_for_images(image_root_dir, class_id=0, default_center_x=0.5, default_center_y=0.5, default_width=0.3,
                             default_height=0.3):
    """
    Create label files for images in the specified directory if they don't already exist.

    Args:
        image_root_dir (str): Root directory containing image files.
        class_id (int): Class ID for labeling (default is 0).
        default_center_x (float): Default center x-coordinate of the bounding box (default is 0.5).
        default_center_y (float): Default center y-coordinate of the bounding box (default is 0.5).
        default_width (float): Default width of the bounding box (default is 0.3).
        default_height (float): Default height of the bounding box (default is 0.3).
    """
    # Traverse all subdirectories and image files
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                label_file = image_path.replace('.png', '.txt')

                # Check if the label file already exists
                if not os.path.exists(label_file):
                    # Create label file and write default values
                    with open(label_file, 'w') as f:
                        f.write(f"{class_id} {default_center_x} {default_center_y} {default_width} {default_height}\n")
                    logging.info(f"Created label file: {label_file}")
                else:
                    logging.info(f"Label file already exists: {label_file}")


def main():
    """
    Main function to load environment variables and create labels for images.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Retrieve image root directory from environment variables
    image_root_directory = os.getenv('TRAINING_IMAGES_PATH')

    if not image_root_directory:
        logging.error("Error: TRAINING_IMAGES_PATH not set in environment variables.")
        return

    # Call the function to create label files
    create_labels_for_images(image_root_directory)


if __name__ == "__main__":
    main()
