import os
import cv2  # OpenCV for image processing
import torch
import logging
from ultralytics import YOLO  # Ultralytics YOLO for model loading
from dotenv import load_dotenv


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(model_dir_path, use_cuda=True):
    """
    Load the YOLO model.

    Args:
        model_dir_path (str): Path to the YOLO model file.
        use_cuda (bool): Whether to use CUDA (GPU) if available.

    Returns:
        YOLO: Loaded YOLO model.
    """
    model = YOLO(model_dir_path)

    # Check if CUDA is available and move model to GPU if applicable
    if use_cuda and torch.cuda.is_available():
        model.to('cuda')
        logging.info("Model loaded on GPU.")
    else:
        logging.info("Model loaded on CPU.")

    return model


def process_image(image_path, model, output_dir, conf_threshold=0.25):
    """
    Process an image to detect people using YOLO and save the result with bounding boxes.

    Args:
        image_path (str): Path to the input image.
        model (YOLO): The loaded YOLO model.
        output_dir (str): Directory to save the processed images.
        conf_threshold (float): Confidence threshold for YOLO predictions.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to load image: {image_path}")
        return

    # Run YOLO detection on the image
    results = model.predict(image, conf=conf_threshold)

    # Process detection results for people
    people = [det for det in results[0].boxes.data if int(det[5]) == 0]  # Assuming '0' is the class ID for 'person'

    # Draw bounding boxes on the image
    for person in people:
        x1, y1, x2, y2, conf, cls = map(int, person[:6])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
        cv2.putText(image, f'Person: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the resulting image with detections
    output_path = os.path.join(output_dir, f'detected_{os.path.basename(image_path)}')
    cv2.imwrite(output_path, image)
    logging.info(f"Processed {image_path}, found {len(people)} people. Result saved to {output_path}")


def process_images_in_directory(image_dir, output_dir, model, conf_threshold=0.25):
    """
    Iterate over all images in the specified directory and process each one.

    Args:
        image_dir (str): Path to the directory containing images.
        output_dir (str): Directory to save the processed images.
        model (YOLO): The loaded YOLO model.
        conf_threshold (float): Confidence threshold for YOLO predictions.
    """
    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            process_image(image_path, model, output_dir, conf_threshold)


def main():
    """
    Main function to load environment variables, model, and process images in a directory.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Setup logging
    setup_logging()

    # Get paths from environment variables
    image_directory = os.getenv('IMAGE_DIRECTORY')
    output_directory = os.getenv('OUTPUT_DIRECTORY')
    model_path = os.getenv('yolov11n')

    if not image_directory or not output_directory or not model_path:
        logging.error("Error: Missing environment variables for paths.")
        return

    # Load the YOLO model
    yolo_model = load_model(model_path)

    # Process all images in the directory
    process_images_in_directory(image_directory, output_directory, yolo_model)


if __name__ == "__main__":
    main()
