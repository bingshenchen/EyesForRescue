import csv
import os
import logging
from dotenv import load_dotenv
from ultralytics import YOLO


def train_yolo_model(model_dir_path, data_path, output_dir, epochs=30, imgsz=640, batch=8, mosaic=1.0, mixup=0.2):
    """
    Function to train3 the YOLO model with specified parameters.

    Args:
        model_dir_path (str): Path to the YOLO model (.pt file).
        data_path (str): Path to the dataset configuration file (.yaml).
        output_dir (str): Directory where training results will be saved.
        epochs (int): Number of epochs for training.
        imgsz (int): Image size for training.
        batch (int): Batch size for training.
        mosaic (float): Mosaic augmentation parameter.
        mixup (float): Mixup augmentation parameter.
    """
    logging.info(f"Loading YOLO model from {model_dir_path}")

    # Load the pre-trained YOLO model
    model = YOLO(model_dir_path)

    logging.info(f"Starting training with dataset: {data_path}")

    # Start training the model with custom output directory and run name
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        mosaic=mosaic,
        mixup=mixup,
        project=output_dir
    )

    logging.info("Training completed.")


# def evaluate_model(model, data_path, results_output_path):
#     """
#     Function to evaluate the YOLO model on a dataset and save results.
#
#     Args:
#         model (YOLO): The trained YOLO model.
#         data_path (str): Path to the dataset.
#         results_output_path (str): Path to save the results CSV file.
#     """
#     logging.info(f"Evaluating model on dataset: {data_path}")
#     results = model.val(data=data_path)
#
#     # Extract evaluation metrics
#     metrics = {
#         'Class': list(model.names.values()),
#         'P': results.box.P.tolist(),
#         'R': results.box.R.tolist(),
#         'mAP50': results.box.map.tolist(),
#         'mAP50-95': results.box.maps.tolist()
#     }
#
#     # Save the evaluation results to a CSV file
#     with open(results_output_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Class', 'P', 'R', 'mAP50', 'mAP50-95'])
#         for i in range(len(metrics['Class'])):
#             writer.writerow([
#                 metrics['Class'][i],
#                 metrics['P'][i],
#                 metrics['R'][i],
#                 metrics['mAP50'][i],
#                 metrics['mAP50-95'][i]
#             ])
#
#     logging.info(f"Results saved to {results_output_path}")
#
#
# def main():
#     # Load environment variables from .env file
#     load_dotenv()
#
#     # Configure logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
#     # Retrieve paths from environment variables
#     model_path = os.getenv('YOLO_MODEL_PATH')
#     data_yaml_path = os.getenv('DATA_YAML_PATH')
#     output_dir_path = os.getenv('OUTPUT_DIRECTORY')
#
#     if not model_path or not data_yaml_path:
#         logging.error("Error: YOLO_MODEL_PATH or DATA_YAML_PATH not set in environment variables.")
#         return
#
#     # Train the model
#     train_yolo_model(model_path, data_yaml_path, output_dir_path, epochs=30, imgsz=640, batch=8, mosaic=1.0, mixup=0.2)
#     #    train_yolo_model(model_path, data_yaml_path, output_dir_path, epochs=20, imgsz=512, batch=16, mosaic=0.5, mixup=0.1)
#
#     # Load the trained model for evaluation
#     model = YOLO(model_path)
#
#     # Evaluate the model and save results
#     results_output_path = os.path.join(output_dir_path, "evaluation_results.csv")
#     evaluate_model(model, data_yaml_path, results_output_path)


def evaluate_model(model, data_path):
    """
    Function to evaluate the YOLO model on a dataset and print results.

    Args:
        model (YOLO): The trained YOLO model.
        data_path (str): Path to the dataset.
    """
    logging.info(f"Evaluating model on dataset: {data_path}")
    results = model.val(data=data_path)

    print("\nEvaluation Metrics:")
    # Assuming the model has `names` attribute to display class names
    if hasattr(model, 'names'):
        print(f"Classes: {list(model.names.values())}")
    else:
        print("Class names are not available in the model.")

    # Accessing results metrics safely
    try:
        print(f"Precision (P): {results.box.p.tolist() if hasattr(results.box, 'p') else 'N/A'}")
        print(f"Recall (R): {results.box.r.tolist() if hasattr(results.box, 'r') else 'N/A'}")
        print(f"mAP@50: {results.box.map50 if hasattr(results.box, 'map50') else 'N/A'}")
        print(f"mAP@50-95: {results.box.map.tolist() if hasattr(results.box, 'map') else 'N/A'}")
    except AttributeError as e:
        print(f"Error accessing metrics: {e}")

    logging.info("Evaluation completed.")


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Retrieve paths from environment variables
    model_path = os.getenv('YOLO_MODEL_PATH')
    data_yaml_path = os.getenv('DATA_YAML_PATH')

    if not model_path or not data_yaml_path:
        logging.error("Error: YOLO_MODEL_PATH or DATA_YAML_PATH not set in environment variables.")
        return

    model = YOLO(model_path)
    evaluate_model(model, data_yaml_path)


if __name__ == "__main__":
    main()
