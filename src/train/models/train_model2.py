# src/train/models/train_model2.py

import os
import logging
import yaml
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def configure_logging():
    """
    Configure logging settings for training process monitoring.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("yolo_training")


def load_environment_variables():
    """
    Load environment variables from .env file and validate required variables.

    Returns:
        dict: Dictionary containing all necessary paths and configurations
    """
    load_dotenv()

    config = {
        "model_path": os.getenv("YOLO_MODEL_PATH"),
        "data_path": os.getenv("DATA_YAML_PATH"),
        "output_dir": os.getenv("OUTPUT_DIRECTORY", "runs/train"),
        "device": os.getenv("DEVICE", "0" if torch.cuda.is_available() else "cpu"),
    }

    # Validate required environment variables
    missing_vars = [k for k, v in config.items() if v is None]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    return config


def prepare_data_config(data_path, train_transform_config=None):
    """
    Update the data YAML configuration with custom transformations if needed.

    Args:
        data_path (str): Path to the data YAML file
        train_transform_config (dict, optional): Custom training transforms configuration

    Returns:
        str: Path to the updated data YAML file
    """
    if train_transform_config is None:
        return data_path

    # Load existing data configuration
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # Add custom transforms configuration
    data_config['train_transforms'] = train_transform_config

    # Create a new temporary YAML file
    temp_yaml_path = f"temp_data_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(data_config, f)

    return temp_yaml_path


def get_optimal_training_settings(model_size='s', task_type='fall_detection'):
    """
    Get optimized training settings based on model size and task type.

    Args:
        model_size (str): Size of the model (n, s, m, l, x)
        task_type (str): Type of task (fall_detection, pose, etc.)

    Returns:
        dict: Dictionary containing optimized training parameters
    """
    # Base settings common for all model sizes
    settings = {
        'optimizer': 'AdamW',  # Default optimizer
        'weight_decay': 0.0005,  # Default weight decay
        'momentum': 0.937,  # Default momentum
        'cos_lr': True,  # Use cosine learning rate scheduler
        'patience': 15,  # Early stopping patience
        'save_period': 5,  # Save checkpoint every X epochs
    }

    # Model size specific settings
    size_settings = {
        'n': {  # Nano
            'lr0': 0.01,
            'lrf': 0.01,
            'batch': 64,
            'imgsz': 640,
            'warmup_epochs': 3
        },
        's': {  # Small
            'lr0': 0.01,
            'lrf': 0.01,
            'batch': 32,
            'imgsz': 640,
            'warmup_epochs': 3
        },
        'm': {  # Medium
            'lr0': 0.01,
            'lrf': 0.01,
            'batch': 16,
            'imgsz': 640,
            'warmup_epochs': 3
        },
        'l': {  # Large
            'lr0': 0.01,
            'lrf': 0.01,
            'batch': 8,
            'imgsz': 640,
            'warmup_epochs': 3
        },
        'x': {  # Extra Large
            'lr0': 0.005,
            'lrf': 0.005,
            'batch': 4,
            'imgsz': 640,
            'warmup_epochs': 5
        }
    }

    # Task-specific settings
    task_settings = {
        'fall_detection': {
            'box': 7.5,  # Increase box loss weight for better localization
            'cls': 0.5,  # Class loss weight
            'dfl': 1.5,  # Distribution focal loss weight
            'hsv_h': 0.015,  # Hue augmentation
            'hsv_s': 0.7,  # Saturation augmentation
            'hsv_v': 0.4,  # Value augmentation
            'degrees': 10.0,  # Rotation augmentation (limited for human pose)
            'translate': 0.1,  # Translation augmentation
            'scale': 0.5,  # Scale augmentation
            'fliplr': 0.5,  # Horizontal flip augmentation
            'mosaic': 1.0,  # Mosaic augmentation
            'mixup': 0.15  # Mixup augmentation
        },
        'pose': {
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,  # Increase pose loss weight
            'hsv_h': 0.01,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'degrees': 5.0,  # Limited rotation for pose
            'translate': 0.1,
            'scale': 0.4,
            'fliplr': 0.5,
            'mosaic': 0.8,
            'mixup': 0.1
        }
    }

    # Combine settings
    settings.update(size_settings.get(model_size, size_settings['s']))
    settings.update(task_settings.get(task_type, task_settings['fall_detection']))

    return settings


def train_yolo_model(
        model_path,
        data_path,
        output_dir,
        logger,
        device='0',
        model_size='s',
        task_type='fall_detection',
        epochs=100,
        custom_settings=None
):
    """
    Train a YOLO model with optimized settings for fall detection.

    Args:
        model_path (str): Path to the base YOLO model
        data_path (str): Path to the dataset YAML file
        output_dir (str): Directory where training results will be saved
        logger (logging.Logger): Logger for training process
        device (str): Device to use for training (e.g., '0', 'cpu')
        model_size (str): Size of the model (n, s, m, l, x)
        task_type (str): Type of task (fall_detection, pose, etc.)
        epochs (int): Number of training epochs
        custom_settings (dict, optional): Custom training settings to override defaults

    Returns:
        YOLO: Trained YOLO model
    """
    logger.info(f"Loading YOLO model from {model_path}")

    # Load the YOLO model
    try:
        model = YOLO(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Get optimized training settings
    settings = get_optimal_training_settings(model_size, task_type)

    # Override with custom settings if provided
    if custom_settings:
        settings.update(custom_settings)

    # Update data configuration with augmentations
    augmentation_config = {k: v for k, v in settings.items() if k in [
        'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
        'fliplr', 'flipud', 'mosaic', 'mixup', 'copy_paste', 'perspective'
    ]}

    # Remove augmentation settings from training parameters
    training_params = {k: v for k, v in settings.items() if k not in augmentation_config}

    # Create run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{task_type}_{model_size}_{timestamp}"

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set hyperparameter logging
    logger.info("=== Training Settings ===")
    for k, v in settings.items():
        logger.info(f"{k}: {v}")

    logger.info(f"Starting training with {epochs} epochs")

    # Start training
    try:
        results = model.train(
            data=data_path,
            epochs=epochs,
            project=output_dir,
            name=run_name,
            device=device,
            exist_ok=True,
            **training_params,
            **augmentation_config
        )
        logger.info("Training completed successfully")
        return model, results
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def evaluate_model(model, data_path, logger, class_names=None):
    """
    Evaluate trained model on validation dataset with detailed metrics.

    Args:
        model (YOLO): Trained YOLO model
        data_path (str): Path to the dataset YAML file
        logger (logging.Logger): Logger for evaluation process
        class_names (list, optional): List of class names for reporting

    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating model on {data_path}")

    try:
        results = model.val(data=data_path, verbose=True)

        # Extract and log metrics
        logger.info("=== Evaluation Results ===")
        metrics = {}

        # Overall metrics
        if hasattr(results, 'box') and hasattr(results.box, 'map'):
            mAP50 = float(results.box.map50)
            mAP50_95 = float(results.box.map)
            logger.info(f"mAP50: {mAP50:.4f}")
            logger.info(f"mAP50-95: {mAP50_95:.4f}")
            metrics['mAP50'] = mAP50
            metrics['mAP50-95'] = mAP50_95

        # Per-class metrics if available
        if hasattr(results, 'box') and hasattr(results.box, 'cls_map50'):
            cls_map50 = results.box.cls_map50.cpu().numpy()
            cls_precision = results.box.p.cpu().numpy()
            cls_recall = results.box.r.cpu().numpy()

            if class_names is None:
                if hasattr(model, 'names'):
                    class_names = list(model.names.values())
                else:
                    class_names = [f"Class {i}" for i in range(len(cls_map50))]

            for i, name in enumerate(class_names):
                if i < len(cls_map50):
                    logger.info(f"Class: {name}, Precision: {cls_precision[i]:.4f}, "
                                f"Recall: {cls_recall[i]:.4f}, mAP50: {cls_map50[i]:.4f}")
                    metrics[f"{name}_precision"] = float(cls_precision[i])
                    metrics[f"{name}_recall"] = float(cls_recall[i])
                    metrics[f"{name}_mAP50"] = float(cls_map50[i])

        return metrics

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def export_optimized_model(model, output_dir, format='onnx', logger=None):
    """
    Export the trained model for efficient inference.

    Args:
        model (YOLO): Trained YOLO model
        output_dir (str): Directory to save exported model
        format (str): Export format (onnx, torchscript, openvino, etc.)
        logger (logging.Logger, optional): Logger for export process

    Returns:
        str: Path to the exported model
    """
    if logger:
        logger.info(f"Exporting model to {format} format")

    try:
        export_path = model.export(
            format=format,
            half=True,  # FP16 for faster inference
            simplify=True,  # Simplify ONNX model
            dynamic=True,  # Dynamic batch size
            optimize=True  # Optimize for inference
        )

        if logger:
            logger.info(f"Model successfully exported to {export_path}")

        return export_path

    except Exception as e:
        if logger:
            logger.error(f"Failed to export model: {e}")
        raise


def plot_training_results(results, output_dir):
    """
    Plot training results and save to output directory.

    Args:
        results: Training results object from YOLO training
        output_dir (str): Directory to save plots
    """
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get results from CSV
    results_file = os.path.join(results.save_dir, "results.csv")
    if not os.path.exists(results_file):
        return

    results_df = pd.read_csv(results_file)

    # Plot metrics
    metrics = [
        ("loss", "training_loss.png"),
        ("box_loss", "box_loss.png"),
        ("cls_loss", "cls_loss.png"),
        ("dfl_loss", "dfl_loss.png"),
        ("precision", "precision.png"),
        ("recall", "recall.png"),
        ("mAP50(B)", "map50.png"),
        ("mAP50-95(B)", "map50_95.png")
    ]

    for metric, filename in metrics:
        if metric in results_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(results_df["epoch"], results_df[metric])
            plt.title(f"Training {metric}")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, filename))
            plt.close()


def main():
    """
    Main function to execute the YOLO training pipeline.
    """
    # Configure logging
    logger = configure_logging()
    logger.info("Starting YOLO training pipeline for fall detection")

    try:
        # Load environment configuration
        config = load_environment_variables()
        logger.info(f"Configuration loaded: {config}")

        # Set training parameters
        epochs = 100
        model_size = 's'  # Small model for balance of speed and accuracy
        task_type = 'fall_detection'

        # Train the model
        model, results = train_yolo_model(
            model_path=config["model_path"],
            data_path=config["data_path"],
            output_dir=config["output_dir"],
            logger=logger,
            device=config["device"],
            model_size=model_size,
            task_type=task_type,
            epochs=epochs
        )

        # Plot training results
        plot_training_results(results, config["output_dir"])

        # Evaluate the model
        evaluate_model(model, config["data_path"], logger)

        # Export model for inference
        export_optimized_model(model, config["output_dir"], format='onnx', logger=logger)

        logger.info("YOLO training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()