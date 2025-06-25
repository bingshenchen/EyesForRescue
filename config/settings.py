# config/settings.py

"""
Configuration management for EyesForRescue project.
Centralizes all environment variable handling based on the actual project structure.
"""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv


class Settings:
    """
    Centralized configuration management for the EyesForRescue project.
    Loads and validates environment variables based on actual project structure.
    """

    def __init__(self):
        """Initialize settings by loading environment variables."""
        load_dotenv()
        self._validate_required_paths()

    # ====================================
    # Project Base Paths
    # ====================================
    @property
    def PROJECT_ROOT(self) -> Path:
        """Get project root directory."""
        root = os.getenv('PROJECT_ROOT')
        if not root:
            raise ValueError("PROJECT_ROOT environment variable is required")
        return Path(root)

    @property
    def DATA_DIR(self) -> Path:
        """Get main data directory."""
        data_dir = os.getenv('DATA_DIR')
        if data_dir:
            return Path(data_dir)
        return self.PROJECT_ROOT / 'data'

    @property
    def DATASETS_DIR(self) -> Path:
        """Get datasets directory."""
        datasets_dir = os.getenv('DATASETS_DIR')
        if datasets_dir:
            return Path(datasets_dir)
        return self.DATA_DIR / 'datasets'

    @property
    def MODELS_DIR(self) -> Path:
        """Get models directory."""
        models_dir = os.getenv('MODELS_DIR')
        if models_dir:
            return Path(models_dir)
        return self.DATA_DIR / 'models'

    @property
    def OUTPUT_DIR(self) -> Path:
        """Get output directory."""
        output_dir = os.getenv('OUTPUT_DIR')
        if output_dir:
            return Path(output_dir)
        return self.PROJECT_ROOT / 'outputs'

    @property
    def CACHE_DIR(self) -> Path:
        """Get cache directory."""
        cache_dir = os.getenv('CACHE_DIR')
        if cache_dir:
            return Path(cache_dir)
        return self.OUTPUT_DIR / 'cache'

    # ====================================
    # Model Paths
    # ====================================
    @property
    def YOLO_MODEL_PATH(self) -> Path:
        """Get YOLO model path."""
        path = os.getenv('YOLO_MODEL_PATH')
        if path:
            return Path(path)
        return self.MODELS_DIR / 'yolo' / 'best1.4.pt'

    @property
    def POSE_MODEL_PATH(self) -> Path:
        """Get pose detection model path."""
        path = os.getenv('POSE_MODEL_PATH')
        if path:
            return Path(path)
        return self.MODELS_DIR / 'yolo' / 'yolo11n-pose.pt'

    @property
    def CLASSIFIER_PATH(self) -> Path:
        """Get classifier model path."""
        path = os.getenv('CLASSIFIER_PATH')
        if path:
            return Path(path)
        return self.MODELS_DIR / 'classifier' / 'rf_classifier.pkl'

    @property
    def CLASSIFIER_MODEL_PATH(self) -> Path:
        """Get Keras classifier model path."""
        path = os.getenv('CLASSIFIER_MODEL_PATH')
        if path:
            return Path(path)
        return self.MODELS_DIR / 'classifier' / 'final_person_help_classifier.keras'

    # ====================================
    # Dataset Paths
    # ====================================
    @property
    def FALL_DETECTION_DATASET(self) -> Path:
        """Get fall detection dataset directory."""
        path = os.getenv('FALL_DETECTION_DATASET')
        if path:
            return Path(path)
        return self.DATASETS_DIR / 'fall_detection'

    @property
    def DATA_YAML_PATH(self) -> Path:
        """Get dataset YAML configuration path."""
        path = os.getenv('DATA_YAML_PATH')
        if path:
            return Path(path)
        return self.FALL_DETECTION_DATASET / 'dataset.yaml'

    @property
    def FRAMES_PATH(self) -> Path:
        """Get frames/images path."""
        path = os.getenv('FRAMES_PATH')
        if path:
            return Path(path)
        return self.FALL_DETECTION_DATASET / 'images'

    @property
    def LABELS_PATH(self) -> Path:
        """Get labels path."""
        path = os.getenv('LABELS_PATH')
        if path:
            return Path(path)
        return self.FALL_DETECTION_DATASET / 'labels'

    @property
    def CLASSIFIER_DATASET(self) -> Path:
        """Get classifier dataset directory."""
        path = os.getenv('CLASSIFIER_DATASET')
        if path:
            return Path(path)
        return self.DATASETS_DIR / 'classifier'

    @property
    def TRAINING_CLASSIFIER_PATH(self) -> Path:
        """Get training classifier dataset path."""
        path = os.getenv('TRAINING_CLASSIFIER_PATH')
        if path:
            return Path(path)
        return self.CLASSIFIER_DATASET / 'train'

    @property
    def TEST_CLASSIFIER_PATH(self) -> Path:
        """Get test classifier dataset path."""
        path = os.getenv('TEST_CLASSIFIER_PATH')
        if path:
            return Path(path)
        return self.CLASSIFIER_DATASET / 'test'

    @property
    def TEST_FINE_DIR(self) -> Path:
        """Get test fine directory."""
        path = os.getenv('TEST_FINE_DIR')
        if path:
            return Path(path)
        return self.CLASSIFIER_DATASET / 'fine'

    @property
    def TEST_NEEDHELP_DIR(self) -> Path:
        """Get test needhelp directory."""
        path = os.getenv('TEST_NEEDHELP_DIR')
        if path:
            return Path(path)
        return self.CLASSIFIER_DATASET / 'need_help'

    # ====================================
    # Output Directories
    # ====================================
    @property
    def TRAINING_RUNS_DIR(self) -> Path:
        """Get training runs directory."""
        path = os.getenv('TRAINING_RUNS_DIR')
        if path:
            return Path(path)
        return self.OUTPUT_DIR / 'training_runs'

    @property
    def EVALUATION_RESULTS_DIR(self) -> Path:
        """Get evaluation results directory."""
        path = os.getenv('EVALUATION_RESULTS_DIR')
        if path:
            return Path(path)
        return self.OUTPUT_DIR / 'evaluation_results'

    @property
    def REPORTS_DIR(self) -> Path:
        """Get reports directory."""
        path = os.getenv('REPORTS_DIR')
        if path:
            return Path(path)
        return self.OUTPUT_DIR / 'reports'

    @property
    def PROCESSED_VIDEOS_DIR(self) -> Path:
        """Get processed videos directory."""
        path = os.getenv('PROCESSED_VIDEOS_DIR')
        if path:
            return Path(path)
        return self.OUTPUT_DIR / 'processed_videos'

    @property
    def TEMP_DIR(self) -> Path:
        """Get temporary files directory."""
        path = os.getenv('TEMP_DIR')
        if path:
            return Path(path)
        return self.OUTPUT_DIR / 'temp'

    # ====================================
    # Detection Settings
    # ====================================
    @property
    def CLASSES(self) -> List[str]:
        """Get detection classes."""
        classes_str = os.getenv('CLASSES', 'person,falling_person,sitting_person,lying_person')
        return [cls.strip() for cls in classes_str.split(',')]

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        """Get confidence threshold for detection."""
        return float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

    @property
    def IOU_THRESHOLD(self) -> float:
        """Get IoU threshold for detection."""
        return float(os.getenv('IOU_THRESHOLD', '0.3'))

    @property
    def BATCH_SIZE(self) -> int:
        """Get batch size for processing."""
        return int(os.getenv('BATCH_SIZE', '16'))

    # ====================================
    # Performance Settings
    # ====================================
    @property
    def CACHE_ENABLED(self) -> bool:
        """Check if caching is enabled."""
        return os.getenv('CACHE_ENABLED', 'true').lower() == 'true'

    @property
    def USE_GPU(self) -> bool:
        """Check if GPU should be used."""
        return os.getenv('USE_GPU', 'true').lower() == 'true'

    @property
    def PERFORMANCE_MONITORING(self) -> bool:
        """Check if performance monitoring is enabled."""
        return os.getenv('PERFORMANCE_MONITORING', 'true').lower() == 'true'

    # ====================================
    # External Services
    # ====================================
    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        """Get OpenAI API key."""
        return os.getenv('OPENAIAPI_KEY')

    @property
    def MINIO_CONFIG(self) -> dict:
        """Get MinIO configuration."""
        return {
            'endpoint': os.getenv('MINIO_URI'),
            'access_key': os.getenv('MINIO_ROOT_USER'),
            'secret_key': os.getenv('MINIO_ROOT_PASSWORD'),
            'bucket': os.getenv('MINIO_BUCKET'),
            'secure': False
        }

    # ====================================
    # Algorithm Settings
    # ====================================
    @property
    def TRACKING_SETTINGS(self) -> dict:
        """Get tracking algorithm settings."""
        return {
            'max_miss': int(os.getenv('TRACKING_MAX_MISS', '5')),
            'min_hits': int(os.getenv('TRACKING_MIN_HITS', '3')),
            'iou_threshold': float(os.getenv('TRACKING_IOU_THRESHOLD', '0.3'))
        }

    @property
    def DANGER_SETTINGS(self) -> dict:
        """Get danger calculation settings."""
        return {
            'threshold': int(os.getenv('DANGER_THRESHOLD', '5')),
            'standup_threshold': int(os.getenv('STANDUP_THRESHOLD', '3')),
            'fall_duration_alert': int(os.getenv('FALL_DURATION_ALERT', '5'))
        }

    # ====================================
    # Test Data Paths
    # ====================================
    @property
    def VIDEO_DIR(self) -> Path:
        """Get video directory."""
        path = os.getenv('VIDEO_DIR')
        if path:
            return Path(path)
        return self.PROJECT_ROOT / 'videos'

    @property
    def TEST_VIDEO_PATH(self) -> Optional[Path]:
        """Get test video path."""
        path = os.getenv('TEST_VIDEO_PATH')
        if path:
            return Path(path)
        return None

    @property
    def TEST_IMAGE(self) -> Optional[Path]:
        """Get test image path."""
        path = os.getenv('TEST_IMAGE')
        if path:
            return Path(path)
        return None

    @property
    def GROUND_TRUTHS(self) -> List[int]:
        """Get ground truth values for evaluation."""
        ground_truths_str = os.getenv('GROUND_TRUTHS', '30,50')
        return [int(x.strip()) for x in ground_truths_str.split(',')]

    # ====================================
    # Training Settings
    # ====================================
    @property
    def EPOCHS(self) -> int:
        """Get number of training epochs."""
        return int(os.getenv('EPOCHS', '100'))

    @property
    def IMGSZ(self) -> int:
        """Get image size for training."""
        return int(os.getenv('IMGSZ', '736'))

    @property
    def BATCH_SIZE_TRAINING(self) -> int:
        """Get batch size for training."""
        return int(os.getenv('BATCH_SIZE_TRAINING', '16'))

    # ====================================
    # Debug Settings
    # ====================================
    @property
    def DEBUG_MODE(self) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    @property
    def LOG_LEVEL(self) -> str:
        """Get logging level."""
        return os.getenv('LOG_LEVEL', 'INFO')

    # ====================================
    # Validation Methods
    # ====================================
    def _validate_required_paths(self):
        """Validate that the project root exists."""
        if not self.PROJECT_ROOT.exists():
            raise ValueError(f"PROJECT_ROOT does not exist: {self.PROJECT_ROOT}")

    def validate_models(self) -> bool:
        """Validate that required models exist."""
        model_paths = [
            self.YOLO_MODEL_PATH,
            self.POSE_MODEL_PATH,
        ]

        missing_models = []
        for path in model_paths:
            if not path.exists():
                missing_models.append(str(path))

        if missing_models:
            print("Missing model files:")
            for model in missing_models:
                print(f"  - {model}")
            return False

        return True

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            self.DATASETS_DIR,
            self.MODELS_DIR,
            self.OUTPUT_DIR,
            self.TRAINING_RUNS_DIR,
            self.EVALUATION_RESULTS_DIR,
            self.REPORTS_DIR,
            self.PROCESSED_VIDEOS_DIR,
            self.TEMP_DIR,
            self.CACHE_DIR,
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✓ Directory ready: {directory}")
            except Exception as e:
                print(f"✗ Failed to create directory {directory}: {e}")

    def print_config_summary(self):
        """Print a summary of current configuration."""
        print("=== EyesForRescue Configuration ===")
        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"Data Directory: {self.DATA_DIR}")
        print(f"Datasets Directory: {self.DATASETS_DIR}")
        print(f"Models Directory: {self.MODELS_DIR}")
        print(f"Output Directory: {self.OUTPUT_DIR}")
        print(f"YOLO Model: {self.YOLO_MODEL_PATH}")
        print(f"Pose Model: {self.POSE_MODEL_PATH}")
        print(f"Classifier: {self.CLASSIFIER_PATH}")
        print(f"Classes: {', '.join(self.CLASSES)}")
        print(f"Confidence Threshold: {self.CONFIDENCE_THRESHOLD}")
        print(f"Cache Enabled: {self.CACHE_ENABLED}")
        print(f"GPU Enabled: {self.USE_GPU}")
        print(f"Debug Mode: {self.DEBUG_MODE}")
        print("=" * 35)

    def get_model_info(self):
        """Get information about available models."""
        models_info = {
            'yolo_model': {
                'path': self.YOLO_MODEL_PATH,
                'exists': self.YOLO_MODEL_PATH.exists()
            },
            'pose_model': {
                'path': self.POSE_MODEL_PATH,
                'exists': self.POSE_MODEL_PATH.exists()
            },
            'classifier': {
                'path': self.CLASSIFIER_PATH,
                'exists': self.CLASSIFIER_PATH.exists()
            }
        }
        return models_info


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


if __name__ == "__main__":
    # Test configuration
    config = get_settings()
    config.print_config_summary()

    # Show model information
    print("\n=== Model Information ===")
    models = config.get_model_info()
    for model_name, info in models.items():
        status = "✅" if info['exists'] else "❌"
        print(f"{status} {model_name}: {info['path']}")

    # Validate models
    print("\n=== Model Validation ===")
    if config.validate_models():
        print("✅ All required models found!")
    else:
        print("❌ Some models are missing!")

    # Create directories
    print("\n=== Directory Creation ===")
    config.create_directories()