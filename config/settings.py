"""
Configuration management for EyesForRescue project.
Centralizes all environment variable handling and provides validation.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv


class Settings:
    """
    Centralized configuration management for the EyesForRescue project.
    Loads and validates all environment variables.
    """

    def __init__(self):
        """Initialize settings by loading environment variables."""
        load_dotenv()
        self._validate_required_paths()

    # ====================================
    # Project Paths
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
        return self.PROJECT_ROOT / 'data'

    @property
    def ASSETS_DIR(self) -> Path:
        """Get assets directory (legacy)."""
        return self.PROJECT_ROOT / 'assets'

    @property
    def OUTPUT_DIR(self) -> Path:
        """Get output directory."""
        return self.PROJECT_ROOT / 'outputs'

    # ====================================
    # Model Paths
    # ====================================
    @property
    def YOLO_MODEL_PATH(self) -> Path:
        """Get YOLO model path."""
        path = os.getenv('YOLO_MODEL_PATH')
        if path:
            return Path(path)
        return self.PROJECT_ROOT / 'src' / 'train' / 'models' / 'best1.4.pt'

    @property
    def POSE_MODEL_PATH(self) -> Path:
        """Get pose detection model path."""
        path = os.getenv('POSE_MODEL_PATH')
        if path:
            return Path(path)
        return self.PROJECT_ROOT / 'src' / 'train' / 'models' / 'yolo11n-pose.pt'

    @property
    def CLASSIFIER_PATH(self) -> Path:
        """Get classifier model path."""
        path = os.getenv('CLASSIFIER_PATH')
        if path:
            return Path(path)
        return self.ASSETS_DIR / 'classifier' / 'classifier.pkl'

    # ====================================
    # Dataset Paths
    # ====================================
    @property
    def FALL_DETECTION_DATASET(self) -> Path:
        """Get fall detection dataset directory."""
        return self.ASSETS_DIR / 'datasets' / 'fall_detection'

    @property
    def CLASSIFIER_DATASET(self) -> Path:
        """Get classifier dataset directory."""
        return self.ASSETS_DIR / 'datasets' / 'classifier'

    @property
    def DATA_YAML_PATH(self) -> Path:
        """Get dataset YAML configuration path."""
        return self.FALL_DETECTION_DATASET / 'dataset.yaml'

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
        """Validate that all required paths exist."""
        required_paths = [
            self.PROJECT_ROOT,
        ]

        for path in required_paths:
            if not path.exists():
                print(f"Warning: Required path does not exist: {path}")

    def validate_models(self) -> bool:
        """Validate that all required models exist."""
        model_paths = [
            self.YOLO_MODEL_PATH,
            self.POSE_MODEL_PATH,
            self.CLASSIFIER_PATH
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
            self.OUTPUT_DIR,
            self.OUTPUT_DIR / 'training_runs',
            self.OUTPUT_DIR / 'evaluation_results',
            self.OUTPUT_DIR / 'reports',
            self.OUTPUT_DIR / 'processed_videos',
            self.DATA_DIR / 'cache',
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory ready: {directory}")

    def print_config_summary(self):
        """Print a summary of current configuration."""
        print("=== EyesForRescue Configuration ===")
        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"YOLO Model: {self.YOLO_MODEL_PATH}")
        print(f"Pose Model: {self.POSE_MODEL_PATH}")
        print(f"Classifier: {self.CLASSIFIER_PATH}")
        print(f"Classes: {', '.join(self.CLASSES)}")
        print(f"Confidence Threshold: {self.CONFIDENCE_THRESHOLD}")
        print(f"Cache Enabled: {self.CACHE_ENABLED}")
        print(f"GPU Enabled: {self.USE_GPU}")
        print(f"Debug Mode: {self.DEBUG_MODE}")
        print("=" * 35)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


if __name__ == "__main__":
    # Test configuration
    config = get_settings()
    config.print_config_summary()

    # Validate models
    if config.validate_models():
        print("✅ All models found!")
    else:
        print("❌ Some models are missing!")

    # Create directories
    config.create_directories()