# config/settings.py

"""
Configuration management for EyesForRescue project.
Centralizes all environment variable handling and provides validation.
Updated to match the new project structure.
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
    def MODELS_DIR(self) -> Path:
        """Get models directory."""
        return self.DATA_DIR / 'models'

    @property
    def CACHE_DIR(self) -> Path:
        """Get cache directory."""
        return self.DATA_DIR / 'cache'

    @property
    def OUTPUT_DIR(self) -> Path:
        """Get output directory."""
        return self.PROJECT_ROOT / 'outputs'

    @property
    def SRC_DIR(self) -> Path:
        """Get source code directory."""
        return self.PROJECT_ROOT / 'src'

    @property
    def CORE_DIR(self) -> Path:
        """Get core modules directory."""
        return self.SRC_DIR / 'core'

    # ====================================
    # Model Paths (New Structure)
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
        """Get deep learning classifier model path."""
        path = os.getenv('CLASSIFIER_MODEL_PATH')
        if path:
            return Path(path)
        return self.MODELS_DIR / 'classifier' / 'final_person_help_classifier.keras'

    # Legacy model paths for backward compatibility
    @property
    def LEGACY_YOLO_MODEL_PATH(self) -> Path:
        """Get legacy YOLO model path."""
        return self.PROJECT_ROOT / 'src' / 'train' / 'models' / 'best1.4.pt'

    @property
    def LEGACY_CLASSIFIER_PATH(self) -> Path:
        """Get legacy classifier path."""
        return self.PROJECT_ROOT / 'assets' / 'classifier' / 'classifier.pkl'

    # ====================================
    # Dataset Paths (Legacy - for backward compatibility)
    # ====================================
    @property
    def LEGACY_DATASETS_DIR(self) -> Path:
        """Get legacy datasets directory."""
        return self.PROJECT_ROOT / 'assets' / 'datasets'

    @property
    def FALL_DETECTION_DATASET(self) -> Path:
        """Get fall detection dataset directory."""
        path = os.getenv('FALL_DETECTION_DATASET')
        if path:
            return Path(path)
        return self.LEGACY_DATASETS_DIR / 'fall_detection'

    @property
    def CLASSIFIER_DATASET(self) -> Path:
        """Get classifier dataset directory."""
        path = os.getenv('CLASSIFIER_DATASET')
        if path:
            return Path(path)
        return self.LEGACY_DATASETS_DIR / 'classifier'

    @property
    def DATA_YAML_PATH(self) -> Path:
        """Get dataset YAML configuration path."""
        path = os.getenv('DATA_YAML_PATH')
        if path:
            return Path(path)
        return self.FALL_DETECTION_DATASET / 'dataset.yaml'

    # ====================================
    # Output Directories (Updated Structure)
    # ====================================
    @property
    def TRAINING_RUNS_DIR(self) -> Path:
        """Get training runs directory."""
        return self.OUTPUT_DIR / 'training_runs'

    @property
    def EVALUATION_RESULTS_DIR(self) -> Path:
        """Get evaluation results directory."""
        return self.OUTPUT_DIR / 'evaluation_results'

    @property
    def REPORTS_DIR(self) -> Path:
        """Get reports directory."""
        return self.OUTPUT_DIR / 'reports'

    @property
    def PROCESSED_VIDEOS_DIR(self) -> Path:
        """Get processed videos directory."""
        return self.OUTPUT_DIR / 'processed_videos'

    @property
    def TEMP_DIR(self) -> Path:
        """Get temporary files directory."""
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
    # Test Settings
    # ====================================
    @property
    def TEST_VIDEO_PATH(self) -> Optional[Path]:
        """Get test video path."""
        path = os.getenv('TEST_VIDEO_PATH')
        return Path(path) if path else None

    @property
    def VIDEO_DIR(self) -> Optional[Path]:
        """Get video directory."""
        path = os.getenv('VIDEO_DIR')
        return Path(path) if path else None

    @property
    def IMAGE_DIRECTORY(self) -> Optional[Path]:
        """Get image directory."""
        path = os.getenv('IMAGE_DIRECTORY')
        return Path(path) if path else None

    # ====================================
    # Training Settings
    # ====================================
    @property
    def TRAINING_SETTINGS(self) -> dict:
        """Get training settings."""
        return {
            'epochs': int(os.getenv('EPOCHS', '100')),
            'imgsz': int(os.getenv('IMGSZ', '736')),
            'batch_size': int(os.getenv('BATCH_SIZE_TRAINING', '16')),
            'train_model': os.getenv('TRAIN_MODEL', '0') == '1',
            'run_tests': os.getenv('RUN_TESTS', '0') == '1'
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

    @property
    def VERBOSE_LOGGING(self) -> bool:
        """Check if verbose logging is enabled."""
        return os.getenv('VERBOSE_LOGGING', 'false').lower() == 'true'

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
        ]

        # Check if models exist in new location first, then fall back to legacy
        missing_models = []
        for path in model_paths:
            if not path.exists():
                # Try legacy path
                if 'yolo' in str(path).lower():
                    legacy_path = self.LEGACY_YOLO_MODEL_PATH
                    if legacy_path.exists():
                        print(f"Warning: Model found in legacy location: {legacy_path}")
                        print(f"Consider moving to: {path}")
                        continue

                missing_models.append(str(path))

        # Check classifier (try new location first, then legacy)
        if not self.CLASSIFIER_PATH.exists() and not self.LEGACY_CLASSIFIER_PATH.exists():
            missing_models.append(str(self.CLASSIFIER_PATH))

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
            self.MODELS_DIR,
            self.MODELS_DIR / 'yolo',
            self.MODELS_DIR / 'classifier',
            self.CACHE_DIR,
            self.OUTPUT_DIR,
            self.TRAINING_RUNS_DIR,
            self.EVALUATION_RESULTS_DIR,
            self.REPORTS_DIR,
            self.PROCESSED_VIDEOS_DIR,
            self.TEMP_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory ready: {directory}")

    def print_config_summary(self):
        """Print a summary of current configuration."""
        print("=== EyesForRescue Configuration ===")
        print(f"Project Root: {self.PROJECT_ROOT}")
        print(f"Data Directory: {self.DATA_DIR}")
        print(f"Models Directory: {self.MODELS_DIR}")
        print(f"Output Directory: {self.OUTPUT_DIR}")
        print()
        print("Model Paths:")
        print(f"  YOLO Model: {self.YOLO_MODEL_PATH}")
        print(f"  Pose Model: {self.POSE_MODEL_PATH}")
        print(f"  Classifier: {self.CLASSIFIER_PATH}")
        print()
        print("Dataset Paths:")
        print(f"  Fall Detection: {self.FALL_DETECTION_DATASET}")
        print(f"  Classifier: {self.CLASSIFIER_DATASET}")
        print(f"  Data YAML: {self.DATA_YAML_PATH}")
        print()
        print("Settings:")
        print(f"  Classes: {', '.join(self.CLASSES)}")
        print(f"  Confidence Threshold: {self.CONFIDENCE_THRESHOLD}")
        print(f"  Cache Enabled: {self.CACHE_ENABLED}")
        print(f"  GPU Enabled: {self.USE_GPU}")
        print(f"  Debug Mode: {self.DEBUG_MODE}")
        print("=" * 35)

    def get_model_path_with_fallback(self, model_type: str) -> Path:
        """
        Get model path with fallback to legacy location.

        Args:
            model_type: 'yolo', 'pose', or 'classifier'

        Returns:
            Path to the model file
        """
        if model_type == 'yolo':
            if self.YOLO_MODEL_PATH.exists():
                return self.YOLO_MODEL_PATH
            elif self.LEGACY_YOLO_MODEL_PATH.exists():
                return self.LEGACY_YOLO_MODEL_PATH
            else:
                return self.YOLO_MODEL_PATH  # Return expected path even if it doesn't exist

        elif model_type == 'pose':
            return self.POSE_MODEL_PATH

        elif model_type == 'classifier':
            if self.CLASSIFIER_PATH.exists():
                return self.CLASSIFIER_PATH
            elif self.LEGACY_CLASSIFIER_PATH.exists():
                return self.LEGACY_CLASSIFIER_PATH
            else:
                return self.CLASSIFIER_PATH

        else:
            raise ValueError(f"Unknown model type: {model_type}")


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