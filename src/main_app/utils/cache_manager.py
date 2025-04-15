import os
import pickle
import hashlib
import time
from pathlib import Path


class DetectionCache:
    """
    A cache manager for storing and retrieving YOLO detection results.
    """

    def __init__(self, cache_dir=None):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files. If None, a default directory will be used.
        """
        if cache_dir is None:
            # Default cache directory in project root/cache
            self.cache_dir = Path(os.getenv('PROJECT_ROOT', '.')) / 'cache' / 'yolo_detections'
        else:
            self.cache_dir = Path(cache_dir)

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory set to: {self.cache_dir}")

    def generate_cache_key(self, video_path, model_name):
        """
        Generate a unique cache key based on video path and model name.

        Args:
            video_path: Path to the video file
            model_name: Name or path of the YOLO model used

        Returns:
            A unique hash string to identify this video+model combination
        """
        # Get video file metadata
        video_path = Path(video_path)
        try:
            video_size = video_path.stat().st_size if video_path.is_file() else 0
            video_mtime = video_path.stat().st_mtime if video_path.is_file() else 0
        except Exception as e:
            print(f"Warning: Could not get file stats for {video_path}: {e}")
            video_size = 0
            video_mtime = 0

        # Create a unique identifier based on video properties and model
        unique_id = f"{video_path.name}_{video_size}_{video_mtime}_{model_name}"
        return hashlib.md5(unique_id.encode()).hexdigest()

    def get_cache_path(self, video_path, model_name):
        """
        Get the cache file path for a specific video and model.
        """
        cache_key = self.generate_cache_key(video_path, model_name)
        return self.cache_dir / f"{cache_key}.pkl"

    def cache_exists(self, video_path, model_name):
        """
        Check if cache exists for a video and model combination.
        """
        cache_path = self.get_cache_path(video_path, model_name)
        return cache_path.exists()

    def save_detections(self, video_path, model_name, detections):
        """
        Save detection results to cache.

        Args:
            video_path: Path to the video file
            model_name: Name or path of the YOLO model
            detections: Dictionary with frame indices as keys and detection results as values
        """
        try:
            cache_path = self.get_cache_path(video_path, model_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(detections, f)

            # Print cache information
            cache_size = os.path.getsize(cache_path) / (1024 * 1024)  # Convert to MB
            print(f"Cache saved: {cache_path} ({cache_size:.2f} MB)")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")

    def load_detections(self, video_path, model_name):
        """
        Load detection results from cache.

        Args:
            video_path: Path to the video file
            model_name: Name or path of the YOLO model

        Returns:
            Dictionary with frame indices as keys and detection results as values
        """
        cache_path = self.get_cache_path(video_path, model_name)

        if not cache_path.exists():
            return None

        try:
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                detections = pickle.load(f)

            load_time = time.time() - start_time
            print(f"Cache loaded from {cache_path} in {load_time:.2f} seconds")
            return detections
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return None