# src/core/analysis/gpt_analyzer_async.py

import base64
import json
import asyncio
import threading
import queue
import mimetypes
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union

import cv2
import numpy as np
from openai import AsyncOpenAI
import aiohttp

from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration settings
settings = get_settings()

# Initialize async OpenAI client
async_client = None
if settings.OPENAI_API_KEY:
    async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
else:
    logger.warning("OpenAI API key not found. GPT analysis will not be available.")


class AsyncGPTAnalyzer:
    """Asynchronous GPT analyzer to prevent video freezing."""

    def __init__(self):
        self.analysis_queue = queue.Queue()
        self.results_cache = {}
        self.is_running = False
        self.worker_thread = None
        self.loop = None

    def start(self):
        """Start the async analyzer worker thread."""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.worker_thread.start()
            logger.info("Async GPT analyzer started")

    def stop(self):
        """Stop the async analyzer worker thread."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        logger.info("Async GPT analyzer stopped")

    def _run_async_loop(self):
        """Run the async event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._process_queue())

    async def _process_queue(self):
        """Process analysis requests from the queue."""
        while self.is_running:
            try:
                # Check for new analysis requests
                if not self.analysis_queue.empty():
                    request = self.analysis_queue.get_nowait()
                    await self._analyze_async(request)
                else:
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Error in async processing: {e}")

    async def _analyze_async(self, request: Dict[str, Any]):
        """Perform async GPT analysis."""
        person_id = request['person_id']
        image = request['image']
        timestamp = request['timestamp']

        try:
            # Encode image
            mime_type, base64_image = self._encode_image(image)

            # Prepare prompt focused on emergency detection
            prompt = """
            Analyze this person in the image and determine if they need help.
            Return ONLY a JSON object with this structure:
            {
                "needs_help": true or false,
                "confidence": 0.0 to 1.0,
                "status": ["falling", "sitting", "laying", "standing", "walking"],
                "emergency_indicators": ["unconscious", "injured", "distressed", "normal"],
                "posture": "upright", "bent", "prone", "supine",
                "movement": "static" or "moving",
                "face_visible": true or false,
                "age_group": "child", "adult", "elderly",
                "environment": "indoor", "outdoor", "road", "home"
            }
            Focus on safety and emergency detection. Be concise.
            """

            # Call OpenAI API asynchronously
            response = await async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                        ],
                    }
                ],
                max_tokens=200,
                temperature=0.1  # Very low for consistent results
            )

            # Parse response
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()

            analysis = json.loads(content)

            # Store result in cache
            result = {
                'person_id': person_id,
                'timestamp': timestamp,
                'analysis': analysis,
                'processed_at': datetime.now().isoformat()
            }

            self.results_cache[person_id] = result
            logger.info(f"GPT analysis completed for person {person_id}: needs_help={analysis.get('needs_help')}")

        except Exception as e:
            logger.error(f"GPT analysis failed for person {person_id}: {e}")
            # Store error result
            self.results_cache[person_id] = {
                'person_id': person_id,
                'timestamp': timestamp,
                'analysis': {
                    'needs_help': False,
                    'confidence': 0.0,
                    'error': str(e)
                },
                'processed_at': datetime.now().isoformat()
            }

    def _encode_image(self, image: Union[np.ndarray, str, Path]) -> tuple:
        """Encode image to base64."""
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise ValueError(f"Image file not found: {image_path}")

            mime_type, _ = mimetypes.guess_type(str(image_path))
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = "image/jpeg"

            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')

        elif isinstance(image, np.ndarray):
            # OpenCV frame - encode as JPEG
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            base64_image = base64.b64encode(buffer).decode('utf-8')
            mime_type = "image/jpeg"
        else:
            raise ValueError("Invalid image type")

        return mime_type, base64_image

    def request_analysis(self, person_id: int, image: np.ndarray, danger_value: float) -> None:
        """
        Request async analysis for a person if danger value exceeds threshold.

        Args:
            person_id: Unique person identifier
            image: Person bounding box image (cropped from frame)
            danger_value: Current danger value for the person
        """
        # Only analyze if danger value exceeds threshold
        if danger_value > 1.0:
            request = {
                'person_id': person_id,
                'image': image,
                'timestamp': datetime.now().isoformat(),
                'danger_value': danger_value
            }

            # Add to queue if not already processing this person recently
            if person_id not in self.results_cache or \
                    self._is_result_stale(self.results_cache[person_id]):
                self.analysis_queue.put(request)
                logger.info(f"Analysis requested for person {person_id} (danger={danger_value:.2f})")

    def get_result(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result for a person.

        Returns:
            Analysis result or None if not available
        """
        return self.results_cache.get(person_id)

    def _is_result_stale(self, result: Dict[str, Any], max_age_seconds: int = 30) -> bool:
        """Check if a cached result is too old."""
        if 'processed_at' not in result:
            return True

        processed_time = datetime.fromisoformat(result['processed_at'])
        age = (datetime.now() - processed_time).total_seconds()
        return age > max_age_seconds


# Global analyzer instance
_analyzer_instance = None


def get_analyzer() -> AsyncGPTAnalyzer:
    """Get or create the global analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AsyncGPTAnalyzer()
        _analyzer_instance.start()
    return _analyzer_instance


def analyze_person_async(person_id: int, person_image: np.ndarray, danger_value: float) -> Dict[str, Any]:
    """
    Request async analysis for a person in potential danger.

    Args:
        person_id: Unique identifier for the person
        person_image: Cropped image of the person (from bounding box)
        danger_value: Current calculated danger value

    Returns:
        Immediate result if available, otherwise returns pending status
    """
    analyzer = get_analyzer()

    # Request analysis if danger threshold exceeded
    analyzer.request_analysis(person_id, person_image, danger_value)

    # Get cached result if available
    result = analyzer.get_result(person_id)

    if result:
        return {
            'status': 'completed',
            'needs_help': result['analysis'].get('needs_help', False),
            'confidence': result['analysis'].get('confidence', 0.0),
            'details': result['analysis']
        }
    else:
        return {
            'status': 'pending',
            'needs_help': False,
            'confidence': 0.0,
            'details': {}
        }


def cleanup_analyzer():
    """Cleanup the analyzer when shutting down."""
    global _analyzer_instance
    if _analyzer_instance:
        _analyzer_instance.stop()
        _analyzer_instance = None