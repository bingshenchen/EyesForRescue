# src/core/analysis/gpt_analyzer.py

import base64
import json
import mimetypes
import logging
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI

from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration settings
settings = get_settings()

# Initialize OpenAI client
client = None
if settings.OPENAI_API_KEY:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
else:
    logger.warning("OpenAI API key not found. GPT analysis will not be available.")


def encode_image_with_type(image):
    """
    Encode the image to Base64 and dynamically set the MIME type.

    Args:
        image: Either a file path (str) or OpenCV frame (np.ndarray)

    Returns:
        tuple: (mime_type, base64_image)

    Raises:
        ValueError: If input format is invalid or MIME type cannot be determined
    """
    if isinstance(image, (str, Path)):  # Image path
        image_path = Path(image)

        # Validate file exists
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type or not mime_type.startswith('image/'):
            raise ValueError(f"Cannot determine valid image MIME type for: {image_path}")

        # Read and encode image
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Error reading image file: {e}")

    elif isinstance(image, (np.ndarray, np.generic)):  # OpenCV frame
        if image.size == 0:
            raise ValueError("Empty image frame provided")

        # Encode frame as JPEG
        try:
            _, buffer = cv2.imencode('.jpg', image)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            mime_type = "image/jpeg"
        except Exception as e:
            raise ValueError(f"Error encoding image frame: {e}")
    else:
        raise ValueError(
            "Invalid input type. Provide a valid image path (str/Path) or OpenCV frame (np.ndarray)."
        )

    return mime_type, base64_image


def analyze_image(image, save_result=False, output_dir=None):
    """
    Analyze an image using GPT-4 Vision to extract emergency-related information.

    Args:
        image: Either a file path (str/Path) or OpenCV frame (np.ndarray)
        save_result: Whether to save the analysis result to file
        output_dir: Directory to save results (defaults to outputs/temp/)

    Returns:
        dict: Analysis results containing GPT analysis

    Raises:
        RuntimeError: If OpenAI client is not initialized or API call fails
        ValueError: If image encoding fails
    """
    if not client:
        raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY in environment variables.")

    logger.info("Starting GPT image analysis...")

    # Encode image
    try:
        mime_type, base64_image = encode_image_with_type(image)
        logger.debug(f"Image encoded successfully. MIME type: {mime_type}")
    except ValueError as e:
        logger.error(f"Image encoding failed: {e}")
        raise

    # Prepare analysis prompt
    analysis_prompt = """
    Analyze the following image and return a JSON object with the following structure:
    {
        "onePerson": "true" or "false",
        "faceToTheGround": "true" or "false", 
        "possible_age": "old_people", "adults", or "children",
        "gender": "male" or "female",
        "status": [
            Multiple choice: "bleeding", "walk", "fall", "sit", "accident", "pain", "hurt", "drowning", "stampede"
        ],
        "environment": "road", "blaze", "water", "bed", "chair" or "indoor",
        "lighting": "bright", "dim", or "dark",
        "time_of_day": "day" or "night"
    }

    Ensure the result is valid JSON and avoid using descriptive or explanatory language.
    Focus on emergency detection and safety assessment.
    """

    # Call OpenAI API
    try:
        logger.info("Sending request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.2  # Lower temperature for more consistent results
        )
        logger.info("OpenAI API response received successfully")

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise RuntimeError(f"Failed to analyze image with GPT: {e}")

    # Extract and validate response content
    content = response.choices[0].message.content.strip()
    if not content:
        raise RuntimeError("Empty response from OpenAI API")

    # Clean up response (remove markdown formatting if present)
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()

    # Parse JSON response
    try:
        gpt_analysis = json.loads(content)
        logger.info("GPT analysis completed successfully")
        logger.debug(f"Analysis result: {json.dumps(gpt_analysis, indent=2)}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response: {content}")
        raise RuntimeError(f"GPT returned invalid JSON: {e}")

    # Prepare result
    result = {
        'gpt_analysis': gpt_analysis,
        'analysis_timestamp': None,  # Could add timestamp if needed
        'model_used': 'gpt-4o-mini'
    }

    # Save result if requested
    if save_result:
        try:
            save_analysis_result(result, output_dir)
        except Exception as e:
            logger.warning(f"Failed to save analysis result: {e}")

    return result


def save_analysis_result(result, output_dir=None):
    """
    Save analysis result to JSON file.

    Args:
        result: Analysis result dictionary
        output_dir: Output directory (defaults to outputs/temp/)
    """
    if output_dir is None:
        output_dir = settings.TEMP_DIR

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gpt_analysis_{timestamp}.json"

    # Save to file
    output_file = output_path / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Analysis result saved to: {output_file}")


def validate_analysis_result(analysis_result):
    """
    Validate that the analysis result contains expected fields.

    Args:
        analysis_result: Dictionary containing GPT analysis

    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = [
        'onePerson', 'faceToTheGround', 'possible_age', 'gender',
        'status', 'environment', 'lighting', 'time_of_day'
    ]

    gpt_analysis = analysis_result.get('gpt_analysis', {})

    for field in required_fields:
        if field not in gpt_analysis:
            logger.warning(f"Missing required field in analysis: {field}")
            return False

    return True


# Example usage and testing
if __name__ == "__main__":
    # Test with sample image (if available)
    test_image_path = settings.DATASETS_DIR / "test_image.jpg"  # Replace with actual test image

    if test_image_path.exists():
        try:
            result = analyze_image(test_image_path, save_result=True)
            print("Analysis Result:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

            # Validate result
            if validate_analysis_result(result):
                print("✅ Analysis result is valid")
            else:
                print("❌ Analysis result is missing required fields")

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    else:
        print(f"Test image not found at: {test_image_path}")
        print("Please provide a test image to run the example.")