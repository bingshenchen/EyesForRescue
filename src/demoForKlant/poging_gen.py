import base64
import json
import os

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import mimetypes

# Load .env file
load_dotenv()
client = OpenAI(
    api_key= os.getenv("OPENAIAPI_KEY")
)


def encode_image_with_type(image):
    """Encode the image to Base64 and dynamically set the MIME type.

    Args:
        image:
    """
    if isinstance(image, str):  # Case 1: Image path
        mime_type, _ = mimetypes.guess_type(image)
        if not mime_type:
            raise ValueError("Cannot determine the MIME type of the image. Please check the file path or format.")

        with open(image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, (np.ndarray, np.generic)):  # Case 2: OpenCV frame
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        mime_type = "image/jpeg"
    else:
        raise ValueError("Invalid input. Provide a valid image path (str) or an OpenCV frame (np.ndarray).")

    return mime_type, base64_image


def analyze_image(image):
    """
    Analyze an image from a file path or OpenCV frame.
    :param image: str (image path) or np.ndarray (OpenCV frame)
    """
    # Encode the image as Base64 and dynamically set the MIME type

    print("Received frame for analysis")
    print(f"Frame shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")

    mime_type, base64_image = encode_image_with_type(image)

    # Call OpenAI API to analyze the image
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":
                        """
                            Analyze the following image and return a JSON object with the following structure:
                            {
                                "onePerson": "true" or "false",
                                "faceToTheGround": "true" or "false",
                                "possible_age": "old_people", "adults", or "children",
                                "gender": "male" or "female",
                                "status": [
                                    Multiple choice: "bleeding", "walk", "fall", "sit", "accident", "pain", "hurt", "drowning", "stampede"
                                ],
                                "environment": "road", "blaze", "water", "bed", "stoel" or "indoor", 
                                "lighting": "bright", "dim", or "dark",
                                "time_of_day": "day" or "night",
                            }.
                            Ensure the result is valid JSON and avoid using descriptive or explanatory language. """
                     },

                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ],
            }
        ],
        max_tokens=300
    )

    # Extract and validate the returned content
    content = response.choices[0].message.content.strip()
    if not content:
        raise ValueError("The response is empty. Please check the API request or the model's behavior.")

    # Remove Markdown code block if present
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()  # Strip off the ```json and ``` markers

    try:
        gpt_analysis = json.loads(content)
    except json.JSONDecodeError:
        print("Raw response content:", content)  # Debugging information
        raise ValueError("The response is not in valid JSON format.")

    print(json.dumps(gpt_analysis, ensure_ascii=False, indent=2))

    return {
        'gpt_analysis': gpt_analysis
    }


# Example usage
if __name__ == "__main__":
    image_path = os.getenv("TEST_IMAGE")
    analysis_result = analyze_image(image_path)
    print(json.dumps(analysis_result, ensure_ascii=False, indent=2))
