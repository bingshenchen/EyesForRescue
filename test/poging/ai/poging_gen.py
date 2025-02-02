import base64
import json
from dotenv import load_dotenv
from openai import OpenAI
import mimetypes

# Load .env file
load_dotenv()
client = OpenAI(
    # api_key=os.getenv("OPENAI_API_KEY")
    api_key="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"
)


def encode_image_with_type(image_path):
    """Encode the image to Base64 and dynamically set the MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        raise ValueError("Cannot determine the MIME type of the image. Please check the file path or format.")

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return mime_type, base64_image


def analyze_image(image_path):
    # Encode the image as Base64 and dynamically set the MIME type
    mime_type, base64_image = encode_image_with_type(image_path)

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
                                "environment": {
                                    "type": "road", "blaze", "water", "bed", "stoel" or "indoor", 
                                    "lighting": "bright", "dim", or "dark",
                                },
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

    return {
        'gpt_analysis': gpt_analysis
    }


# Example usage
if __name__ == "__main__":
    image_path = r"C:\Users\Bingshen\Pictures\AI Train\f.jpg"
    analysis_result = analyze_image(image_path)
    print(json.dumps(analysis_result, ensure_ascii=False, indent=2))
