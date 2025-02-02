from PIL import Image, ImageOps
import os

def resize_images_to_square(input_folder, output_folder, target_size=(480, 480)):
    """
    Resizes images in the input folder to square dimensions with white padding and saves to the output folder.

    :param input_folder: The folder containing input images.
    :param output_folder: The folder to save processed images.
    :param target_size: Tuple indicating the final size of the image (width, height).
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        # Check for image file extensions
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Open the image
                with Image.open(input_path) as img:
                    # Get dimensions
                    width, height = img.size
                    # Determine the new size for the square
                    new_size = max(width, height)
                    # Create a new white background image
                    square_image = Image.new("RGB", (new_size, new_size), "white")
                    # Paste the original image onto the center of the square
                    square_image.paste(img, ((new_size - width) // 2, (new_size - height) // 2))
                    # Resize to the target size
                    resized_image = square_image.resize(target_size)
                    # Save the result
                    resized_image.save(output_path)
                    print(f"Processed and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
input_folder = r"C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications\assets\datasets\classifier\test\needhelp"
output_folder = r"C:\Users\Bingshen\Pictures\classifier\test\needhelp"

resize_images_to_square(input_folder, output_folder)
