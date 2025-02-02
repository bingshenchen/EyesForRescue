import os
import random
from pathlib import Path


def limit_images_in_folder(folder_path, target_count):
    """
    Randomly deletes images in a folder to limit the total count to the target_count.
    :param folder_path: Path to the folder containing images.
    :param target_count: Target number of images to retain.
    """
    # Get all image files in the folder
    image_files = list(Path(folder_path).glob("*.jpg"))

    # If the folder contains fewer or equal images than the target, no need to delete
    if len(image_files) <= target_count:
        print(f"No need to delete images in {folder_path}. Current count: {len(image_files)}")
        return

    # Randomly shuffle and select images to delete
    random.shuffle(image_files)
    images_to_delete = image_files[target_count:]

    # Delete the selected images
    for image_path in images_to_delete:
        os.remove(image_path)
        print(f"Deleted: {image_path}")

    print(f"Reduced {folder_path} to {target_count} images.")


def main():
    # Define paths for train and test directories
    base_dir = r"C:\Users\Bingshen\Pictures\people_static_fine_and_needhelp"
    train_fine_dir = os.path.join(base_dir, "train", "fine")
    train_needhelp_dir = os.path.join(base_dir, "train", "needhelp")
    test_fine_dir = os.path.join(base_dir, "test", "fine")
    test_needhelp_dir = os.path.join(base_dir, "test", "needhelp")

    # Target counts
    train_target = 2500
    test_target = 500

    # Process each folder
    limit_images_in_folder(train_fine_dir, train_target)
    limit_images_in_folder(train_needhelp_dir, train_target)
    limit_images_in_folder(test_fine_dir, test_target)
    limit_images_in_folder(test_needhelp_dir, test_target)


if __name__ == "__main__":
    main()
