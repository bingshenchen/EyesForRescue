from skimage.metrics import structural_similarity as ssim
import cv2
import os
from send2trash import send2trash
from concurrent.futures import ThreadPoolExecutor

def is_similar(image1, image2, threshold=0.98):
    """Check if two images are similar based on a given similarity threshold."""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Apply edge detection for stricter comparison
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)

    if edges1.shape == edges2.shape:
        score, _ = ssim(edges1, edges2, full=True)
        return score >= threshold
    else:
        return False

def process_image_pair(file1, path1, files, i, directory, threshold, compare_range, checked_files):
    img1 = cv2.imread(path1)

    # Compare with the next `compare_range` images only
    for j in range(i + 1, min(i + 1 + compare_range, len(files))):
        file2 = files[j]
        path2 = os.path.join(directory, file2)

        if file2 in checked_files or not os.path.exists(path2):
            continue

        img2 = cv2.imread(path2)

        if img1 is not None and img2 is not None and is_similar(img1, img2, threshold):
            send2trash(path2)  # 将文件移到垃圾箱
            checked_files.add(file2)
            print(f"Deleted {file2} as it is {threshold * 100}% similar to {file1}")

            # Remove the corresponding label file if it exists
            label_file = os.path.splitext(path2)[0] + '.txt'
            if os.path.exists(label_file):
                send2trash(label_file)  # 将标签文件移到垃圾箱
                print(f"Deleted corresponding label file {label_file}")

def remove_similar_images(directory, threshold=0.95, compare_range=5, max_workers=4):
    """Remove images and their corresponding label files in a directory if they are similar to others based on the given threshold.
       Use threading for better performance on large datasets."""
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    checked_files = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, file1 in enumerate(files):
            if file1 in checked_files:
                continue

            path1 = os.path.join(directory, file1)
            executor.submit(process_image_pair, file1, path1, files, i, directory, threshold, compare_range, checked_files)

# Define paths
directory_path = r"C:\Users\Bingshen\Desktop\Ucll verkorte toegepast informatica\Fase1\Semester 1\AI Applications\AI-Applications\assets\datasets\fall_detection\frames\train\train"
remove_similar_images(directory_path)
