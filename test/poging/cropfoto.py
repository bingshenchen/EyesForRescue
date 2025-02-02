import os
import cv2

# 文件夹路径
images_folder = r"C:\Users\Bingshen\Pictures\AI Train\Final\sleep detection.v4i.yolov11\valid\images"
labels_folder = r"C:\Users\Bingshen\Pictures\AI Train\Final\sleep detection.v4i.yolov11\valid\labels"
output_folder = r"C:\Users\Bingshen\Pictures\AI Train\Final\sleep detection.v4i.yolov11\output"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 全局计数器，用于裁剪图片的唯一命名
global_counter = 1


def crop_person_from_image(image_path, label_path, output_folder, prefix="sleeping_person3"):
    """
    根据YOLO标签裁剪图片中的每个目标并保存，仅裁剪标签ID为0的目标
    """
    global global_counter

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # 获取图像尺寸
    img_height, img_width, _ = image.shape

    # 读取标签文件
    with open(label_path, "r") as label_file:
        lines = label_file.readlines()

    # 遍历每一行标签（每行代表一个目标）
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Invalid label format in {label_path}: {line}")
            continue

        # YOLO格式: 类别中心X中心Y宽高（归一化到0~1）
        class_id, center_x, center_y, width, height = map(float, parts)

        # 只处理标签ID为0的目标
        if int(class_id) != 0:
            print(f"Skipping non-target label ID {int(class_id)} in {label_path}")
            continue

        # 转换到像素坐标
        x1 = int((center_x - width / 2) * img_width)
        y1 = int((center_y - height / 2) * img_height)
        x2 = int((center_x + width / 2) * img_width)
        y2 = int((center_y + height / 2) * img_height)

        # 确保坐标在图像范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_width, x2), min(img_height, y2)

        # 裁剪图像
        cropped_img = image[y1:y2, x1:x2]
        if cropped_img.size == 0:
            print(f"Empty crop for {label_path} at {line}")
            continue

        # 保存裁剪后的图片
        output_path = os.path.join(output_folder, f"{prefix}_{global_counter}.jpg")
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved cropped image: {output_path}")

        # 增加全局计数器
        global_counter += 1


if __name__ == "__main__":
    # 遍历images文件夹中的图片
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(images_folder, image_file)
            label_path = os.path.join(labels_folder, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))

            if not os.path.exists(label_path):
                print(f"No label file found for image: {image_file}")
                continue

            # 调用函数进行裁剪
            crop_person_from_image(image_path, label_path, output_folder)
