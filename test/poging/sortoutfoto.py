import os
import shutil
import cv2
from ultralytics import YOLO
import numpy as np

# 定义路径
input_folder = r"C:\Users\Bingshen\Pictures\AI Train\Final\classifier\test\fine"
output_folders = {
    "falling_person": r"C:\Users\Bingshen\Pictures\AI Train\Final\croped\falling_person",
    "lying_person": r"C:\Users\Bingshen\Pictures\AI Train\Final\croped\lying_person",
    "standing_person": r"C:\Users\Bingshen\Pictures\AI Train\Final\croped\standing_person",
    "sitting_person": r"C:\Users\Bingshen\Pictures\AI Train\Final\croped\sitting_person",
}

# 确保目标文件夹存在
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

# 加载 YOLO11n-pose 模型
model = YOLO('yolo11n-pose.pt')


# 定义姿态分类函数
def classify_pose(keypoints):
    """
    根据关键点坐标判断姿态类别。
    keypoints: 关键点坐标列表，长度为17，每个元素为(x, y)元组。
    返回对应的姿态类别字符串。
    """
    if len(keypoints) < 17:  # 确保关键点完整
        return "unknown"

    # 提取所需的关键点
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    # 计算肩膀和臀部的中点
    mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
    mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

    # 计算头部到臀部的垂直距离
    vertical_distance = abs(nose[1] - mid_hip[1])

    # 计算肩膀到臀部的水平距离
    horizontal_distance = abs(mid_shoulder[0] - mid_hip[0])

    # 判断姿态
    if vertical_distance < horizontal_distance:
        if mid_hip[1] > mid_shoulder[1]:
            return "lying_person"
        else:
            return "falling_person"
    else:
        if left_knee[1] < left_hip[1] and right_knee[1] < right_hip[1]:
            return "standing_person"
        else:
            return "sitting_person"


# 批量加载图片
batch_size = 16
img_names = [img for img in os.listdir(input_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
img_paths = [os.path.join(input_folder, img) for img in img_names]

for i in range(0, len(img_paths), batch_size):
    batch_paths = img_paths[i:i + batch_size]
    images = [cv2.imread(p) for p in batch_paths]

    # 使用模型批量推理
    results = model(images)

    # 遍历每张图片的结果
    for img_path, result in zip(batch_paths, results):
        img_name = os.path.basename(img_path)
        if result.keypoints:  # 确保关键点存在
            keypoints = result.keypoints.xy.cpu().numpy()  # 提取关键点坐标
            if keypoints.size > 0:
                keypoints = keypoints[0]  # 取第一个检测结果的关键点
                # 分类姿态
                category = classify_pose(keypoints)
                if category in output_folders:
                    shutil.move(img_path, os.path.join(output_folders[category], img_name))
                    print(f"Moved {img_name} to {category}")
                else:
                    print(f"Skipping {img_name}, unknown category.")
            else:
                print(f"No valid keypoints detected in {img_name}. Skipping.")
        else:
            print(f"No keypoints detected in {img_name}. Skipping.")
