import os
import random
import shutil
from PIL import Image


def create_train_test_folders(base_path):
    # 创建所需的文件夹结构
    folders = [
        os.path.join(base_path, 'train', 'fine'),
        os.path.join(base_path, 'train', 'needhelp'),
        os.path.join(base_path, 'test', 'fine'),
        os.path.join(base_path, 'test', 'needhelp')
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)


def copy_images(source_folder, destination_folder, num_images):
    # 获取源文件夹中的所有图片文件
    all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    # 随机选择num_images张图片
    selected_images = random.sample(all_images, num_images)

    # 复制图片到目标文件夹
    for image in selected_images:
        src_path = os.path.join(source_folder, image)
        dst_path = os.path.join(destination_folder, image)
        shutil.copy(src_path, dst_path)


def prepare_data(falling_person_folder, lying_person_folder, base_output_path):
    # 创建训练和测试的文件夹结构
    create_train_test_folders(base_output_path)

    # 从falling_person中提取600张图片，并分配到train和test
    copy_images(falling_person_folder, os.path.join(base_output_path, 'train', 'fine'), 500)
    copy_images(falling_person_folder, os.path.join(base_output_path, 'test', 'fine'), 100)

    # 从lying_person中提取600张图片，并分配到train和test
    copy_images(lying_person_folder, os.path.join(base_output_path, 'train', 'needhelp'), 500)
    copy_images(lying_person_folder, os.path.join(base_output_path, 'test', 'needhelp'), 100)

    print("Data preparation complete!")


# 定义源文件夹路径和目标文件夹路径
falling_person_folder = r"C:\Users\Bingshen\Pictures\AI Train\Minio\falling_person"
lying_person_folder = r"C:\Users\Bingshen\Pictures\AI Train\Minio\lying_person"
base_output_path = r"C:\Users\Bingshen\Pictures\AI Train\Minio\fine_needhelp"

# 执行数据准备
prepare_data(falling_person_folder, lying_person_folder, base_output_path)
