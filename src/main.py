import gc
import os
import torch
import tempfile
import yaml
import random
import pandas as pd
from ultralytics import YOLO


def group_test_images_by_subfolder(test_txt_path):
    with open(test_txt_path, 'r') as f:
        test_images = [line.strip() for line in f.readlines()]

    subfolder_images = {}
    for img_path in test_images:
        parts = os.path.normpath(img_path).split(os.sep)
        if len(parts) >= 3:
            subject = parts[-3]
            subfolder = parts[-2]
            key = (subject, subfolder)
            subfolder_images.setdefault(key, []).append(img_path)
    return subfolder_images


def evaluate_subfolders_generator(model, data_yaml_template_path, subfolder_images):
    # Get the dataset root from environment variables and normalize the path
    dataset_root = os.getenv('DATASET_ROOT').replace('\\', '/')

    for (subject, subfolder), images in subfolder_images.items():
        if not images:
            continue

        temp_test_txt = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_test_txt_name = temp_test_txt.name
        temp_test_txt.close()  # Close the file to avoid Windows locking issues

        with open(temp_test_txt_name, 'w') as f:
            for img_path in images:
                f.write(f"{img_path}\n")

        temp_data_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        temp_data_yaml_name = temp_data_yaml.name
        temp_data_yaml.close()  # Close the file to avoid Windows locking issues

        with open(data_yaml_template_path, 'r') as f:
            data_yaml_content = f.read()
        data_yaml_content = data_yaml_content.replace('{DATASET_ROOT}', dataset_root)
        data_yaml_dict = yaml.safe_load(data_yaml_content)
        data_yaml_dict['test'] = temp_test_txt_name
        data_yaml_dict['train'] = data_yaml_dict.get('train', '')
        data_yaml_dict['val'] = data_yaml_dict.get('val', '')

        # Write the updated data.yaml
        with open(temp_data_yaml_name, 'w') as f:
            yaml.dump(data_yaml_dict, f)

        # Run validation with reduced memory usage
        print(f"Evaluating Subject: {subject}, Subfolder: {subfolder}")
        results = model.val(
            data=temp_data_yaml_name,
            imgsz=736,
            single_cls=True,
            split='test',
            # batch=1  # Reduce batch size to reduce memory usage
        )

        metrics_dict = results.results_dict

        precision = metrics_dict.get('metrics/precision(B)', None)
        recall = metrics_dict.get('metrics/recall(B)', None)
        map50 = metrics_dict.get('metrics/mAP50(B)', None)
        map50_95 = metrics_dict.get('metrics/mAP50-95(B)', None)

        # Remove temporary files
        os.unlink(temp_test_txt_name)
        os.unlink(temp_data_yaml_name)

        # Explicitly delete variables and collect garbage
        del results
        gc.collect()

        yield {
            'Subject': subject,
            'Subfolder': subfolder,
            'Precision': precision,
            'Recall': recall,
            'mAP50': map50,
            'mAP50-95': map50_95,
        }


def get_train_val_test_data():
    train_ratio = float(os.getenv('TRAIN_RATIO', '0.7'))
    val_ratio = float(os.getenv('VAL_RATIO', '0.2'))
    test_ratio = float(os.getenv('TEST_RATIO', '0.1'))

    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"The sum of TRAIN_RATIO, VAL_RATIO, and TEST_RATIO must be 1.0, but got {total_ratio}")

    root_path = os.getenv('DATASET_ROOT')
    if not root_path:
        raise EnvironmentError("DATASET_ROOT environment variable must be set.")

    subjects = [f"Subject.{i}" for i in range(1, 11)]

    # Add all subfolders
    subfolders = ['Fall backwards', "Fall forward", "Fall left", "Fall right"]

    train_txt_path = os.path.join(root_path, 'train.txt')
    val_txt_path = os.path.join(root_path, 'val.txt')
    test_txt_path = os.path.join(root_path, 'test.txt')

    if os.path.exists(train_txt_path) and os.path.exists(val_txt_path) and os.path.exists(test_txt_path):
        print("Train, val, and test txt files already exist. Using existing data splits.")
        # Read existing files
        # with open(train_txt_path, 'r') as f:
        #     train_images = [line.strip() for line in f if line.strip()]
        # with open(val_txt_path, 'r') as f:
        #     val_images = [line.strip() for line in f if line.strip()]
        # with open(test_txt_path, 'r') as f:
        #     test_images = [line.strip() for line in f if line.strip()]
    else:
        print("Generating new train, val, and test splits.")
        train_images = []
        val_images = []
        test_images = []

        for subject in subjects:
            for subfolder in subfolders:
                image_dir = os.path.join(root_path, subject, subfolder)
                if not os.path.isdir(image_dir):
                    continue  # Skip if subfolder doesn't exist
                images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

                random.shuffle(images)

                num_images = len(images)
                train_size = int(num_images * train_ratio)
                val_size = int(num_images * val_ratio)

                train_images.extend(images[:train_size])
                val_images.extend(images[train_size:train_size + val_size])
                test_images.extend(images[train_size + val_size:])

        train_images = [img_path.replace('\\', '/') for img_path in train_images]
        val_images = [img_path.replace('\\', '/') for img_path in val_images]
        test_images = [img_path.replace('\\', '/') for img_path in test_images]

        with open(train_txt_path, 'w') as f:
            for img_path in train_images:
                f.write(f"{img_path}\n")

        with open(val_txt_path, 'w') as f:
            for img_path in val_images:
                f.write(f"{img_path}\n")

        with open(test_txt_path, 'w') as f:
            for img_path in test_images:
                f.write(f"{img_path}\n")

    return subjects, subfolders


def get_data_yaml():
    dataset_root = os.getenv('DATASET_ROOT')
    data_yaml_template_path = os.getenv('DATA_FILE')

    if not dataset_root or not data_yaml_template_path:
        raise EnvironmentError("DATASET_ROOT and DATA_FILE environment variables must be set.")

    with open(data_yaml_template_path, 'r') as f:
        data_yaml_str = f.read()

    data_yaml_str = data_yaml_str.replace('{DATASET_ROOT}', dataset_root.replace('\\', '/'))

    temp_data_yaml = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_data_yaml.write(data_yaml_str)
    temp_data_yaml.close()

    return temp_data_yaml.name


def train_model():
    model_base = os.getenv('MODEL_BASE')
    model_file = os.getenv('MODEL_FILE')
    if not model_base or not model_file:
        raise EnvironmentError("MODEL_BASE and MODEL_FILE environment variables must be set.")
    model_path = os.path.join(model_base, model_file)

    model = YOLO(model_path)

    # Check if CUDA is available
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        model.to('cuda')
        print("CUDA is available. Running on GPU.")

    get_train_val_test_data()

    data_yaml_path = get_data_yaml()

    should_train_model = os.getenv("TRAIN_MODEL", "0")

    if should_train_model == '1':
        print("\nTraining model\n")
        model.train(
            data=data_yaml_path,
            epochs=100,
            imgsz=736,
            single_cls=True,
            batch=-1,
        )

    else:
        print("Skipping training")

    should_run_tests = os.getenv('RUN_TESTS', '0')

    if should_run_tests == '1':
        print("\nTesting model\n")
        data_yaml_template_path = os.getenv('DATA_FILE')
        test_txt_path = os.path.join(os.getenv('DATASET_ROOT'), 'test.txt')
        subfolder_images = group_test_images_by_subfolder(test_txt_path)

        metrics_list = []
        for result in evaluate_subfolders_generator(model, data_yaml_template_path, subfolder_images):
            metrics_list.append(result)
            print(f"Finished evaluating {result['Subject']} - {result['Subfolder']}")
            # Force garbage collection after each iteration
            gc.collect()

        metrics_df = pd.DataFrame(metrics_list)
        print("\nPer-Subject and Per-Subfolder Metrics:")
        print(metrics_df)

        root_path = os.getenv('DATASET_ROOT')
        output_file = os.path.join(root_path, 'subject_subfolder_metrics.xlsx')
        metrics_df.to_excel(output_file, index=False)
        print(f"\nMetrics saved to {output_file}")
    else:
        print("Skipping tests")


if __name__ == '__main__':
    train_model()
