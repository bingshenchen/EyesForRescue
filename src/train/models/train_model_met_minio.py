from minio import Minio
from dotenv import load_dotenv
import os
import logging
from ultralytics import YOLO
from minio.error import S3Error

load_dotenv()


def make_minio_client():
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")
    minio_uri = os.getenv("MINIO_URI")

    if not access_key or not secret_key or not minio_uri:
        raise ValueError("Minio credentials or URI not found in environment variables.")

    return Minio(minio_uri, secure=False, access_key=access_key, secret_key=secret_key)


def download_dataset_from_minio(client, bucket_name, object_name, download_path):
    try:
        client.fget_object(bucket_name, object_name, download_path)
        logging.info(f"Downloaded {object_name} to {download_path}")
    except S3Error as e:
        logging.error(f"Failed to download {object_name}: {e}")


def upload_results_to_minio(client, bucket_name, local_path, object_name):
    try:
        client.fput_object(bucket_name, object_name, local_path)
        logging.info(f"Uploaded {local_path} to MinIO as {object_name}")
    except S3Error as e:
        logging.error(f"Failed to upload {local_path}: {e}")


def train_yolo_model(model_dir_path, data_path, output_dir, epochs=30, imgsz=640, batch=8, mosaic=1.0, mixup=0.2):
    logging.info(f"Loading YOLO model from {model_dir_path}")

    model = YOLO(model_dir_path)

    logging.info(f"Starting training with dataset: {data_path}")

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        mosaic=mosaic,
        mixup=mixup,
        project=output_dir
    )

    logging.info("Training completed.")


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    client = make_minio_client()

    model_path = os.getenv('YOLO_MODEL_PATH')
    data_yaml_path = os.getenv('DATA_YAML_PATH')
    output_dir_path = os.getenv('OUTPUT_DIRECTORY')
    minio_bucket = os.getenv('MINIO_BUCKET')
    minio_object = os.getenv('MINIO_OBJECT_NAME')
    local_download_path = 'local_dataset.yaml'

    if not model_path or not data_yaml_path or not minio_bucket or not minio_object:
        logging.error("Error: Required environment variables are not set.")
        return

    download_dataset_from_minio(client, minio_bucket, minio_object, local_download_path)

    train_yolo_model(model_path, local_download_path, output_dir_path, epochs=30, imgsz=640, batch=8, mosaic=1.0,
                     mixup=0.2)

    upload_results_to_minio(client, minio_bucket, output_dir_path, "training_results.zip")


if __name__ == "__main__":
    main()
