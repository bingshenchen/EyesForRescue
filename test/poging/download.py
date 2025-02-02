import kagglehub

# Download latest version
path = kagglehub.dataset_download("wearefuture01/fall-detection")

print("Path to dataset files:", path)