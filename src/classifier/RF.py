import os

import cv2
import joblib
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd

load_dotenv()

classifier_location = Path(os.getenv('PROJECT_ROOT')) / "assets" / "classifier" / "classifier.pkl"

fine_features_location = Path(os.getenv('PROJECT_ROOT')) / "assets" / "classifier" / "fine_features.pkl"
fine_labels_location = Path(os.getenv('PROJECT_ROOT')) / "assets" / "classifier" / "fine_labels.pkl"

needhelp_features_location = Path(os.getenv('PROJECT_ROOT')) / "assets" / "classifier" / "needhelp_features.pkl"
needhelp_labels_location = Path(os.getenv('PROJECT_ROOT')) / "assets" / "classifier" / "needhelp_labels.pkl"


# Set paths for data
data_dir = Path(os.getenv('PROJECT_ROOT')) / "assets" / "data"
fine_dir = data_dir / "fine"
needhelp_dir = data_dir / "needhelp"
print(f"Fine folder {fine_dir} contains: {len(list(fine_dir.glob('*.*')))} images")
print(f"Need Help folder {needhelp_dir} contains: {len(list(needhelp_dir.glob('*.*')))} images")

# Load the YOLO pose model
model = YOLO("yolo11n-pose.pt")  # Make sure to use the pose model

# Initialize empty dataframe
pose_df = pd.DataFrame(columns=["image_path", "label", "features"])


# Feature extraction function using OpenCV for resizing
def extract_features_from_image(image_path, label):
    img = cv2.imread(str(image_path))
    img = cv2.resize(img, (224, 224))  # Resize image using OpenCV
    img = img.astype(np.uint8)  # Ensure it's in the proper format

    # Get pose features using YOLO model
    results = model(img)  # Perform pose detection

    # Assuming you want to extract the pose keypoints (this depends on the model output)
    keypoints = results[0].keypoints  # First element in results, and get the keypoints

    # Check if keypoints are present
    if keypoints is not None:
        # Extract keypoint coordinates (xy)
        keypoints_xy = keypoints.xy.numpy()  # Convert tensor to numpy array

        # Flatten keypoints (x, y) values
        features = keypoints_xy.flatten()  # Flatten the coordinates into a 1D array
    else:
        features = np.zeros(34)  # If no keypoints detected, use a zero vector (adjust size if necessary)

    return features


def extract_features_from_folder(folder, label):
    feature_vectors = []
    labels = []
    for image_path in folder.glob("*.*"):
        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            features = extract_features_from_image(image_path, label)
        if features is not None:
            feature_vectors.append(features)
            labels.append(label)
            pose_df.loc[len(pose_df)] = [image_path, label, features]
    return feature_vectors, labels


if fine_labels_location.is_file() and needhelp_labels_location.is_file()\
    and needhelp_features_location.is_file() and classifier_location.is_file():
    print("Loading features and labels from file")
    fine_features = joblib.load(fine_features_location)
    needhelp_features = joblib.load(needhelp_features_location)

    fine_labels = joblib.load(fine_labels_location)
    needhelp_labels = joblib.load(needhelp_labels_location)
else:
    print("Extracting features and labels from images")
    # Extract features for both classes
    fine_features, fine_labels = extract_features_from_folder(fine_dir, "Fine")
    needhelp_features, needhelp_labels = extract_features_from_folder(needhelp_dir, "Need Help")

    joblib.dump(fine_features, fine_features_location)
    joblib.dump(needhelp_features, needhelp_features_location)

    joblib.dump(needhelp_labels, needhelp_labels_location)
    joblib.dump(fine_labels, fine_labels_location)


# Combine both sets of labels before encoding
all_labels = fine_labels + needhelp_labels

# Encode labels (fit the encoder on all labels)
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)  # Fit the encoder on all labels (fine_labels + needhelp_labels)

# Encode the labels for both sets
fine_labels_encoded = label_encoder.transform(fine_labels)
needhelp_labels_encoded = label_encoder.transform(needhelp_labels)

# Ensure all feature vectors are the same size by padding or truncating
# Let's assume we want 34 elements (based on the flattened output of 17 keypoints with (x, y))

# Option 1: Use padding if feature vectors have different lengths
max_length = 34  # Adjust this based on your specific feature size



fine_features_padded = [np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length] for f
                        in fine_features]
needhelp_features_padded = [np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length]
                            for f in needhelp_features]

# Combine features and labels from both classes
X = np.concatenate([fine_features_padded, needhelp_features_padded], axis=0)
y = np.concatenate([fine_labels_encoded, needhelp_labels_encoded], axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=78)

if classifier_location.is_file():
    print("Loading classifier from file")
    clf = joblib.load(classifier_location)
else:
    # Train a Random Forest Classifier
    print("Classifier file not found, creating a new one")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)

if not classifier_location.is_file():
    print("Saving classifier to file")
    joblib.dump(clf, classifier_location)

# Predict on test set
y_pred = clf.predict(X_test)

# Print classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)
