import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands

# Define file paths for saving/loading data
DATASET_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
BASE_DATA_DIR = 'data_tf'
BATCH_SIZE = 1000  # Adjust the batch size as needed

def get_feature_label_files(letter):
    return {
        'features': os.path.join(BASE_DATA_DIR, f'features_{letter}.pkl'),
        'labels': os.path.join(BASE_DATA_DIR, f'labels_{letter}.pkl')
    }

# Ensure the directory exists
def ensure_directory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Resize images to speed up processing
def resize_image(image, target_height=480):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image

# Function to process a single image
def process_image(image_path, hands):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image: {image_path}")
            return None, None
        
        image = resize_image(image)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image, results
    except Exception as e:
        print(f"Exception occurred while processing image {image_path}: {e}")
        return None, None

# Normalize landmarks to a consistent scale
def normalize_landmarks(landmarks, image_shape):
    height, width = image_shape[:2]
    normalized_landmarks = [(lm.x * width, lm.y * height) for lm in landmarks]
    return np.array(normalized_landmarks).flatten()

# Function to process a dataset and extract hand landmarks in batches
def process_dataset(letter, dataset_path):
    X = []
    y = []
    total_images = 0
    processed_images = 0
    error_images = 0

    # Path to the dataset folder
    specific_folder_path = os.path.join(dataset_path, letter)

    if not os.path.isdir(specific_folder_path):
        print(f"Error: Folder {specific_folder_path} does not exist.")
        return np.array(X), np.array(y)

    # List of image files in the specific folder
    image_files = os.listdir(specific_folder_path)
    total_images = len(image_files)
    
    for batch_start in range(0, total_images, BATCH_SIZE):
        batch_files = image_files[batch_start:batch_start + BATCH_SIZE]
        with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
            for image_name in tqdm(batch_files, desc=f"Processing Dataset {letter}, Batch {batch_start//BATCH_SIZE + 1}"):
                image_path = os.path.join(specific_folder_path, image_name)
                image, results = process_image(image_path, hands)
                processed_images += 1
                if image is None:
                    error_images += 1
                    continue
                if results and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = normalize_landmarks(hand_landmarks.landmark, image.shape)
                        if len(landmarks) == 42:  # Ensure correct number of landmarks
                            X.append(landmarks)
                            y.append(ord(letter) - ord('A'))  # Label for dataset
                else:
                    error_images += 1

    print(f"Dataset {letter} - Total images: {total_images}")
    print(f"Dataset {letter} - Processed images: {processed_images}")
    print(f"Dataset {letter} - Error images: {error_images}")

    return np.array(X), np.array(y)

# Save data to files
def save_data(X, y, letter):
    file_paths = get_feature_label_files(letter)
    ensure_directory(file_paths['features'])  # Ensure directory exists before saving
    with open(file_paths['features'], 'wb') as f:
        pickle.dump(X, f)
    with open(file_paths['labels'], 'wb') as f:
        pickle.dump(y, f)

def main():
    # Define path to the dataset directory
    dataset_path = 'C:/Users/Cody/Desktop/AI Gesture Recognizer/asl_alphabet_train'

    # Process datasets A through Z
    for letter in DATASET_LETTERS:
        print(f"Processing dataset {letter}...")
        X, y = process_dataset(letter, dataset_path)
        save_data(X, y, letter)
        print(f"Data processing and saving complete for dataset {letter}.")

if __name__ == "__main__":
    main()