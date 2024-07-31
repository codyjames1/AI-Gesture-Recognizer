# AI Gesture Recognizer Documentation

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
  - [OpenCV](#opencv)
  - [MediaPipe](#mediapipe)
  - [TensorFlow](#tensorflow)
  - [Pyttsx3](#pyttsx3)
  - [Other Libraries](#other-libraries)
- [Dataset](#dataset)
- [Environment Setup](#environment-setup)
- [Future Work](#future-work)

## Introduction
The AI Gesture Recognizer is designed to translate American Sign Language (ASL) gestures into spoken words. This tool aims to bridge the communication gap between deaf and hearing individuals by providing real-time or phone-based ASL translation.

## Project Overview
The project consists of three main scripts:

- **run.py**: Runs the webcam feed, processes hand gestures, and predicts ASL letters using a trained model.
- **extract_features.py**: Processes a dataset of ASL images to extract hand landmarks and save them for training.
- **retrain.py**: Trains a neural network model using the extracted landmarks to predict ASL letters.

## Technologies Used

### OpenCV
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It contains more than 2500 optimized algorithms, which can be used for various computer vision tasks.

**Usage in the Project**:
- Capturing video from the webcam.
- Reading and resizing images.

**Relevant Code Snippet**:
```python
import cv2
```
# Capture video from the webcam
```Python
cap = cv2.VideoCapture(0)
```
# Resize images to speed up processing
```Python
def resize_image(image, target_height=480):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image
```
# MediaPipe
MediaPipe is a cross-platform framework for building multimodal applied machine learning pipelines. It is used for tasks such as hand tracking, face detection, and pose estimation.

## Usage in the Project:

Detecting and tracking hand landmarks in real-time using the webcam feed.
Relevant Code Snippet:

```python
Copy code
import mediapipe as mp
```
### Initialize MediaPipe Hands model and drawing utilities
```Python
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
```
### Initialize MediaPipe Hands model
```Python
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
```
### Process a frame from the webcam
```Python
results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```
### Draw landmarks on the image
```Python
for hand_landmarks in results.multi_hand_landmarks:
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style())
```
# TensorFlow
TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive ecosystem of tools, libraries, and community resources for building and deploying machine learning models.
## Usage in the Project:

Training a neural network model to predict ASL letters based on hand landmarks.
Loading and using the trained model for real-time predictions.
Relevant Code Snippet:

```python
Copy code
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
```
### Define the model
```Python
def build_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```
### Load the model from a file
```Python
model = tf.keras.models.load_model(MODEL_FILE)
```
# Pyttsx3
Pyttsx3 is a text-to-speech conversion library in Python. Unlike other libraries, it works offline and is compatible with both Python 2 and 3.

## Usage in the Project:

Converting the predicted ASL letter into speech.
Relevant Code Snippet:

```python
Copy code
import pyttsx3
```
### Initialize text-to-speech engine
```Python
engine = pyttsx3.init()
```
# Function to speak a message
```Python
def speak_message(message):
    engine.say(message)
    engine.runAndWait()
```
# Speak the recognized letter
```Python
def speak_letter(letter):
    engine.say(letter)
    engine.runAndWait()
```
# Other Libraries
## Numpy: 
Used for numerical operations and handling arrays.
## Pickle: 
Used for serializing and deserializing Python object structures.
## TQDM: 
Used for displaying progress bars.
### Relevant Code Snippets:

```python
Copy code
import numpy as np
import pickle
from tqdm import tqdm
```
### Normalize landmarks to a consistent scale
```Python
def normalize_landmarks(landmarks, image_shape):
    height, width = image_shape[:2]
    normalized_landmarks = [(lm.x * width, lm.y * height) for lm in landmarks]
    return np.array(normalized_landmarks).flatten()
```
### Save data to files
```Python
def save_data(X, y, letter):
    file_paths = get_feature_label_files(letter)
    ensure_directory(file_paths['features'])
    with open(file_paths['features'], 'wb') as f:
        pickle.dump(X, f)
    with open(file_paths['labels'], 'wb') as f:
        pickle.dump(y, f)
```
# Dataset
The model is trained using a dataset of approximately 600,000 images of the ASL alphabet. The dataset contains images for each letter of the alphabet (A-Z), with each image containing a hand gesture representing a letter.

# Environment Setup
To set up the environment for running the project, follow these steps:

### Install Python: Ensure you have Python installed. You can download it from python.org.

### Create a Virtual Environment:

## sh
Copy code
python -m venv newenv
Activate the Virtual Environment:

## On Windows:
sh
Copy code
.\newenv\Scripts\activate

## On macOS and Linux:
sh
Copy code
source newenv/bin/activate
Install Required Packages:

## sh
Copy code
pip install opencv-python mediapipe tensorflow pyttsx3
Future Work
The goal of this project is to fully translate ASL and other sign languages into spoken words, either over the phone or in real-time for in-person use. This will bridge the communication gap between deaf and hearing individuals, enabling more seamless interactions.

# Future improvements and enhancements could include:

Expanding the dataset to include more gestures and phrases.
Improving the model's accuracy and robustness.
Integrating the system into mobile and web applications for broader accessibility.
Adding support for multiple languages and dialects.
