import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import pyttsx3
import threading

# Initialize MediaPipe Hands model and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize text-to-speech engine
engine = pyttsx3.init()

voices = engine.getProperty('voices')
# Set voice to female
for voice in voices:
    if 'female' in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('voice', voices[1].id)

# Function to speak a message
def speak_message(message):
    engine.say(message)
    engine.runAndWait()

# Define file paths for data and model
BASE_DATA_DIR = 'data_tf'
MODEL_FILE = os.path.join(BASE_DATA_DIR, 'trained_model.keras')
COLLECTED_DATA_DIR = 'collected_data'

# Load the model from a file
def load_model():
    if os.path.exists(MODEL_FILE):
        print("Loading model from file...")
        try:
            model = tf.keras.models.load_model(MODEL_FILE)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print("Model file not found.")
        return None

# Resize images to speed up processing
def resize_image(image, target_height=480):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image

# Normalize landmarks
def normalize_landmarks(landmarks, image_shape):
    height, width = image_shape[:2]
    normalized_landmarks = []
    for landmark in landmarks:
        normalized_landmarks.append(landmark.x * width)
        normalized_landmarks.append(landmark.y * height)
    return np.array(normalized_landmarks).flatten()

# Draw a bounding box around the hand
def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min = int(min([landmark.x for landmark in hand_landmarks]) * image_width)
    y_min = int(min([landmark.y for landmark in hand_landmarks]) * image_height)
    x_max = int(max([landmark.x for landmark in hand_landmarks]) * image_width)
    y_max = int(max([landmark.y for landmark in hand_landmarks]) * image_height)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return x_min, y_min

# Function to save frame data
def save_frame_data(letter, landmarks):
    letter_dir = os.path.join(COLLECTED_DATA_DIR, letter)
    os.makedirs(letter_dir, exist_ok=True)
    filename = os.path.join(letter_dir, f'{letter}_{int(time.time())}.npy')
    np.save(filename, landmarks)
    print(f'Saved data for letter {letter} at {filename}')

# Speak the recognized letter
def speak_letter(letter):
    engine.say(letter)
    engine.runAndWait()

# Function to process a single frame from webcam
def process_frame(frame, hands, model, current_prediction, confidence_threshold, last_spoken_time, cooldown):
    image = resize_image(frame)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            landmarks = normalize_landmarks(hand_landmarks.landmark, image.shape)
            if len(landmarks) == 42:
                landmarks = landmarks.reshape(1, -1)
                prediction = model.predict(landmarks)
                label = np.argmax(prediction)
                confidence_score = np.max(prediction)
                
                if confidence_score > confidence_threshold:
                    current_time = time.time()
                    if current_prediction[0] != label or (current_time - last_spoken_time) > cooldown:
                        current_prediction = (label, confidence_score)
                        last_spoken_time = current_time
                        # Speak the predicted letter in a separate thread
                        threading.Thread(target=speak_letter, args=(chr(current_prediction[0] + 65),)).start()
                
                x_min, y_min = draw_bounding_box(image, hand_landmarks.landmark)
                cv2.putText(image, f'{chr(label + 65)} ({confidence_score:.2f})',
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                return image, current_prediction[0], confidence_score, landmarks, last_spoken_time
    return image, current_prediction[0], current_prediction[1], None, last_spoken_time

# Main function to run webcam and model
def main():
    try:
        # Speak the initial greeting message
        threading.Thread(target=speak_message, args=("Hello, I am Gertrude. Start signing and I'll translate for you!",)).start()
        
        model = load_model()
        if model is None:
            print("Model not found or failed to load.")
            return

        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Webcam opened successfully.")
        
        for _ in range(10):
            cap.read()
        
        current_prediction = (0, 0.0)
        confidence_threshold = 0.7
        last_spoken_time = 0
        cooldown = 5  # 5 seconds cooldown
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            
            image, label, confidence, landmarks, last_spoken_time = process_frame(frame, hands, model, current_prediction, confidence_threshold, last_spoken_time, cooldown)
            cv2.imshow('Webcam Feed', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key >= ord('a') and key <= ord('z'):
                if landmarks is not None:
                    save_frame_data(chr(key).upper(), landmarks)
            
            if key == ord('q'):
                break
    
        cap.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()