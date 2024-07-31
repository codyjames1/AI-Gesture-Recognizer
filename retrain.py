import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import shutil

# Define file paths for data and model
BASE_DATA_DIR = 'data_tf'
COLLECTED_DATA_DIR = 'collected_data'
MODEL_FILE = os.path.join(BASE_DATA_DIR, 'trained_model.keras')

# Backup the old model if it exists
def backup_old_model():
    if os.path.exists(MODEL_FILE):
        backup_file = MODEL_FILE.replace('.keras', '_backup.keras')
        shutil.move(MODEL_FILE, backup_file)
        print(f"Old model backed up to {backup_file}")

# Function to load data
def load_data():
    X, y = [], []
    # Load original dataset
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        file_paths = {
            'features': os.path.join(BASE_DATA_DIR, f'features_{letter}.pkl'),
            'labels': os.path.join(BASE_DATA_DIR, f'labels_{letter}.pkl')
        }
        if os.path.exists(file_paths['features']) and os.path.exists(file_paths['labels']):
            with open(file_paths['features'], 'rb') as f:
                X_data = pickle.load(f)
            with open(file_paths['labels'], 'rb') as f:
                y_data = pickle.load(f)
            X.extend(X_data)
            y.extend(y_data)
    
    # Load collected data
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        directory = os.path.join(COLLECTED_DATA_DIR, letter)
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.npy'):
                    landmarks = np.load(os.path.join(directory, file)).flatten()
                    if landmarks.shape == (42,):  # Ensure the shape is correct
                        X.append(landmarks)
                        y.append(ord(letter) - 65)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y, random_state=42)
    
    return X, y

# Preprocess the data
def preprocess_data(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = to_categorical(y, num_classes=26)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
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

# Train the model
def train_model(X_train, y_train, X_val, y_val):
    input_shape = (X_train.shape[1],)
    model = build_model(input_shape)
    
    # Save the best model during training
    model_checkpoint = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint, early_stopping, reduce_lr]
    )

    return history

# Main function to load data, preprocess, and train the model
def main():
    backup_old_model()
    X, y = load_data()
    X_train, X_val, y_train, y_val = preprocess_data(X, y)
    history = train_model(X_train, y_train, X_val, y_val)
    print(f"Training completed with {len(history.epoch)} epochs.")

if __name__ == "__main__":
    main()