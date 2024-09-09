import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Set up paths
DATA_DIR = 'dataset'  # Replace with the actual path0
SAMPLE_RATE = 22050  # Sample rate for audio files

# Labels for classification
LABELS = ['lowSpeed', 'HighSpeed', 'Abnormal']

# Load and preprocess the dataset
def load_data(data_dir):
    X, y = [], []
    for label in LABELS:
        folder_path = os.path.join(data_dir, label)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Load audio file
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            # Convert to Mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            # Resize to a fixed shape (e.g., 64x64)
            mel_spec_db = librosa.util.fix_length(mel_spec_db, size=64, axis=1)
            mel_spec_db = np.resize(mel_spec_db, (64, 64))
            X.append(mel_spec_db)
            y.append(label)
    return np.array(X), np.array(y)

# Load data
X, y = load_data(DATA_DIR)

# Encode labels as integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Expand dimensions for CNN (from 2D to 3D)
X = np.expand_dims(X, axis=-1)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(LABELS), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Save the trained model
model.save('audio_classification_model.h5')

# Load the model once, outside the loop
model = tf.keras.models.load_model('audio_classification_model.h5')

# Function to capture live audio and predict the label
def predict_live_audio(duration):
    print("Recording...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")

    # Convert recording to Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=recording.flatten(), sr=SAMPLE_RATE)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = librosa.util.fix_length(mel_spec_db, size=64, axis=1)
    mel_spec_db = np.resize(mel_spec_db, (64, 64))
    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)  # Add channel dimension

    # Make a prediction
    prediction = model.predict(np.array([mel_spec_db]))
    predicted_label = LABELS[np.argmax(prediction)]
    print(f"Predicted label: {predicted_label}")

try:
    while True:
        predict_live_audio(duration=5)
except KeyboardInterrupt:
    print("Stopped by user.")
