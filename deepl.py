import os
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set up paths
DATA_DIR = 'dataset'  # Replace with the actual path
SAMPLE_RATE = 22050  # Sample rate for audio files

# Labels for classification
LABELS = ['lowSpeed', 'HighSpeed', 'Abnormal']

# Remove the existing model if present
if os.path.exists('audio_classification_model.h5'):
    os.remove('audio_classification_model.h5')

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
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512)
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
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(LABELS), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the trained model
model.save('audio_classification_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('audio_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert TFLite model to C source file
def convert_to_c_source_file(tflite_file, c_file):
    import subprocess
    # Generate the C source file from the TFLite model using xxd
    subprocess.run(['xxd', '-i', tflite_file], stdout=open(c_file, 'w'))

# Generate C source file
convert_to_c_source_file('audio_classification_model.tflite', 'audio_classification_model.cc')

# Function to compute Mel-spectrogram using TensorFlow
def compute_mel_spectrogram(audio, sample_rate, mel_spec_size):
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
    stft = tf.signal.stft(audio_tensor, frame_length=1024, frame_step=512, pad_end=True)
    num_spectrogram_bins = stft.shape[-1]
    mel_filter = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel_spec_size,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=125.0,
        upper_edge_hertz=3800.0
    )
    mel_spectrogram = tf.tensordot(tf.square(tf.abs(stft)), mel_filter, 1)
    mel_spectrogram_db = tf.math.log(mel_spectrogram + 1e-6)
    mel_spectrogram_db = tf.image.resize(mel_spectrogram_db[..., tf.newaxis], [mel_spec_size, mel_spec_size])
    return mel_spectrogram_db

# Function to capture live audio, predict the label, and plot the results
def plot_and_predict_live_audio(duration):
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    def update_plot(recording):
        # Clear previous plots
        ax1.clear()
        ax2.clear()

        # Plot waveform
        ax1.plot(recording)
        ax1.set_title('Waveform')
        ax1.set_ylim([-1, 1])  # Adjust y-limits for better visualization

        # Compute and plot Mel-spectrogram
        mel_spec_db = compute_mel_spectrogram(recording, SAMPLE_RATE, 64)
        mel_spec_db = mel_spec_db.numpy().squeeze()

        ax2.imshow(mel_spec_db, aspect='auto', cmap='viridis')
        ax2.set_title('Mel-Spectrogram')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Frequency')

        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to update

    try:
        while True:
            print("Recording...")
            recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()  # Wait for the recording to finish
            print("Recording complete.")

            # Update the plots
            update_plot(recording.flatten())

            # Convert recording to Mel-spectrogram for prediction
            mel_spec_db = compute_mel_spectrogram(recording.flatten(), SAMPLE_RATE, 64)
            mel_spec_db = np.expand_dims(mel_spec_db.numpy(), axis=-1)  # Add channel dimension

            # Make a prediction
            prediction = model.predict(np.array([mel_spec_db]))

            # Print prediction details
            predicted_index = np.argmax(prediction)
            predicted_label = LABELS[predicted_index]
            print(f"Predicted label: {predicted_label}")

            plt.pause(1)  # Pause before next recording

    except KeyboardInterrupt:
        print("Stopped by user.")
        plt.ioff()  # Turn off interactive mode
        plt.show()

# Start live audio plotting and prediction
plot_and_predict_live_audio(duration=5)
