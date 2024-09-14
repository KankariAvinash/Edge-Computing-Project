import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_DIR = 'dataset/'  # Replace with your dataset directory
MODEL_PATH = 'final/model.h5'
TFLITE_MODEL_PATH = 'final/model.tflite'
TFLITE_MODEL_C_PATH = 'final/model.cc'

# Preprocess the data
def extract_mfcc(file_path, num_mfcc=13, max_pad_len=200):
    signal, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=num_mfcc)  # Corrected
    if mfcc.shape[1] < max_pad_len:
        padding = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

def load_data(data_dir):
    labels = ['lowSpeed', 'HighSpeed', 'Abnormal']
    data = []
    targets = []
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(label_dir, file_name)
                mfcc = extract_mfcc(file_path)
                data.append(mfcc)
                targets.append(labels.index(label))
    return np.array(data), np.array(targets)

# Load and preprocess data
X, y = load_data(DATA_DIR)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Flatten(input_shape=(13, 200)),  # Adjust input shape based on MFCC size
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the trained model
model.save(MODEL_PATH)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Model saved to {MODEL_PATH}")
print(f"TensorFlow Lite model saved to {TFLITE_MODEL_PATH}")

# Convert TensorFlow Lite model to C source file
def convert_to_c_source(tflite_model_path, c_source_path):
    os.system(f"xxd -i {tflite_model_path} > {c_source_path}")

def update_c_source_file(c_source_path, model_name):
    replace_text = TFLITE_MODEL_PATH.replace('/', '_').replace('.', '_')
    with open(c_source_path, 'r') as file:
        filedata = file.read()
    filedata = filedata.replace(replace_text, model_name)
    with open(c_source_path, 'w') as file:
        file.write(filedata)

# Generate C source file
convert_to_c_source(TFLITE_MODEL_PATH, TFLITE_MODEL_C_PATH)
update_c_source_file(TFLITE_MODEL_C_PATH, 'g_model')

print(f"C source file generated: {TFLITE_MODEL_C_PATH}")
