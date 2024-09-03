import sounddevice as sd
import librosa
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.svm import OneClassSVM
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Function to extract features from audio data
def extract_enhanced_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    rmse = np.mean(librosa.feature.rms(y=audio))
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr))

    feature_vector = np.hstack([
        mfccs_mean,
        spectral_centroid,
        spectral_bandwidth,
        spectral_contrast,
        spectral_rolloff,
        zero_crossing_rate,
        rmse,
        chroma
    ])
    return feature_vector

# Capture live audio data from the microphone
def capture_and_visualize_live_audio(duration=10, sr=16000):
    print("Listening...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    audio = audio.flatten()
    print("Processing...")

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title("Live Audio Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show(block=False)
    plt.pause(0.1)  # Adjust this if the plot does not update properly
    plt.close()

    return audio, sr

# Use the trained model to predict anomalies on live audio
def predict_anomaly_on_live_audio(model):
    audio, sr = capture_and_visualize_live_audio()
    features = extract_enhanced_features(audio, sr)
    features = features.reshape(1, -1)  # Reshape for the model
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)
    return "Abnormal" if prediction == -1 else "Normal"

# Load and train the model
# Assuming you've already trained your OneClassSVM or another model
# Replace the following with your actual training code if needed

# Example training data (use actual data in practice)
# Load your audio files and labels here
# For simplicity, here we are assuming the model is already trained and loaded

abnormal_dataset_path = "dataset/Abnormal"
abnormal_files = [join(abnormal_dataset_path, f) for f 
                        in listdir(abnormal_dataset_path) 
                        if isfile(join(abnormal_dataset_path, f))]

highspeed_dataset_path = "dataset/HighSpeed"
highspeed_files = [join(highspeed_dataset_path, f) for f 
                        in listdir(highspeed_dataset_path) 
                        if isfile(join(highspeed_dataset_path, f))]

lowspeed_dataset_path = "dataset/lowSpeed"
lowspeed_files = [join(lowspeed_dataset_path, f) for f 
                        in listdir(lowspeed_dataset_path) 
                        if isfile(join(lowspeed_dataset_path, f))]

general_labels = ["abnormal","highspeed","lowspeed"]

i = 0
features = []
labels = []
for audio_path in [abnormal_files,highspeed_files,lowspeed_files]:
    label = general_labels[i]
    show_first = True
    for audio_file in audio_path:
        audio,sr = librosa.load(audio_file)
        labels.append(label)
        features.append(extract_enhanced_features(librosa.load(audio_file, sr=16000)[0], 16000))
        if show_first:
            plt.figure()
            plt.plot(audio)
            plt.show()
            show_first = False
    i += 1

X = np.array(features)
print(X.shape)
y = labels
print(len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
model = RandomForestClassifier(n_estimators=100, random_state=42)
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_train)
#y_train_preds = model.fit_predict(X_scaled)
model.fit(X_train, y_train)
#print("Train metrics")
#print(classification_report(y_train, y_train_preds))

#X_scaled_test = scaler.transform(X_test)
y_test_preds = model.predict(X_test)
print("Test metrics")
print(classification_report(y_test, y_test_preds))
print(confusion_matrix(y_test, y_test_preds, 
                    labels=["abnormal","highspeed","lowspeed"]))

# For live audio predictions, use the same enhanced feature extraction

# Continuously listen, visualize, and predict
try:
    while True:
        result = predict_anomaly_on_live_audio(model)
        print(f"Predicted sound condition: {result}")
except KeyboardInterrupt:
    print("Stopped by user.")

