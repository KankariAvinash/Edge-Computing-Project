import sounddevice as sd
import librosa
import numpy as np
from sklearn.svm import OneClassSVM
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

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

audio_files = [
    './NewAudio/1-speed.wav',
    './NewAudio/test-case.wav'
]

for audio_file in audio_files:
    y,sr = librosa.load(audio_file)
    plt.figure()
    plt.plot(y)
    plt.show()

labels = [0, 1]  # 0 = Normal, 1 = Abnormal
features = [extract_enhanced_features(librosa.load(f, sr=16000)[0], 16000) for f in audio_files]
X = np.array(features)

model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled)
# For live audio predictions, use the same enhanced feature extraction

# Continuously listen, visualize, and predict
try:
    while True:
        result = predict_anomaly_on_live_audio(model)
        print(f"Predicted sound condition: {result}")
except KeyboardInterrupt:
    print("Stopped by user.")

