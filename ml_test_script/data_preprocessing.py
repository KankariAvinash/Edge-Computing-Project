import numpy as np
import librosa

def extract_features(audio, sample_rate, segment_duration=5):
    # Ensure audio is a float32 numpy array
    audio = np.array(audio, dtype=np.float32)

    # Segment the audio if needed
    segment_length = segment_duration * sample_rate
    num_segments = int(np.ceil(len(audio) / segment_length))
    features = []

    for i in range(num_segments):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(audio))
        segment = audio[start:end]
        
        # Extract MFCC features from the segment
        mfccs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=40)
        mfccs = np.mean(mfccs, axis=1)  # Take the mean of MFCCs across time
        features.append(mfccs)

    return features



