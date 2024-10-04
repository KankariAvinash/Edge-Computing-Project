import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from data_preprocessing import extract_features
from tensorflow.keras.models import load_model

# Load the trained Neural Network model
model = load_model('sound_classification_model.h5')  # Use your trained model file name
print(model.summary())


# Directory to save graphs where 'Abnormal' is predicted
output_dir = 'abnormal_plots'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Label mapping
label_mapping = {0: 'ABNormalSpeed', 1: 'NormalSpeed'}

def predict_live_audio(segment_duration=5, sample_rate=44100, duration=60):
    print("Recording and predicting live audio...")
    start_time = time.time()  # Record start time
    plot_count = 0  # Counter for plot files

    fig, ax = plt.subplots()
    
    while (time.time() - start_time) < duration:
        # Record live audio in chunks of `segment_duration`
        audio = sd.rec(int(segment_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Wait for the recording to finish
        audio = audio.flatten()  # Flatten the array

        # Ensure audio is exactly the expected length (sample_rate * segment_duration)
        if audio.shape[0] != segment_duration * sample_rate:
            print("Audio segment is not the correct length; skipping this segment.")
            continue
        
        # Plot the waveform
        ax.clear()
        ax.plot(audio)
        ax.set_title("Live Audio Waveform")
        plt.pause(0.01)  # Update the plot in real-time

        # Calculate RMS (Root Mean Square) for sound intensity
        rms = np.sqrt(np.mean(audio**2))
        
        # Extract features
        features = extract_features(audio, sample_rate, segment_duration)
        print("Predicting...")

        for feature in features:
            feature = np.array(feature).reshape(1, -1)  # Reshape for model input
            prediction = model.predict(feature)
            predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index
            
            class_label = label_mapping.get(predicted_class, "Unknown")  # Map to label
            
            print(f'Predicted Class: {class_label}, Sound Intensity (RMS): {rms:.2f}')
            
            # Save the graph if the prediction is 'Abnormal'
            if predicted_class == 1:  # Assuming class '1' represents 'Abnormal'
                plot_count += 1
                plot_filename = os.path.join(output_dir, f'abnormal_plot_{plot_count}.png')
                fig.savefig(plot_filename)
                print(f'Saved abnormal plot as {plot_filename}')

    plt.show()  # Show the final plot

# Call the function to predict live audio for 1 minute
predict_live_audio(segment_duration=5, duration=60)
