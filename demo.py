import os
import preprocessing_tf
import numpy as np
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model

def select_audio_device():
    """Lists available audio devices and prompts the user to select one."""
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
    device_index = int(input("Select the device index for the demo: "))
    return device_index

def record_audio(duration=1.0, samplerate=16000, device=None):
    """Records audio for a fixed duration using SoundDevice."""
    print("\nPlease speak now...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16', device=device)
    sd.wait()
    return recording.flatten()


def compute_log_mel_spectrogram(audio, sample_rate, winlen=25, winstep=10, nfilt=40, nfft=512, lowfreq=300, highfreq=None):
    """Computes log Mel spectrogram from audio input."""
    audio = tf.cast(audio, tf.float32)
    frame_length = int(round(sample_rate * winlen / 1000))
    frame_step = int(round(sample_rate * winstep / 1000))
    highfreq = highfreq or sample_rate / 2
    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=nfft)
    spectrogram = tf.abs(stft)
    mel_weights = tf.signal.linear_to_mel_weight_matrix(nfilt, stft.shape[-1], sample_rate, lowfreq, highfreq)
    mel_spectrogram = tf.tensordot(spectrogram, mel_weights, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    return log_mel_spectrogram

def compute_mfccs(log_mel_spectrogram):
    """Computes MFCCs from log Mel spectrogram."""
    return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., tf.newaxis]

def recreate_string_lookup():
    """Recreates a StringLookup layer from a training dataset for label decoding."""
    file_path = r"C:\Users\maren\OneDrive\HDA_Project\project_data_split"
    train_df = preprocessing_tf.get_file_list(os.path.join(file_path, "train"))
    labels = tf.constant(train_df['mapped_label'].values)

    label_lookup = tf.keras.layers.StringLookup(num_oov_indices=0)
    label_lookup.adapt(labels)

    print("StringLookup Vocabulary (Label: Index):")
    for index, label in enumerate(label_lookup.get_vocabulary()):
        print(f"{label}: {index}")

    return label_lookup

def main():
    """Main function to run the audio classification demo."""

    #recreate string lookup for class decoding
    label_lookup = recreate_string_lookup()
    #load the model
    model_path = input("Enter the path of the model (.keras or .h5 file) you want to demonstrate: ")
    model = load_model(model_path)

    #selection of audio device
    device_index = select_audio_device()
    print("Speak into the selected device. Press 'q' and enter to quit the demo.")

    try:
        while True:
            #record and process audio
            audio = record_audio(device=device_index)
            log_mel_spec = compute_log_mel_spectrogram(audio, 16000)
            mfcc_features = compute_mfccs(log_mel_spec)
            mfcc_features = np.expand_dims(mfcc_features, axis=0)

            #predict audio with the chosen model
            prediction = model.predict(mfcc_features)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_class = label_lookup.get_vocabulary()[predicted_index]

            print(f"Predicted class: {predicted_class}")

            if input("Press 'q' and enter to quit, or just enter to continue: ").lower() == 'q':
                break
    except KeyboardInterrupt:
        print("Demo ended by user.")
    finally:
        print("Demo")


if __name__ == "__main__":
    main()