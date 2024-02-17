import os
import warnings
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ level warnings
import numpy as np
import tensorflow as tf
from scipy.stats import mode
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress TensorFlow Python level warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore')
import sounddevice as sd
import preprocessing_tf

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

    return label_lookup

def decide_final_class(labels, confidences):
    """determine final prediction of ensemble,first majority vote when there is no majority confidence based"""
    unique, counts = np.unique(labels, return_counts=True)
    if np.any(counts > 1):  # A majority vote exists
        majority_vote_label = mode(labels, axis=1)[0].flatten()[0]
        return majority_vote_label
    else:  # No majority, pick the class with the highest confidence
        max_confidence_index = np.argmax(confidences)
        return labels.flatten()[max_confidence_index]

def main():
    # Model selection
    label_lookup = recreate_string_lookup()
    vocab = label_lookup.get_vocabulary()

    model_path = input("Enter the path of the model you want to load (or enter 'e' to load the ensemble model): ")

    if model_path == 'e':
        model1 = tf.keras.models.load_model(r"G:\Meine Ablage\Uni\UniPD\HumanDataProject\Models\ensembel_model1")
        model2 = tf.keras.models.load_model(r"G:\Meine Ablage\Uni\UniPD\HumanDataProject\Models\ensembel_model2")
        model3 = tf.keras.models.load_model(r"G:\Meine Ablage\Uni\UniPD\HumanDataProject\Models\ensembel_model3")
    else:
        model = tf.keras.models.load_model(model_path)

    # Select input audio device
    device_index = select_audio_device()

    # Instructions to end the demo
    print("Speak into the selected device. Press 'q' and enter to quit the demo.")

    try:
        while True:
            # Record audio
            print("\nPlease speak now...")
            time.sleep(0.3)
            audio = record_audio(device=device_index)
            log_mel_spec = compute_log_mel_spectrogram(audio, 16000)  # Ensure sample rate matches recording
            mfcc_features = compute_mfccs(log_mel_spec)
            mfcc_features = np.expand_dims(mfcc_features, axis=0)  # Model expects batch dimension


            # Predict and print result
            if model_path == 'e':
                #load the ensemble models
                predictions_model1 = model1.predict(mfcc_features)
                predictions_model2 = model2.predict(mfcc_features)
                predictions_model3 = model3.predict(mfcc_features)

                #get confidence of predicted classes
                max_confidence_model1 = np.max(predictions_model1, axis=1)
                max_confidence_model2 = np.max(predictions_model2, axis=1)
                max_confidence_model3 = np.max(predictions_model3, axis=1)

                #get label (integer label) of predicted classes
                label_model1 = np.argmax(predictions_model1, axis=1)
                label_model2 = np.argmax(predictions_model2, axis=1)
                label_model3 = np.argmax(predictions_model3, axis=1)

                # Stack the labels and the confidences
                stacked_labels = np.stack([label_model1, label_model2, label_model3], axis=1)
                stacked_confidences = np.stack([max_confidence_model1, max_confidence_model2, max_confidence_model3],
                                               axis=1)

                # Determine the final class
                final_class_index = decide_final_class(stacked_labels, stacked_confidences)
                # get string label of predicted class
                predicted_class = vocab[final_class_index]

                #print(stacked_labels)
                #print(stacked_confidences)
                #print(final_class_index)


            else:
                prediction = model.predict(mfcc_features)
                predicted_index = np.argmax(prediction, axis=1)[0] # Get the index of the max probability

                # get string label of predicted class
                predicted_class = vocab[predicted_index]

            print(f"Predicted class: {predicted_class}")


            # Check if user wants to end the demo
            if input("Press 'q' and enter to quit, or just enter to continue: ").lower() == 'q':
                break

    except KeyboardInterrupt:
        pass

    print("Demo ended.")


if __name__ == "__main__":
    main()
