import os
import glob
import random
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import resample
import tensorflow_io as tfio
import warnings
#from config import NOISE_FOLDER

def get_file_list(main_directory):

    # Replace this with the path to your main directory
    data = []

    # Walk through the main directory
    for dirpath, dirnames, filenames in os.walk(main_directory):
        for file in filenames:
            if file.endswith('.wav'):  # Assuming audio files are .mp3, change if needed
                # Create a dictionary for each file with filepath and label
                file_info = {
                    "filepath": os.path.join(dirpath, file),
                    "label": os.path.basename(dirpath)
                }
                data.append(file_info)

    # Create DataFrame
    df = pd.DataFrame(data)

    non_trigger_words = set(["bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"])
    df['mapped_label'] = df['label'].apply(lambda x: 'unknown' if x in non_trigger_words else x)

    return df


def read_wavfile(file_path):
    file_path = file_path.numpy()  # Convert the tensor to numpy array
    _, data = wavfile.read(file_path)
    return data.squeeze()

def tf_read_wavfile(file_path):
    # Wrap read_wavfile using tf.py_function
    [data] = tf.py_function(read_wavfile, [file_path], [tf.float32])
    data.set_shape([None])
    return data

def perform_padding(data, data_length=16000):
    padding_amount = data_length - tf.shape(data)[0]
    pad_before = padding_amount // 2
    pad_after = padding_amount - pad_before
    paddings = [[pad_before, pad_after]]
    data_padded = tf.pad(data, paddings, mode='CONSTANT', constant_values=0)
    data_padded.set_shape([data_length])
    print("Padding shape:", data_padded.shape)
    return data_padded



def add_noise(data, noise_paths ):
    #noise_file = tf.random.shuffle(noise_paths)[0]
    noise_file = random.choice(noise_paths)

    print(noise_file)


    # Adjust noise strength based on the selected noise file
    if 'white_noise.wav' in noise_file or 'pink_noise.wav' in noise_file:
        noise_strength = tf.random.uniform([], 0.1, 0.3, dtype=tf.float32)
    else:
        noise_strength = tf.random.uniform([], 0.2, 0.5, dtype=tf.float32)

    noise_data = tf_read_wavfile(noise_file)
    start_index = tf.random.uniform([], 0, tf.shape(noise_data)[0] - tf.shape(data)[0], dtype=tf.int32)
    noise_segment = noise_data[start_index:start_index + tf.shape(data)[0]]
    data_noisy = data + noise_strength * noise_segment
    data_noisy.set_shape([data.get_shape()[0]])
    print("Noisy shape:", data_noisy.shape)
    return data_noisy

def get_spectrogram(signal, samplerate, winlen=25, winstep=10, nfft=512, winfunc=tf.signal.hamming_window):
    #spectrogram = tf.signal.stft(tf.cast(signal, tf.float32), int(samplerate * winlen / 1000),
    #                             int(samplerate * winstep / 1000), nfft, winfunc)

    print("Signal shape:", signal.shape)
    spectrogram = tf.signal.stft(
        signal, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    print("Spectrogram shape:", spectrogram.shape)
    return tf.math.log(spectrogram + tf.keras.backend.epsilon())

def tf_resample_audio(data, original_sample_rate = 16000, target_sample_rate=8000):
    resampled_data = tfio.audio.resample(data, rate_in=original_sample_rate, rate_out=target_sample_rate)
    resampled_data.set_shape([target_sample_rate])
    print("Resampled shape:", resampled_data.shape)
    return resampled_data


def compute_log_mel_spectrogram(audio, sample_rate, winlen=25, winstep=10, nfilt=40, nfft=512, lowfreq=300,
                                highfreq=None):
    # Convert milliseconds to samples
    frame_length = int(round(sample_rate * winlen / 1000))
    frame_step = int(round(sample_rate * winstep / 1000))

    # Set highfreq to samplerate / 2 if None
    if highfreq is None:
        highfreq = sample_rate / 2

    stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=nfft)
    spectrogram = tf.abs(stft)

    num_spectrogram_bins = stft.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        nfilt, num_spectrogram_bins, sample_rate, lowfreq, highfreq)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    print("log_mel_spectrogram shape:", log_mel_spectrogram.shape)
    return log_mel_spectrogram


def compute_mfccs(log_mel_spectrogram, num_mfccs=13):
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., tf.newaxis]
    print("Mfcc shape:", mfccs.shape)
    return mfccs


def preprocess_map_new(file_path,label,noise=False,resample=False,logmel=False,mfcc=False):
    try:
        sample_rate = 16000
        NOISE_FOLDER = "/content/data/project_data_split/_background_noise_"

        data = tf_read_wavfile(file_path)
        data = perform_padding(data)

        if noise:
            noise_files = glob.glob(os.path.join(NOISE_FOLDER, "*.wav"))
            precomputed_noise_paths = [os.path.join(NOISE_FOLDER, f) for f in noise_files]
            data = add_noise(data, precomputed_noise_paths)

        if resample:
            data = tf_resample_audio(data)
            sample_rate = 8000

        if logmel or mfcc:
            feature = compute_log_mel_spectrogram(data, sample_rate)

            if mfcc:
                feature = compute_mfccs(feature)
            else:
                feature = feature[..., tf.newaxis]
        else:
            feature = get_spectrogram(data, sample_rate)

        return feature, label
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        placeholder_feature_shape = (128, 128, 1) 
        placeholder_feature = tf.zeros(placeholder_feature_shape)
        return placeholder_feature, label




