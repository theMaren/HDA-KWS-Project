import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import resample
from python_speech_features import logfbank, mfcc, delta


def read_wavfile(file_path):
    _, data = wavfile.read(file_path)
    return data.squeeze()


def perform_padding(data, data_length=16000):
    padding_amount = data_length - len(data)
    pad_before = padding_amount // 2
    pad_after = padding_amount - pad_before
    return np.pad(data, pad_width=(pad_before, pad_after), mode='constant', constant_values=0)


def add_noise(data, dataset_path, noise_strength=None, noise_file=None):
    #DATASET_PATH = r'C:\Users\maren\OneDrive\HDA_Project\project_data'
    noise_strength = noise_strength or np.random.choice(np.arange(0.2, 0.55, 0.05))
    noise_list = ["doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav", "pink_noise.wav",
                  "running_tap.wav", "white_noise.wav"]
    noise_file = noise_file or random.choice(noise_list)
    noise_data = read_wavfile(os.path.join(dataset_path, "_background_noise_", noise_file))
    start_index = np.random.randint(0, len(noise_data) - len(data))
    noise_segment = noise_data[start_index:start_index + len(data)]
    return (data + noise_strength * noise_segment).astype(np.float32)


def resample_data(sample_data, original_sample_rate=16000, new_sample_rate=8000):
    num_samples = int(len(sample_data) * new_sample_rate / original_sample_rate)
    return resample(sample_data, num_samples)


def get_spectrogram(signal, samplerate, winlen=25, winstep=10, nfft=512, winfunc=tf.signal.hamming_window):
    spectrogram = tf.signal.stft(signal.astype(float), int(samplerate * winlen / 1000),
                                 int(samplerate * winstep / 1000), nfft, winfunc)
    spectrogram = tf.abs(spectrogram)

    #return spectrogram.astype(np.float32)
    return np.log(np.array(spectrogram).T + np.finfo(float).eps).astype(np.float32)


def get_logfbank(signal, samplerate, winlen=25, winstep=10, nfilt=40, nfft=512, lowfreq=300, highfreq=None):
    highfreq = highfreq or samplerate / 2
    logfbank_feat = logfbank(signal, samplerate, winlen / 1000, winstep / 1000, nfilt, nfft, lowfreq, highfreq).T
    return logfbank_feat.astype(np.float32)

def get_mfcc( signal,samplerate, delta_order=2, delta_window=1, winlen=25, winstep=10,
              numcep=13, nfilt=40, nfft=512, lowfreq=300, highfreq=None,
              appendEnergy=True, winfunc=np.hamming):

    if highfreq is None:
        highfreq = samplerate / 2

    features = []

    # Extract MFCC features
    mfcc_feat = mfcc(signal,samplerate,winlen/1000,winstep/1000,numcep,nfilt,nfft,lowfreq,highfreq,
                     appendEnergy=appendEnergy,winfunc=winfunc)

    mfcc_feat = mfcc_feat.T
    features.append(mfcc_feat)

    for i in range(delta_order):
        features.append(delta(features[-1], delta_window))

    full_feat = np.vstack(features)

    return full_feat.astype(np.float32)



def preprocess(data_path, directory, resample=True, logfbank=True, mfcc=True):
    directory_path = os.path.join(data_path, directory)
    data_records = []

    for dirpath, _, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            data = read_wavfile(file_path)
            data_padded = perform_padding(data)
            data_noisy = add_noise(data_padded,data_path)
            sample_rate = 16000

            if resample:
                data_noisy = resample_data(data_noisy)
                sample_rate = 8000

            if logfbank:
                features = get_logfbank(data_noisy, sample_rate)
            elif mfcc:
                features = get_mfcc(data_noisy, sample_rate)
            else:
                features = get_spectrogram(data_noisy, sample_rate)

            command_label = os.path.basename(dirpath)
            data_records.append({'Command': command_label, 'Filename': filename, 'Spectrogram': features})

    print("Done preprocessing")
    return pd.DataFrame(data_records)


