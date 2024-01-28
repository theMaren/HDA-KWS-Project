import os
import pathlib
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import resample
from python_speech_features import logfbank
from python_speech_features import mfcc, delta



def read_wavfile(file_path):
    _, data  = wavfile.read(file_path)
    data.squeeze()
    return data


def perform_padding(data, data_length=16000):
    data_shape = data.shape[0]
    # padding
    if data_shape < data_length:
        tot_pad = data_length - data_shape
        pad_before = int(np.ceil(tot_pad / 2))
        pad_after = int(np.floor(tot_pad / 2))
        data = np.pad(data, pad_width=(pad_before, pad_after), mode='constant', constant_values=0)

    return data


def add_noise(data, noise_strength=None, noise_file=None):
    DATASET_PATH = r'C:\Users\maren\OneDrive\HDA_Project\project_data'

    if not noise_strength:
        noise_strength = np.random.choice(np.arange(0.2, 0.55, 0.05))

    if not noise_file:
        noise_list = ["doing_the_dishes.wav", "dude_miaowing.wav", "exercise_bike.wav", "pink_noise.wav",
                      "running_tap.wav", "white_noise.wav"]
        noise_file = random.choice(noise_list)

    noise_data = read_wavfile(os.path.join(DATASET_PATH, "_background_noise_", noise_file))

    target_size = data.shape[0]
    noise_size = noise_data.shape[0]
    start = np.random.randint(0, int(noise_size - target_size))
    end = start + target_size
    noise_data = noise_data[start:end]

    # add noise to input audio
    data_with_noise = data + noise_strength * noise_data

    return data_with_noise.astype(np.float32)

#adjust
def resample(sample_data, sample_rate = 16000,new_sample_rate = 8000):
    resampled_data = resample(sample_data, int(new_sample_rate / sample_rate * sample_data.shape[0]))
    return resampled_data

def get_spectrogram(
        signal,  # audio signal from which to compute features (N*1 array)
        samplerate=16000,  # samplerate of the signal we are working with
        winlen=25,  # length of the analysis window (milliseconds)
        winstep=10,  # step between successive windows (milliseconds)
        nfft=512,  # FFT size
        winfunc=tf.signal.hamming_window  # analysis window to apply to each frame
):
    # Convert the waveform to a spectrogram via a STFT
    spectrogram = tf.signal.stft(
        signal.astype(float),
        int(samplerate * winlen / 1000),
        int(samplerate * winstep / 1000),
        nfft,
        winfunc
    )

    spectrogram = tf.abs(spectrogram)

    spectrogram = np.array(spectrogram)

    spectrogram = np.log(spectrogram.T + np.finfo(float).eps)

    return spectrogram.astype(np.float32)


def get_logfbank(
                signal,             # audio signal from which to compute features (N*1 array)
                samplerate = 16000, # samplerate of the signal we are working with
                winlen     = 25,    # length of the analysis window (milliseconds)
                winstep    = 10,    # step between successive windows (milliseconds)
                nfilt      = 40,    # number of filters in the filterbank
                nfft       = 512,   # FFT size
                lowfreq    = 300,   # lowest band edge of mel filters (Hz)
                highfreq   = None,  # highest band edge of mel filters (Hz)
                ):
    if highfreq is None:
        highfreq = samplerate / 2

    # Extract log Mel-filterbank energy features
    logfbank_feat = logfbank(
                            signal,
                            samplerate,
                            winlen/1000,
                            winstep/1000,
                            nfilt,
                            nfft,
                            lowfreq,
                            highfreq,
                            )
    logfbank_feat = logfbank_feat.T

    return logfbank_feat.astype(np.float32)


def get_mfcc(
            signal,                    # audio signal from which to compute features (N*1 array)
            delta_order  = 2,          # maximum order of the Delta features
            delta_window = 1,          # window size for the Delta features
            samplerate   = 16000,      # samplerate of the signal we are working with
            winlen       = 25,         # length of the analysis window (milliseconds)
            winstep      = 10,         # step between successive windows (milliseconds)
            numcep       = 13,         # number of cepstrum to return
            nfilt        = 40,         # number of filters in the filterbank
            nfft         = 512,        # FFT size
            lowfreq      = 300,        # lowest band edge of mel filters (Hz)
            highfreq     = None,       # highest band edge of mel filters (Hz)
            appendEnergy = True,       # if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy
            winfunc      = np.hamming  # analysis window to apply to each frame
            ):

    if highfreq is None:
        highfreq = samplerate / 2

    features = []

    # Extract MFCC features
    mfcc_feat = mfcc(
                    signal,
                    samplerate,
                    winlen/1000,
                    winstep/1000,
                    numcep,
                    nfilt,
                    nfft,
                    lowfreq,
                    highfreq,
                    appendEnergy=appendEnergy,
                    winfunc=winfunc
                    )
    mfcc_feat = mfcc_feat.T
    features.append(mfcc_feat)

    # Extract Delta features
    for i in range(delta_order):

        features.append(delta(features[-1], delta_window))

    # Full feature vector
    full_feat = np.vstack(features)

    return full_feat.astype(np.float32)



def preprocess(data_path,directory):
    directory_path = os.path.join(data_path, directory)  # Replace with the path to your directory
    data_records = []  # List to store the records

    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            print(f"Processing file: {file_path}")
            #Information to be stored, spectogram and label(subdirectory (as dataframe?)
            #read file
            data = read_wavfile(file_path)
            #padding
            data_pad = perform_padding(data, data_length=16000)
            #adding noise
            data_noise = add_noise(data_pad)
            #resample
            data_res = resample(data_noise)
            #spectogram (add the other methods)
            spectrogram = get_spectrogram(data_res)

            # Extract the command label from the directory name
            command_label = os.path.basename(dirpath)
            # Append the information to the list
            data_records.append({
                'Command': command_label,
                'Filename': filename,
                'Spectrogram': spectrogram
            })

        df = pd.DataFrame(data_records)
        return df
