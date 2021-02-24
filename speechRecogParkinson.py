#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 23:26:44 2021

@author: oredata
"""

import os
import pathlib
import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

hc24_data_dir = '/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/ReadText/HC/'
pd24_data_dir = '/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/ReadText/PD/'

def convertAllFilesInDirectoryTo16Bit(directory, label):
    for file in os.listdir(directory):
         if(file.endswith('.wav')):
             nameSolo = file.rsplit('.', 1)[0]
             print(directory + nameSolo )
             data, samplerate = sf.read(directory + file)
             if label=='PD':
                 sf.write('/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/ReadText/PD16/' \
                               + nameSolo + '.wav', data, samplerate, subtype='PCM_16')
             else:
                 sf.write('/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/ReadText/HC16/' \
                               + nameSolo + '.wav', data, samplerate, subtype='PCM_16')

convertAllFilesInDirectoryTo16Bit(hc24_data_dir, 'HC')
convertAllFilesInDirectoryTo16Bit(pd24_data_dir, 'PD')

hc_data_dir = pathlib.Path('/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/ReadText/HC16/')
pd_data_dir = pathlib.Path('/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/ReadText/PD16/')

hc_filenames = tf.io.gfile.glob(str(hc_data_dir) + '/*')
hc_num_samples = len(hc_filenames)
print('Number of healthy subjects :', hc_num_samples)

pd_filenames = tf.io.gfile.glob(str(pd_data_dir) + '/*')
pd_num_samples = len(pd_filenames)
print('Number of PD subjects:', pd_num_samples)

train_files = pd_filenames[:int(math.ceil(0.8*pd_num_samples))]
train_files.extend(hc_filenames[:int(math.ceil(0.8*hc_num_samples))])
test_files = pd_filenames[-int(math.floor(0.2*pd_num_samples)):]
test_files.extend(hc_filenames[-int(math.floor(0.2*hc_num_samples)):])

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2]

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

def get_spectrogram(waveform, max_sample_size):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([max_sample_size] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

from IPython import display

max_sample_size=0
for waveform,_ in waveform_ds.take(30):
    if waveform.shape[0]>max_sample_size:
        max_sample_size=waveform.shape[0]
    
for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform, max_sample_size)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)
print('Audio playback')
display.display(display.Audio(waveform, rate=16000))

label_names = ['HC16','PD16']

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio, max_sample_size)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == label_names)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  X = np.arange(70144*129, step=height)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)
  
rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
  ax.set_title(label_names[label_id.numpy()])
  ax.axis('off')

plt.show()


def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(label_names)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32), 
    norm_layer,
    #layers.Conv2D(32, 3, activation='relu'),
    #layers.Conv2D(64, 3, activation='relu'),
    #layers.MaxPooling2D(),
    #layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(16, activation='relu'),
    #layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 15
history = model.fit(
    train_ds,  
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')