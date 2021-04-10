import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

audio_paths = ['/content/drive/MyDrive/Parkinson/preprocessedSpeechFiles/data2sec/ReadText/HC16C/',
               '/content/drive/MyDrive/Parkinson/preprocessedSpeechFiles/data2sec/ReadText/PD16C/',
               '/content/drive/MyDrive/Parkinson/preprocessedSpeechFiles/data3sec/ReadText/PD16C/',
               '/content/drive/MyDrive/Parkinson/preprocessedSpeechFiles/data3sec/ReadText/HC16C/',
               '/content/drive/MyDrive/Parkinson/preprocessedSpeechFiles/data3sec/SpontaneousDialogue/PD16C/',
               '/content/drive/MyDrive/Parkinson/preprocessedSpeechFiles/data3sec/SpontaneousDialogue/HC16C/',
               ]  # Source paths

output_paths = ['/content/Parkinson-Disease-Detection/data/KGL/data2sec/RGBimages/ReadTextHCimages/',
                '/content/Parkinson-Disease-Detection/data/KGL/data2sec/RGBimages/ReadTextPDimages/',
                '/content/Parkinson-Disease-Detection/data/KGL/data3sec/RGBimages/ReadTextPDimages/',
                '/content/Parkinson-Disease-Detection/data/KGL/data3sec/RGBimages/ReadTextHCimages/',
                '/content/Parkinson-Disease-Detection/data/KGL/data3sec/RGBimages/SpontaneousDialoguePDimages/',
                '/content/Parkinson-Disease-Detection/data/KGL/data3sec/RGBimages/SpontaneousDialogueHCimages/'
                ]  # Output Paths

# Create folders if they don't exist
for path in output_paths:
    if not os.path.exists(path):
        os.makedirs(path)


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def bw_spectrogram_image(sample, FileName, outFolder, sampleRate):
    # Full Path
    filePath = outFolder + FileName[:-4]

    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(sample, sr=sampleRate, n_mels=128)
    mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
    img = 255 - img  # invert. make black==more energy
    img = np.stack((img,) * 3, axis=-1)  # 3 channel conversion

    # save as PNG
    skimage.io.imsave(filePath + '.png', img)

def rgb_spectrogram_image(sample, FileName, outFolder, sampleRate):
    # Full Path
    filePath = outFolder + FileName[:-4]
    #create a figure
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(sample, sr=sampleRate)
    librosa.display.specshow(librosa.power_to_db(mels, ref=np.max))
    #save as png
    plt.savefig(filePath + '.png', dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close('all')

def convertToSpectogram(audioPaths, outPaths, typ='RGB'):

    for i in range(len(audioPaths)):
        audioPath = audioPaths[i]
        outPath = outPaths[i]

        files = [f for f in os.listdir(audioPath) if f.endswith('.wav')]

        for f in files:

            sample, sampleRate = librosa.load(str(audioPath) + f)
            if typ=='RGB':
              rgb_spectrogram_image(sample, f, outPath, sampleRate)
            else:
              bw_spectrogram_image(sample, f, outPath, sampleRate)

convertToSpectogram(audio_paths, output_paths, 'RGB')
