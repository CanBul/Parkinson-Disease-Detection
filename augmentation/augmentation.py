import numpy as np 
import os 
import librosa

class Augmentation():
    def __init__(self, data_path, maxShift=1, maxPadding=.5):
        self.sample, self.rate = librosa.load(data_path)
        self.whiteNoise = np.random.choice([True, False])
        self.shiftOrPadding = np.random.choice([None, 'leftShift', 'rightShift', 'randomPadding'])

        self.maxShift= maxShift
        self.maxPadding = maxPadding

    def process(self, test=False):
        if test:
            img = self.bw_spectrogram_image()
            return img

        if self.shiftOrPadding == 'leftShift':
            self.shiftLeft()
        
        elif self.shiftOrPadding == 'rightShift':
            self.shiftRight()
        
        elif self.shiftOrPadding == 'randomPadding':
            self.randomPadding()
        
        if self.whiteNoise:
            self.addWhiteNoise()

        img = self.bw_spectrogram_image()
        
        return img
    
    def shiftLeft(self):

        shift = np.random.randint(0, self.rate*self.maxShift) * -1

        self.sample = np.roll(self.sample, shift)
        self.sample[shift:] = 0
    
    def shiftRight(self):

        shift = np.random.randint(0, self.rate*self.maxShift)

        self.sample = np.roll(self.sample, shift)
        self.sample[:shift] = 0

    def randomPadding(self):
        
        startPosition =  np.random.randint(0,len(self.sample))
        paddingLength = np.random.randint(self.rate*self.maxPadding)

        self.sample[startPosition:startPosition+paddingLength] = 0
    
    def addWhiteNoise(self):
        coefs= [0.0001, 0.01]
        
        noise = np.random.normal(0,1, len(self.sample))

        self.sample += noise * np.random.uniform(coefs[0], coefs[1])

    def scale_minmax(self,X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def bw_spectrogram_image(self):
        
        # use log-melspectrogram
        mels = librosa.feature.melspectrogram(self.sample, sr=self.rate, n_mels=128)
        mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

        # min-max scale to fit inside 8-bit range
        img = self.scale_minmax(mels, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        img = 255 - img  # invert. make black==more energy
        img = img.astype('float32')
        img /= 255 
        
        return img    

        
        
        

