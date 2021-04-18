from pydub import AudioSegment
import soundfile as sf
import numpy as np
import math
import os
import pathlib

#Convert audio files from 24 bit to 16 bit for King College dataset
r_hc24_data_dir = '/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/ReadText/HC/'
r_pd24_data_dir = '/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/ReadText/PD/'
s_hc24_data_dir = '/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/SpontaneousDialogue/HC/'
s_pd24_data_dir = '/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/SpontaneousDialogue/PD/'

def convertAllFilesInDirectoryTo16Bit(directory, dir_type, label):
    for file in os.listdir(directory):
         if(file.endswith('.wav')):
             nameSolo = file.rsplit('.', 1)[0]
             print(directory + nameSolo )
             data, samplerate = sf.read(directory + file)
             if label=='PD':
                 sf.write('/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/'+dir_type+'/PD16/' \
                               + nameSolo + '.wav', data, samplerate, subtype='PCM_16')
             else:
                 sf.write('/Users/oredata/Desktop/parkinson/26-29_09_2017_KCL/'+dir_type+'/HC16/' \
                               + nameSolo + '.wav', data, samplerate, subtype='PCM_16')

convertAllFilesInDirectoryTo16Bit(r_hc24_data_dir, 'ReadText', 'HC')
convertAllFilesInDirectoryTo16Bit(r_pd24_data_dir, 'ReadText','PD')
convertAllFilesInDirectoryTo16Bit(s_hc24_data_dir, 'SpontaneousDialogue', 'HC')
convertAllFilesInDirectoryTo16Bit(s_pd24_data_dir, 'SpontaneousDialogue','PD')

#Split audio files into chunks
class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
        self.max_chunk_long=0
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def cut_start_point(self, cut_second): #cut initial seconds from audio
        self.audio = self.audio[cut_second*1000:]
    
    def addSilencePadding(self,silence_duration): #ms
        self.audio = self.audio + AudioSegment.silent(duration=silence_duration)
        self.audio.export(self.filepath, format="wav")
    
    def single_split(self, from_sec, to_sec, split_filename):
        t1 = from_sec * 1000
        t2 = to_sec * 1000
        split_audio = self.audio[t1:t2]
        long=t2-t1
        if long>self.max_chunk_long:
            self.max_chunk_long=long
        split_audio.export(self.folder + split_filename, format="wav")
        
    def multiple_split(self, chunks): #split with chunks
        total_secs = self.get_duration()
        for ind, i in zip(range(chunks),np.linspace(0, total_secs, chunks+1)[:-1]):
            split_fn = 'chunk' +str(ind+1) + '_' + self.filename
            self.single_split(i, i+(total_secs/chunks), split_fn)
            print(str(i) + ' Done')
            if i == total_secs - (total_secs/chunks):
                print('All splitted successfully')
                
    def multiple_split2(self, sec_per_split): #split into equal times
        total_secs = math.ceil(self.get_duration())
        for ind, i in zip(range(len(range(0, total_secs, sec_per_split))),range(0, total_secs, sec_per_split)):
            split_fn = 'chunk'+str(ind+1) + '_' + self.filename
            self.single_split(i, i+sec_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_secs - sec_per_split:
                print('All splited successfully')

start_cut = 15 #seconds

sec_per_split=2 #split audio files into 2 secs

max_c_long=0
def splitAudioChunks(directory, long):
    for file in os.listdir(directory):
         if(file.endswith('.wav')):
             split_wav = SplitWavAudioMubin(directory[:-1]+'16', file)
             split_wav.cut_start_point(start_cut) 
             split_wav.multiple_split2(sec_per_split)
             if split_wav.max_chunk_long > long:
                 long=split_wav.max_chunk_long
    return long
             
max_c_long=splitAudioChunks(r_hc24_data_dir, max_c_long)
max_c_long=splitAudioChunks(r_pd24_data_dir,max_c_long)
max_c_long=splitAudioChunks(s_hc24_data_dir,max_c_long)
max_c_long=splitAudioChunks(s_pd24_data_dir,max_c_long)

#Add silence padding to equalize chunks
def addSilencePadding(directory):
    for file in os.listdir(directory):
         if(file.endswith('.wav')):
             split_wav = SplitWavAudioMubin(directory, file)
             duration = split_wav.get_duration()*1000
             if max_c_long-duration>0:
                 split_wav.addSilencePadding(max_c_long-duration)

addSilencePadding(r_hc24_data_dir[:-1]+'16C')
addSilencePadding(r_pd24_data_dir[:-1]+'16C')
addSilencePadding(s_hc24_data_dir[:-1]+'16C')
addSilencePadding(s_pd24_data_dir[:-1]+'16C')


#split italian dataset into chunks

hc_y_data_dir = '/Users/oredata/Desktop/parkinson/Italian Parkinson Voice and speech/15 Young Healthy Control/'
hc_e_data_dir = '/Users/oredata/Desktop/parkinson/Italian Parkinson Voice and speech/22 Elderly Healthy Control/'
pd_data_dir = '/Users/oredata/Desktop/parkinson/Italian Parkinson Voice and speech/28 People with Parkinson disease/'

start_cut = 0 #seconds

sec_per_split=2

max_c_long=0
def splitAudioChunks(directory, long):
    p=Path(directory)
    for idfile in os.listdir(directory):
        if (not idfile.endswith(".xlsx")) & (idfile!='.DS_Store'):
            for file in os.listdir(directory + idfile):
                if(file.endswith('.wav')):
                     split_wav = SplitWavAudioMubin(directory+idfile, file)
                     split_wav.folder = str(p.parent) + '/HC/' if p.parts[-1].split(" ")[-2]=="Healthy" else str(p.parent) + '/PD/'
                     split_wav.filename= idfile.replace(" ", "") + '_' + file
                     split_wav.multiple_split2(sec_per_split)
                     if split_wav.max_chunk_long > long:
                         long=split_wav.max_chunk_long
                else:
                    if (not file.endswith(".xlsx")) & (file!='.DS_Store'):
                        for subfile in os.listdir(directory + idfile + '/' +file):
                            if(subfile.endswith('.wav')):
                                split_wav = SplitWavAudioMubin(directory+idfile+ '/' +file, subfile)
                                split_wav.folder = str(p.parent) + '/HC/' if p.parts[-1].split(" ")[-2]=="Healthy" else str(p.parent) + '/PD/'
                                split_wav.filename= file.replace(" ", "") + '_' + subfile 
                                split_wav.multiple_split2(sec_per_split)
                                if split_wav.max_chunk_long > long:
                                    long=split_wav.max_chunk_long
    return long
             
max_c_long=splitAudioChunks(hc_y_data_dir, max_c_long)
max_c_long=splitAudioChunks(hc_e_data_dir,max_c_long)
max_c_long=splitAudioChunks(pd_data_dir,max_c_long)

#Add silence padding to equalize chunks
def addSilencePadding(directory):
    for file in os.listdir(directory):
         if(file.endswith('.wav')):
             split_wav = SplitWavAudioMubin(directory, file)
             duration = split_wav.get_duration()*1000
             if max_c_long-duration>0:
                 split_wav.addSilencePadding(max_c_long-duration)

hc_data_dir='/Users/oredata/Desktop/parkinson/Italian Parkinson Voice and speech/HC/'
pd_data_dir='/Users/oredata/Desktop/parkinson/Italian Parkinson Voice and speech/PD/'
  
addSilencePadding(hc_data_dir)
addSilencePadding(pd_data_dir)
 
