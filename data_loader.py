import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import pandas as pd
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
import random
import re
from itertools import cycle
'''
TODO
audio path when training on all 3 datasets
add argument to arg parser

'''

class COVID_dataset(Dataset):
    '''
    Custom COVID dataset.
    '''
    PATHS = {
        'coswara':'/vol/bitbucket/hgc19/covid_cross_datasets/content/coswara/splits/list_coswara_{dset}.txt',
        'epfl':'/vol/bitbucket/hgc19/covid_cross_datasets/content/epfl/splits/list_epfl_{dset}.txt',
        'comapre': '/vol/bitbucket/hgc19/COMPARE_data/cough/lab/{dset}.csv'

    }
    def __init__(self, dset,
                 eval_type='random',
                 transform=None,
                 window_size=1,
                 sample_rate=48000,
                 hop_length=512,
                 n_fft=2048,
                 masking=False,
                 pitch_shift=False, 
                 modality='breathcough',
                 kdd=False,
                 feature_type='stft',
                 n_mfcc=40,
                 onset_sample_method=False,
                 repetitive_padding = False,
                 dataset='coswara'):
        if dataset == 'epfl' or dataset == 'coswara':
            path = '/vol/bitbucket/hgc19/covid_cross_datasets/content'
            file_path = os.path.join(path,
                                dataset,
                                'splits',
                                'list_'+str(dataset)+'_'+str(dset)+'.txt')
            self.audio_dir = os.path.join(path, dataset, 'cough')
        elif dataset == 'compare':
            path = '/vol/bitbucket/hgc19/COMPARE_data/cough'
            file_path = os.path.join(path,
                                'lab',
                                str(dset) + '.csv')
            self.audio_dir = os.path.join(path, 'wav')

        elif dataset == 'all':
            li = []
            for name, path in PATHS.items():
                df = pd.read_csv(file_path,
                            names=['file', 'label'],
                            delimiter=',' if name == 'compare' else ' ',
                            skiprows=1 if name == 'compare' else 0)
                li.append(df)
            metadata = pd.concat(li, axis=0, ignore_index=True)
        else:
            raise 'This should not happen, investigate!'

    
        metadata = pd.read_csv(file_path,
                            names=['file', 'label'],
                            delimiter=',' if dataset == 'compare' else ' ',
                            skiprows=1 if dataset == 'compare' else 0)
        print(metadata)
        
        train_fold = metadata['file'].to_list()
        metadata = metadata.set_index('file')

        self.window_size = window_size * sample_rate
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.transform = transform
        self.eval_type = eval_type
        self.masking = masking
        self.pitch_shift = pitch_shift
        self.modality = modality
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.onset_sample_method = onset_sample_method
        self.repetitive_padding = repetitive_padding
        self.path = path
        self.train_fold = train_fold
        self.metadata = metadata
        self.dset = dset
        self.dataset = dataset

    
    def __len__(self):
        return len(self.train_fold)

    def custom_transform(self, signal):
        """
        create log spectrograph of signal
        """
        if self.feature_type == 'stft':
            stft = librosa.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length)
            spectrogram = np.abs(stft)
            features = librosa.amplitude_to_db(spectrogram)
        if self.feature_type == 'mfcc':
            features = librosa.feature.mfcc(signal, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        if self.masking:
            features = self.spec_augment(features)
        if self.transform:
            features = self.transform(features)
        return features

    def pad(self, signal):
        sample_signal = np.zeros((self.window_size,))
        sample_signal[:signal.shape[0],] = signal
        return sample_signal

    def pad_repetitive(self, signal):
        rpt_cnt = np.ceil(self.window_size / len(signal))
        signal = np.tile(signal, int(rpt_cnt))
        sample_signal = signal[:self.window_size, ]
        return sample_signal

    def __getitem__(self, index):

        # get path of chosen index

        audio_path = self.train_fold[index]
        cov_pos = 'positive' if self.dataset == 'compare' else 'p'
        cov_neg = 'negative' if self.dataset == 'compare' else 'n'
        if self.metadata.loc[audio_path, 'label'] == cov_pos:
            label = 1
        elif self.metadata.loc[audio_path, 'label'] == cov_neg:
            label = 0
        else:
            raise f"Error, {self.metadata.loc[audio_path, 'label']} is not a valid category"

        audio_path = os.path.join(self.audio_dir, audio_path)
        chunks = self.load_process(audio_path)

        return chunks, label



    def load_process(self, audio_path):
        # load the data
        signal, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        # perform pitch shift:
        if self.pitch_shift:
            step = np.random.uniform(-6,6)
            signal = librosa.effects.pitch_shift(
                signal, sample_rate, step)

        # For train, sample random window size from audiofile
        if self.dset == 'train' or self.eval_type != 'maj_vote':
            # Apply padding if necessary. Else sampsle random window.
            if self.onset_sample_method:
                onsets = librosa.onset.onset_detect(signal, units='time')
                onsets = onsets * self.sample_rate
                onsets = [int(i) for i in onsets]
                if len(onsets)==0:
                    sample_signal = signal
                else:
                    rand_onset = random.choice(onsets)
                    left_ind = int(rand_onset - (self.window_size/2))
                    right_ind = int(rand_onset + (self.window_size/2))
                    if rand_onset - (self.window_size/2) < 0:
                        left_ind=0
                    if rand_onset + (self.window_size/2) >= signal.shape[0]:
                        right_ind=signal.shape[0]-1

                    sample_signal = signal[left_ind:right_ind]
                if sample_signal.shape[0] <= self.window_size:
                    if self.repetitive_padding:
                        sample_signal = self.pad_repetitive(sample_signal)
                    else:
                        sample_signal = self.pad(sample_signal)

            else:

                if signal.shape[0] <= self.window_size:
                    if self.repetitive_padding:
                        sample_signal = self.pad_repetitive(signal)
                    else:
                        sample_signal = self.pad(signal)
                else:
                    if self.eval_type == 'random':
                        rand_indx = np.random.randint(0, signal.shape[0] - self.window_size)
                    else:
                        rand_indx = 0
                    sample_signal = signal[rand_indx:rand_indx + self.window_size]

            # perform transformations
            sample_signal = self.custom_transform(sample_signal)

            return sample_signal
        # For eval/test, chunk audiofile into chunks of size wsz and
        # process and return all
        else:
            if self.onset_sample_method:
                chunks = []
                onsets = librosa.onset.onset_detect(signal, units='time')
                onsets = onsets * self.sample_rate
                onsets = [int(i) for i in onsets]
                if len(onsets)==0:
                    sample_signal = signal
                    if sample_signal.shape[0] <= self.window_size:
                        if self.repetitive_padding:
                            sample_signal = self.pad_repetitive(sample_signal)
                        else:
                            sample_signal = self.pad(sample_signal)
                    sample_signal = self.custom_transform(sample_signal)
                    chunks.append(sample_signal)
                else:
                    for onset in onsets:
                        left_ind = int(onset - (self.window_size / 2))
                        right_ind = int(onset + (self.window_size / 2))
                        if onset - (self.window_size / 2) < 0:
                            left_ind = 0
                        if onset + (self.window_size / 2) >= signal.shape[0]:
                            right_ind = signal.shape[0] - 1

                        sample_signal = signal[left_ind:right_ind]
                        if sample_signal.shape[0] <= self.window_size:
                            if self.repetitive_padding:
                                sample_signal = self.pad_repetitive(sample_signal)
                            else:
                                sample_signal = self.pad(signal)
                        sample_signal = self.custom_transform(sample_signal)
                        chunks.append(sample_signal)

            else:
                chunks = np.array_split(signal, int(np.ceil(signal.shape[0] / self.window_size)))
                def process_chunk(chunk):
                    if chunk.shape[0] <= self.window_size:
                        if self.repetitive_padding:
                            sample_signal = self.pad_repetitive(chunk)
                        else:
                            sample_signal = self.pad(chunk)
                    chunk =  self.custom_transform(sample_signal)
                    return chunk
                chunks = [process_chunk(chunk) for chunk in chunks]

            return chunks
    

    def spec_augment(self,
                     spec: np.ndarray,
                     num_mask=2,
                     freq_masking_max_percentage=0.15,
                     time_masking_max_percentage=0.3):

        spec = spec.copy()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0,
                                   high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = 0

            time_percentage = random.uniform(0.0, time_masking_max_percentage)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0,
                                   high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = 0

        return spec



    def nth_repl(self, s, sub, repl, n):
        find = s.find(sub)
        # If find is not -1 we have found at least one match for the substring
        i = find != -1
        # loop util we find the nth or we find no match
        while find != -1 and i != n:
            # find + 1 means we start searching from after the last match
            find = s.find(sub, find + 1)
            i += 1
        # If i is equal to n we found nth match so replace
        if i == n:
            return s[:find] + repl + s[find + len(sub):]
        return s

    def overlap(self, list_1, list_2):
        '''
        sanity check that there is no leakage from test set into train/val
        '''
        overlap = [i for i in list_1 if i in list_2]

        if len(overlap) > 0:
            raise 'You have cross over between test and train!!!! Investigate'


if __name__ == "__main__":
    test_dataset = COVID_dataset('val', None)
    for i in tqdm(range(len(test_dataset))):
        sample, label = test_dataset[i]

        plt.figure()
        librosa.display.specshow(sample,
                                sr=24000,
                                hop_length=512)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram (dB)")
        path_to_save = 'figs/log_spectrogram'+str(i)+'.png'
        #plt.savefig(path_to_save)
        plt.show()
        plt.close()
        print(sample.shape)