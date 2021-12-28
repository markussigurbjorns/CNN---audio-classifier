import torch
import torchaudio
from torch import nn

from torch.utils.data import Dataset

import os
import pandas as pd



class UniversalSoundDataset(Dataset):
    
    def __init__(self, annotations_file, audio_dir, transformation, sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    
    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        singal = self._cut_sample(signal)
        signal = self._zero_pad(singal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_sample(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:,:self.num_samples]
        return signal
    
    def _zero_pad(self, signal):
        if signal.shape[1] < self.num_samples:
            missing_samples = self.num_samples - signal.shape[1]
            dim_padding = (0, missing_samples)
            signal = torch.nn.functional.pad(signal, dim_padding)
        return signal
    
    def _resample(self, signal, sr):
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down(self, signal):  
        if signal.shape[0] > 1: 
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
        