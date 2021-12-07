import os
import torch
import librosa
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from os.path import dirname
from pathlib import Path
from tqdm.auto import tqdm
from demixing import utils
from demixing import parse_config
from sklearn.model_selection import train_test_split
MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()
utils.seed_everything(seed=914)


def read_timit_speaker_mapping():
    store_path = os.path.join(MODULE_PATH, 'data', 'processed')
    train_file_path = os.path.join(store_path, 'timit.speaker.train.txt')
    test_file_path = os.path.join(store_path, 'timit.speaker.test.txt')
    with open(train_file_path, 'r') as f:
        train_lst = [line.strip().split('\t') for line in f]
    with open(test_file_path, 'r') as f:
        test_lst = [line.strip().split('\t') for line in f]
    return train_lst, test_lst


def joint_different_speakers(audio_files, speakers, num_mix):
    """
    Parameters
    ----------
    audio_files: list
    speakers: list
    num_mix: int
    """
    mixed = []
    for i, file in enumerate(audio_files):
        current_speaker = speakers[i]
        different_speakers = list(map(lambda x: x!=current_speaker, speakers))
        different_speakers_idx = [i for i, spk in enumerate(different_speakers) if spk]
        select_idx = list(np.random.choice(different_speakers_idx, num_mix, replace=False))
        for j in select_idx:
            mixed.append([(file, audio_files[j]), (current_speaker, speakers[j])])
    return mixed


def mix_speakers_by_snr(audio_files, speakers, num_mix, sample_rate, slice_dur, snr):
    mixed = joint_different_speakers(audio_files, speakers, num_mix)
    mixed_data = []
    for data in tqdm(mixed, leave=False):
        trg_audio_file, itf_audio_file = data[0]
        trg_speaker, itf_speaker = data[1]

        trg_speaker_wav, _ = librosa.load(trg_audio_file, sr=sample_rate)
        itf_speaker_wav, _ = librosa.load(itf_audio_file, sr=sample_rate)

        # Calculate the scale to mix two speakers based on fixed SNR
        itf_speaker_power = np.mean(np.square(trg_speaker_wav)) / (10**(snr/10))
        scale = np.sqrt(itf_speaker_power / np.mean(np.square(itf_speaker_wav)))

        # Mix two speakers
        trg_speaker_length, itf_speaker_length = len(trg_speaker_wav), len(itf_speaker_wav)
        if trg_speaker_length == itf_speaker_length:
            speakers_mix_wav = trg_speaker_wav + scale * itf_speaker_wav
        elif trg_speaker_length > itf_speaker_length:
            itf_speaker_wav_aug = utils.pad_sequences(itf_speaker_wav, 
                                                      maxlen=len(trg_speaker_wav), 
                                                      padding='zero', 
                                                      batch=False)
            speakers_mix_wav = trg_speaker_wav + scale * itf_speaker_wav_aug
        elif trg_speaker_length < itf_speaker_length:
            speakers_mix_wav = trg_speaker_wav + scale * itf_speaker_wav[:len(trg_speaker_wav)]

        if len(speakers_mix_wav) <= sample_rate * slice_dur:
            speakers_mix_wav = utils.pad_sequences(speakers_mix_wav, maxlen=sample_rate*slice_dur*2, padding='zero', batch=False)

        mixed_data.append([speakers_mix_wav, trg_speaker, itf_speaker])
    return mixed_data


class ChannelDemixingDataset(Dataset):

    def __init__(self, audio_files, speakers, num_mix, slice_dur, sr, n_mfcc, snr):
        super().__init__()
        self.audio_files = audio_files
        self.speakers = speakers
        self.num_mix = num_mix
        self.slice_dur = slice_dur
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.snr = snr

        self.mixed_audio_mfcc, self.trg_speaker, self.itf_speaker = self._create_data()
        self.mixed_audio_mfcc = torch.tensor(self.mixed_audio_mfcc, dtype=torch.float)
        self.labels = torch.tensor(self.speakers, dtype=torch.long)
        
    def _create_data(self):
        mixed_data = mix_speakers_by_snr(
            self.audio_files, self.speakers, self.num_mix, self.sr, self.slice_dur, self.snr
        )
        mixed_audio = [self._random_slice(data[0]) for data in mixed_data]
        trg_speaker = [data[1] for data in mixed_data]
        itf_speaker = [data[2] for data in mixed_data]
        mixed_audio_mfcc = utils.compute_mfcc(mixed_audio, n_mfcc=self.n_mfcc, sr=self.sr)
        return mixed_audio_mfcc, trg_speaker, itf_speaker

    def _random_slice(self, audio):
        slice_len = int(self.slice_dur * self.sr)
        diff = len(audio) - slice_len
        start = np.random.randint(diff)
        return audio[start:start+slice_len]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mfcc = self.mixed_audio_mfcc[index]
        label = self.labels[index]
        return mfcc, label


def main():
    train_lst, test_lst = read_timit_speaker_mapping()
    audio_files = [ele[0] for ele in train_lst] + [ele[0] for ele in test_lst]
    labels = [ele[1] for ele in train_lst] + [ele[1] for ele in test_lst]
    classes = list(set(labels))
    class2index = {cls:idx for idx, cls in enumerate(classes)}
    labels = list(map(class2index.get, labels))

    X_train, X_test, y_train, y_test = train_test_split(audio_files, labels, test_size=CONFIG['test_split'], random_state=914, stratify=labels)
    print(f'Training size: {len(y_train)}')
    print(f'Testing size: {len(y_test)}')

    train_ds = ChannelDemixingDataset(X_train, y_train, num_mix=1, sr=16000, slice_dur=1, n_mfcc=20, snr=5)
    print(f'Training size: {len(train_ds)}')
    print(train_ds[0:2])


if __name__ == '__main__':
    main()