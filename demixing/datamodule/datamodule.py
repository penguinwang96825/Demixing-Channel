import pickle
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
from demixing.utils.utils import compute_mfcc, seed_everything, pad_sequences
from demixing.model.transforms import MFCC
seed_everything(914)


class SpeakerIdDataset(Dataset):

    def __init__(self, audio_files, labels, sr=16000, n_mfcc=20, slice_dur=1, num_seg=1):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.slice_dur = slice_dur
        self.num_seg = num_seg

        self.mfcc, self.labels = self._create_data()
        self.mfcc = torch.tensor(self.mfcc, dtype=torch.float)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.mfcc = self._normalise_mfcc(self.mfcc)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mfcc = self.mfcc[index]
        label = self.labels[index]
        return mfcc, label

    def _create_data(self):
        wavforms, labels = [], []
        for file, label in zip(self.audio_files, self.labels):
            wavform, sr = librosa.load(file, sr=self.sr)
            for _ in range(self.num_seg):
                if len(wavform) <= sr * self.slice_dur:
                    wavform = pad_sequences(wavform, maxlen=sr*self.slice_dur*2, padding='zero', batch=False)
                    # break
                wavform_slice = self._random_slice(wavform)
                wavforms.append(wavform_slice)
                labels.append(label)
        mfcc = compute_mfcc(wavforms, n_mfcc=self.n_mfcc, sr=sr)
        return mfcc, labels

    def _random_slice(self, audio):
        slice_len = int(self.slice_dur * self.sr)
        diff = len(audio) - slice_len
        start = np.random.randint(diff)
        return audio[start:start+slice_len]

    def _normalise_mfcc(self, mfcc):
        mean_ = torch.mean(mfcc, dim=0)
        std_ = torch.std(mfcc, dim=0)
        return (mfcc - mean_) / std_


class StepOneV2(Dataset):

    def __init__(self, data_root, config, sample_rate=16000, n_mfcc=20, state='train'):
        super().__init__()
        with open(data_root, 'rb') as f:
            step_one_data = pickle.load(f)
        if state == 'train':
            data = step_one_data['train']
        elif state == 'test':
            data = step_one_data['test']
        self.wavforms = [d[0] for d in data]
        self.speakers = [d[1] for d in data]
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        
    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, index):
        wavform = torch.tensor(self.wavforms[index], dtype=torch.float32)
        mfcc = MFCC(
            sample_rate=self.sample_rate, 
            n_mfcc=self.n_mfcc, 
            melkwargs={
                'n_fft':int(self.sample_rate*0.025), 
                'hop_length':int(self.sample_rate*0.01), 
                'n_mels':80
            }
        )(wavform)[:, :-1]
        mfcc = mfcc.transpose(0, 1)
        speaker = torch.tensor(self.speakers[index], dtype=torch.long)
        return mfcc, speaker