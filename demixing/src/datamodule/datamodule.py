import pickle
import torch
from torch.utils.data import Dataset
from os.path import dirname
from pathlib import Path
from parse_config import parse_config
from utils.utils import compute_mfcc
from model.transforms import MFCC


MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()


class StepOneV1(Dataset):

    def __init__(self, data_root, config, state='train'):
        super().__init__()
        with open(data_root, 'rb') as f:
            step_one_data = pickle.load(f)
        if state == 'train':
            data = step_one_data['train']
        elif state == 'test':
            data = step_one_data['test']
        self.wavforms = [d[0] for d in data]
        self.speakers = [d[1] for d in data]
        self.mfccfeat, _ = self.mfcc_and_label(self.wavforms, n_mfcc=config['mfcc'], sr=config['sr'])

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, index):
        mfcc = torch.tensor(self.mfccfeat[index, :, :], dtype=torch.float32)
        speaker = torch.tensor(self.speakers[index], dtype=torch.long)
        return mfcc, speaker

    @staticmethod
    def mfcc_and_label(wavforms, speakers=None, n_mfcc=20, sr=16000):
        mfccs = compute_mfcc(wavforms, n_mfcc, sr)
        return mfccs, speakers


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