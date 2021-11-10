import pickle
import torch
from torch.utils.data import Dataset
from os.path import dirname
from pathlib import Path
from read import parse_config
from utils import compute_mfcc
MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()


class StepOneDataset(Dataset):

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
        mfcc = torch.tensor(self.mfccfeat[index], dtype=torch.float32)
        speaker = torch.tensor(self.speakers[index], dtype=torch.long)
        return mfcc, speaker

    @staticmethod
    def mfcc_and_label(wavforms, speakers=None, n_mfcc=20, sr=16000):
        mfccs = compute_mfcc(wavforms, n_mfcc, sr)
        return mfccs, speakers