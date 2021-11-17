import os
import re
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset
from os.path import dirname
from pathlib import Path
from parse_config import parse_config
from demixing.utils import compute_mfcc



MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()


class StepOneDataset(Dataset):

    def __init__(self, data_root_folder, config, state='train'):
        super().__init__()
        with open(data_root_folder, 'rb') as f:
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


class TIMIT(object):
    """
    /home/yangwang/Desktop/Demixing-Channel/demixing/src/data/TIMIT/
    """
    def __init__(self, timit_root_folder, store_path):
        super(TIMIT).__init__()
        self.timit_root_folder = timit_root_folder
        self.store_path = store_path

    def build(self):
        spk_files_train, spk_id_train = self.build_speaker_mapping(mode='TRAIN')
        spk_files_test, spk_id_test = self.build_speaker_mapping(mode='TEST')

        with open(f'{self.store_path}/timit.speaker.train.txt', 'w') as f:
            for file_path, spk in zip(spk_files_train, spk_id_train):
                to_write = f'{file_path}\t{spk}'
                f.write(f'{to_write}\n')
        with open(f'{self.store_path}/timit.speaker.test.txt', 'w') as f:
            for file_path, spk in zip(spk_files_test, spk_id_test):
                to_write = f'{file_path}\t{spk}'
                f.write(f'{to_write}\n')

    def build_speaker_mapping(self, mode='TRAIN'):
        root_folder = os.path.join(self.timit_root_folder, mode)
        timit_data = []
        for dialect_region in os.listdir(root_folder):
            dialect_region_dir_path = os.path.join(root_folder, dialect_region)
            for speaker_id in os.listdir(dialect_region_dir_path):
                speaker_id_dir_path = os.path.join(dialect_region_dir_path, speaker_id)
                for file in os.listdir(speaker_id_dir_path):
                    if file.endswith("WAV"):
                        id_ = file.split(".")[0]
                        sentence_type = re.findall("[A-Za-z]+", id_.strip())[0]
                        file_path = os.path.join(speaker_id_dir_path, file)
                        timit_data.append([
                            dialect_region, 
                            file_path, 
                            id_, 
                            sentence_type, 
                            speaker_id
                        ])
        timit_data = pd.DataFrame(
            timit_data, 
            columns=["dialect_region", "file", "id", "sentence_type", "speaker_id"]
        )
        timit_data = timit_data.sort_values("speaker_id").reset_index(drop=True)
        return timit_data['file'].tolist(), timit_data['speaker_id'].tolist()


if __name__ == '__main__':
    timit = TIMIT(
        timit_root_folder='/home/yangwang/Desktop/Demixing-Channel/demixing/data/TIMIT/', 
        store_path='/home/yangwang/Desktop/Demixing-Channel/demixing/data/processed'
    )
    timit.build()