import torch
import librosa
import torch.nn as nn
import pytorch_lightning as pl
from model.transforms import MFCC
from model.xvector import XVector
from engine.engine import ClassificationEngine
from datamodule.datamodule import StepOneV2
from parse_config import parse_config
from torch.utils.data import ConcatDataset, DataLoader, random_split


CONFIG = parse_config()


class StepOneModel(nn.Module):

    def __init__(self, sample_rate=16000, n_mfcc=20, dropout_p=0.2):
        super(StepOneModel, self).__init__()
        xvector = XVector(n_mfcc=n_mfcc, dropout_p=dropout_p)
        mfcc = MFCC(
            sample_rate=sample_rate, 
            n_mfcc=n_mfcc, 
            melkwargs={
                'n_fft':int(sample_rate*0.025), 
                'hop_length':int(sample_rate*0.01), 
                'n_mels':80
            }
        )

    def forward(self, x):
        x = mfcc(x)[:, :, :-1]
        x = xvector(x)
        return x


def main():
    #################
    ### Load data ###
    #################

    trn_seg = CONFIG['train_segments']
    tst_seg = CONFIG['test_segments']
    path = f'./data/processed/trn{trn_seg}_tst{tst_seg}.pkl'
    train_ds = StepOneV2(path, CONFIG, state='train')
    train_size = int(len(train_ds)*CONFIG['test_size'])
    valid_size = len(train_ds)-int(len(train_ds)*CONFIG['test_size'])
    train_ds, valid_ds = random_split(
        train_ds, 
        [train_size, valid_size]
    )
    test_ds = StepOneV2(path, CONFIG, state='test')
    train_dl = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True, 
        drop_last=True
    )
    valid_dl = DataLoader(
        valid_ds, 
        batch_size=CONFIG['batch'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True, 
        drop_last=True
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=CONFIG['batch'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True, 
        drop_last=False
    )

    ###################
    ### Build model ###
    ###################

    model = ClassificationEngine(
        XVector(n_mfcc=CONFIG['mfcc'], dropout_p=CONFIG['dropout'])
    )
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else None), 
        deterministic=True, 
        max_epochs=CONFIG['epoch'], 
        precision=(16 if torch.cuda.is_available() else 32), 
        num_sanity_val_steps=0, 
        fast_dev_run=CONFIG['debug']
    )
    trainer.fit(model, train_dl, valid_dl)
    if CONFIG['plot']:
        model.plot()


if __name__ == '__main__':
    main()