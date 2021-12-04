import os
import torch
import numpy as np
from sklearn import metrics
from os.path import dirname
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from demixing.parse_config import parse_config
from demixing.dataset import TIMIT
from demixing.datamodule import SpeakerIdDataset
from demixing.utils.utils import seed_everything, chunks
from demixing.model.xvector import Classifier
MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()
seed_everything(seed=914)


def build_timit_speaker_mapping():
    timit_root_folder = os.path.join(MODULE_PATH, 'data', 'TIMIT')
    store_path = os.path.join(MODULE_PATH, 'data', 'processed')
    timit = TIMIT(timit_root_folder, store_path)
    timit.build()


def read_timit_speaker_mapping():
    store_path = os.path.join(MODULE_PATH, 'data', 'processed')
    train_file_path = os.path.join(store_path, 'timit.speaker.train.txt')
    test_file_path = os.path.join(store_path, 'timit.speaker.test.txt')
    with open(train_file_path, 'r') as f:
        train_lst = [line.strip().split('\t') for line in f]
    with open(test_file_path, 'r') as f:
        test_lst = [line.strip().split('\t') for line in f]
    return train_lst, test_lst


def train_test_dataset(dataset, test_split=0.25):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def main():
    ################################
    ### Load speeches and labels ###
    ################################

    build_timit_speaker_mapping()
    train_lst, test_lst = read_timit_speaker_mapping()
    audio_files = [ele[0] for ele in train_lst] + [ele[0] for ele in test_lst]
    labels = [ele[1] for ele in train_lst] + [ele[1] for ele in test_lst]
    classes = list(set(labels))
    class2index = {cls:idx for idx, cls in enumerate(classes)}
    labels = list(map(class2index.get, labels))

    #######################
    ### Make DataLoader ###
    #######################

    X_train, X_test, y_train, y_test = train_test_split(audio_files, labels, test_size=CONFIG['test_split'], random_state=914, stratify=labels)
    print(f'Training size: {len(y_train)}')
    print(f'Testing size: {len(y_test)}')

    train_ds = SpeakerIdDataset(X_train, y_train, n_mfcc=CONFIG['mfcc'], sr=CONFIG['sr'], num_seg=CONFIG['train_segments'])
    test_ds = SpeakerIdDataset(X_test, y_test, n_mfcc=CONFIG['mfcc'], sr=CONFIG['sr'], num_seg=CONFIG['test_segments'])
    train_ds, valid_ds = train_test_dataset(train_ds, test_split=0.1)
    train_dl = DataLoader(train_ds, batch_size=CONFIG['batch'], shuffle=True, num_workers=0, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=CONFIG['batch'], shuffle=False, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=CONFIG['batch'], shuffle=False, num_workers=0, pin_memory=True)

    print(f'Training size: {len(train_ds)}')
    print(f'Testing size: {len(test_ds)}')

    #################
    ### Modelling ###
    #################

    model = Classifier(n_mfcc=CONFIG['mfcc'], embed_dim=512, dropout_p=CONFIG['dropout'], num_classes=len(labels))
    model.compile(F.cross_entropy, torch.optim.Adam(model.parameters(), lr=CONFIG['lr']), metrics.accuracy_score, precision=16)
    model.fit(train_dl, valid_dataloader=valid_dl, max_epoch=CONFIG['epoch'], gpu=True)
    loss, acc = model.evaluate(test_dl)
    print(f"Accuracy before averaging: {acc:.4f}")
    y_proba = model.predict_proba(test_dl)
    y_proba = [np.mean(p, axis=0) for p in chunks(y_proba, CONFIG['test_segments'])]
    y_pred = np.array(y_proba).argmax(-1)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy after averaging: {acc:.4f}")
    model.plot()

    embeddings = model.encode_batch(test_dl, gpu=True)


if __name__ == '__main__':
    main()