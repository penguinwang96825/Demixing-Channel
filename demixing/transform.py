import pickle
from os.path import dirname
from pathlib import Path
from read import parse_config
from utils import compute_mfcc
from collections import Counter
MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()


def mfcc_and_label(wavforms, speakers=None, n_mfcc=20, sr=16000):
    mfccs = compute_mfcc(wavforms, n_mfcc, sr)
    return mfccs, speakers


def main():
    trn_seg = CONFIG['train_segments']
    tst_seg = CONFIG['test_segments']
    with open(f'./data/trn{trn_seg}_tst{tst_seg}.pkl', 'rb') as f:
        step_one_data = pickle.load(f)
    train_data = step_one_data['train']
    test_data = step_one_data['test']
    train_wavforms = [d[0] for d in train_data]
    train_speakers = [d[1] for d in train_data]
    test_wavforms = [d[0] for d in test_data]
    test_speakers = [d[1] for d in test_data]
    print(len(Counter(train_speakers)))
    print(len(Counter(test_speakers)))
    print(Counter(train_speakers))
    print(Counter(test_speakers))
    # train_mfcc, _ = mfcc_and_label(train_wavforms, n_mfcc=CONFIG['mfcc'], sr=CONFIG['sr'])
    # test_mfcc, _ = mfcc_and_label(test_wavforms, n_mfcc=CONFIG['mfcc'], sr=CONFIG['sr'])
    # print(train_mfcc.shape)
    # print(test_mfcc.shape)


if __name__ == '__main__':
    main()