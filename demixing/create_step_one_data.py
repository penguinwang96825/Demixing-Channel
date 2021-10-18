import pickle
from os.path import dirname
from pathlib import Path
from read import parse_config
from utils import compute_mfcc
MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()


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
    train_mfcc = compute_mfcc(train_wavforms, CONFIG['mfcc'], CONFIG['sr'])
    test_mfcc = compute_mfcc(test_wavforms, CONFIG['mfcc'], CONFIG['sr'])
    print(train_mfcc.shape, test_mfcc.shape)


if __name__ == '__main__':
    main()