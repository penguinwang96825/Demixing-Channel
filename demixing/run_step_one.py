import pickle
from dataset import StepOneDataset
from os.path import dirname
from pathlib import Path
from read import parse_config
MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()


def main():
    trn_seg = CONFIG['train_segments']
    tst_seg = CONFIG['test_segments']
    path = f'./data/trn{trn_seg}_tst{tst_seg}.pkl'
    train_ds = StepOneDataset(path, CONFIG, state='train')
    test_ds = StepOneDataset(path, CONFIG, state='test')


if __name__ == '__main__':
    main()