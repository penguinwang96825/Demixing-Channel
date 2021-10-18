import os
import re
import pickle
import random
import librosa
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from os.path import dirname
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from utils import pad_sequences, contiguous_slice, check_dir
from read import parse_config, get_timit_metadata, TRAIN_ROOT, TEST_ROOT
MODULE_PATH = Path(dirname(__file__))
CONFIG = parse_config()


def wav_and_label(audio_files, speakers, num_segments, sample_rate=16000, padding='zero', discard=True):
    assert len(audio_files) == len(speakers)
    if isinstance(audio_files, pd.Series):
        audio_files = audio_files.tolist()
    if isinstance(speakers, pd.Series):
        speakers = speakers.tolist()

    def _wav_and_label(*args, **kwargs):
        for idx, file in enumerate(tqdm(audio_files)):
            try:
                wavform, sr = librosa.load(file, sr=sample_rate)
                if len(wavform) < sample_rate:
                    if discard:
                        continue
                    wavform = pad_sequences(wavform, maxlen=sr, padding=padding, batch=False)
                spk = speakers[idx]
                wavforms = contiguous_slice(wavform, windows=sr, num_samples=num_segments)
                for wav in wavforms:
                    yield (wav, spk)
            except:
                print("Error reading WAV file!")

    return list(_wav_and_label(audio_files, speakers, sample_rate=16000, padding='zero'))


def main():
    train_df = get_timit_metadata(TRAIN_ROOT)
    test_df = get_timit_metadata(TEST_ROOT)
    data = pd.concat([train_df, test_df], axis=0)
    spk2idx = {spk:idx for idx, spk in enumerate(data.speaker_id.unique())}
    idx2spk = {idx:spk for spk, idx in spk2idx.items()}
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=914, stratify=data['speaker_id'])
    train_data = wav_and_label(
        train_df['file'], train_df['speaker_id'].apply(lambda x: spk2idx[x]), 
        num_segments=CONFIG['train_segments'], 
        sample_rate=CONFIG['sr'], 
        padding=CONFIG['padding'], 
        discard=CONFIG['discard']
    )
    test_data = wav_and_label(
        test_df['file'], test_df['speaker_id'].apply(lambda x: spk2idx[x]), 
        num_segments=CONFIG['test_segments'], 
        sample_rate=CONFIG['sr'], 
        padding=CONFIG['padding'], 
        discard=CONFIG['discard']
    )
    trn_seg = CONFIG['train_segments']
    tst_seg = CONFIG['test_segments']
    step_one_data = {
        'train': train_data, 
        'test': test_data
    }
    check_dir('./data')
    with open(f'./data/trn{trn_seg}_tst{tst_seg}.pkl', 'wb') as f:
        pickle.dump(step_one_data, f)


if __name__ == '__main__':
    main()