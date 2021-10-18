import os
import re
import yaml
import pandas as pd
from os.path import dirname
from pathlib import Path
MODULE_PATH = Path(dirname(__file__))


HUGGINGFACE_ROOT = '/home/yangwang/.cache/huggingface/datasets/downloads/extracted/592effd0a12bf1cf783adc699d189209f861ae414612ba8fd21c2927d955e588/'
WIN_HUGGINGFACE_ROOT = ''
TRAIN_ROOT = HUGGINGFACE_ROOT + 'data/TRAIN'
TEST_ROOT = HUGGINGFACE_ROOT + 'data/TEST'


def parse_config():
    with open(MODULE_PATH / 'config.yml') as f:
        config = yaml.safe_load(f)
    return config


def get_timit_metadata(timit_dir):
    """
    Parameters
    ----------
    timit_dir: str
        Path directory for TRAIN or TEST data.
    """
    timit_data = []
    for dialect_region in os.listdir(timit_dir):
        dialect_region_dir_path = os.path.join(timit_dir, dialect_region)
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
    return timit_data