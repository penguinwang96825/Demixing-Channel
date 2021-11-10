import yaml
from os.path import dirname
from pathlib import Path
MODULE_PATH = Path(dirname(__file__))


def parse_config():
    with open(MODULE_PATH / 'config.yml') as f:
        config = yaml.safe_load(f)
    return config
