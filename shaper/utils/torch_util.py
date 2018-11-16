import random
import numpy as np

import torch
from torch.utils.collect_env import get_pretty_env_info


def get_PIL_version():
    try:
        import PIL
    except ImportError as e:
        return "\n No Pillow is found."
    else:
        return "\nPillow ({})".format(PIL.__version__)


def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += get_PIL_version()
    return env_str


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
