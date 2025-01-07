import random

import numpy as np
import torch


def set_random_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    print("Set random seed to:", seed)
