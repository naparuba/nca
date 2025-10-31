from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch import optim, nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AdamW = optim.AdamW

generator = torch.Generator(device=DEVICE)

def get_MSELoss():
    # type: () -> nn.MSELoss
    return nn.MSELoss()


def get_matrix_boolean(matrix_size, fill_value):
    # type: (Tuple[int, int], bool) -> Tensor
    return torch.full(matrix_size, fill_value, dtype=torch.bool, device=DEVICE)


def get_matrix_float(matrix_size, fill_value):
    # type: (Tuple[int, int], float) -> Tensor
    return torch.full(matrix_size, fill_value, dtype=torch.bool, device=DEVICE)


def set_random_seed(seed):
    # type: (int) -> None
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    generator.manual_seed(seed)
    
def get_random_int(low, high):
    # type: (int, int) -> int
    return torch.randint(low=low, high=high, size=(1,), generator=generator, device=DEVICE).item()