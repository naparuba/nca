from typing import Tuple

import torch
from torch import Tensor
from torch import optim, nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AdamW = optim.AdamW


def get_MSELoss():
    # type: () -> nn.MSELoss
    return nn.MSELoss()


def get_matrix_boolean(matrix_size, fill_value):
    # type: (Tuple[int, int], bool) -> Tensor
    return torch.full(matrix_size, fill_value, dtype=torch.bool, device=DEVICE)


def get_matrix_float(matrix_size, fill_value):
    # type: (Tuple[int, int], float) -> Tensor
    return torch.full(matrix_size, fill_value, dtype=torch.bool, device=DEVICE)
