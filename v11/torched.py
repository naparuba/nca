from typing import Tuple

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_matrix_boolean(matrix_size, fill_value):
    # type: (Tuple[int, int], bool) -> torch.Tensor
    return torch.full(matrix_size, fill_value, dtype=torch.bool, device=DEVICE)
