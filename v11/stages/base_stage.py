from abc import ABC
from typing import Tuple

import torch


class BaseStage(ABC):
    NAME = 'UNSET'

    def get_name(self):
        return self.NAME

    def generate_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")
    
    