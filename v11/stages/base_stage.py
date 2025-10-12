from abc import ABC
from typing import Tuple

import torch


class BaseStage(ABC):
    NAME = 'UNSET'
    DISPLAY_NAME = 'Étape non définie'

    def __init__(self):
        self._stage_nb = -1  # Numéro de l'étape, à définir dans les sous-classes

    def get_name(self):
        return self.NAME
    
    def get_display_name(self):
        return self.DISPLAY_NAME

    def set_stage_nb(self, stage_nb: int):
        self._stage_nb = stage_nb
        
    def get_stage_nb(self):
        return self._stage_nb

    def generate_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")
    
    