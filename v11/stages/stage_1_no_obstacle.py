from typing import Tuple

import torch
from config import DEVICE
from stages.base_stage import BaseStage


class Stage1NoObstacle(BaseStage):
    NAME = 'no_obstacle'
    DISPLAY_NAME = "Sans obstable"
    COLOR = 'green'
    
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        """Étape 1: Aucun obstacle - grille vide pour apprentissage de base."""
        return torch.zeros((size, size), dtype=torch.bool, device=DEVICE)
