from typing import Tuple

import torch
from config import DEVICE
from stages.base_stage import BaseStage
from torched import get_matrix_boolean


class Stage1NoObstacle(BaseStage):
    NAME = 'no_obstacle'
    DISPLAY_NAME = "Sans obstable"
    COLOR = 'green'
    
    
    def generate_environment(self, size, source_pos):
        # type: (int, Tuple[int, int]) -> torch.Tensor
        """Étape 1: Aucun obstacle - grille vide pour apprentissage de base."""
        return get_matrix_boolean((size, size), False)
