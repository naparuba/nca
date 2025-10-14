from typing import Tuple

import torch
from stages.base_stage import BaseStage
from torching import DEVICE

from config import CONFIG


class Stage2OneObstacle(BaseStage):
    NAME = 'one_obstacle'
    DISPLAY_NAME = "Un obstable"
    COLOR = 'orange'
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        """Étape 2: Un seul obstacle pour apprentissage du contournement."""
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=DEVICE)
        
        g = torch.Generator(device=DEVICE)
        g.manual_seed(CONFIG.SEED)
        
        # Un seul obstacle de taille aléatoire
        obstacle_size = torch.randint(CONFIG.MIN_OBSTACLE_SIZE, CONFIG.MAX_OBSTACLE_SIZE + 1,
                                      (1,), generator=g, device=DEVICE).item()
        
        # Placement en évitant la source et les bords
        max_pos = size - obstacle_size
        if max_pos <= 1:
            return obstacle_mask  # Grille trop petite
        
        source_i, source_j = source_pos
        
        for attempt in range(100):
            i = torch.randint(1, max_pos, (1,), generator=g, device=DEVICE).item()
            j = torch.randint(1, max_pos, (1,), generator=g, device=DEVICE).item()
            
            # Vérifier non-chevauchement avec source
            if not (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
                obstacle_mask[i:i + obstacle_size, j:j + obstacle_size] = True
                break
        
        return obstacle_mask
