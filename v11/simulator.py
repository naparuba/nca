from typing import Tuple, List, TYPE_CHECKING

import torch
from torch.nn import functional as F

from config import CONFIG, DEVICE

if TYPE_CHECKING:
    from stages.base_stage import BaseStage


# =============================================================================
# Simulateur de diffusion
# =============================================================================
class HeatDiffusionSimulator:
    """
    Simulateur de diffusion de chaleur adapté pour l'apprentissage modulaire.
    Utilise le gestionnaire d'obstacles progressifs.
    """
    
    
    def __init__(self):
        self.kernel = torch.ones((1, 1, 3, 3), device=DEVICE) / 9.0  # Average 3x3
    
    
    def step(self, grid, source_mask, obstacle_mask):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """Un pas de diffusion de chaleur avec obstacles."""
        x = grid.unsqueeze(0).unsqueeze(0)
        new_grid = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)
        
        # Contraintes
        new_grid[obstacle_mask] = 0.0
        new_grid[source_mask] = grid[source_mask]
        
        return new_grid
    
    
    def generate_stage_sequence(self, stage, n_steps, size):
        # type: (BaseStage, int, int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]
        """
        Génère une séquence adaptée à l'étape d'apprentissage courante.
        
        Args:
            stage: Stage
            n_steps: Nombre d'étapes de simulation
            size: Taille de la grille
            
        Returns :
            (séquence, masque_source, masque_obstacles)
        """
        # Position aléatoire de la source
        g = torch.Generator(device=DEVICE)
        g.manual_seed(CONFIG.SEED)
        i0 = torch.randint(2, size - 2, (1,), generator=g, device=DEVICE).item()
        j0 = torch.randint(2, size - 2, (1,), generator=g, device=DEVICE).item()
        
        # Génération d'obstacles selon l'étape
        obstacle_mask = stage.generate_environment(size, (i0, j0))
        
        # Initialisation
        grid = torch.zeros((size, size), device=DEVICE)
        grid[i0, j0] = CONFIG.SOURCE_INTENSITY
        
        source_mask = torch.zeros_like(grid, dtype=torch.bool)
        source_mask[i0, j0] = True
        
        # S'assurer que la source n'est pas dans un obstacle
        if obstacle_mask[i0, j0]:
            obstacle_mask[i0, j0] = False
        
        # Simulation temporelle
        sequence = [grid.clone()]
        for _ in range(n_steps):
            grid = self.step(grid, source_mask, obstacle_mask)
            sequence.append(grid.clone())
        
        return sequence, source_mask, obstacle_mask


simulator = None


def get_simulator() -> HeatDiffusionSimulator:
    global simulator
    if simulator is None:
        simulator = HeatDiffusionSimulator()
    return simulator
