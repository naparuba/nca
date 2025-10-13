from typing import Tuple, Dict

import torch

from stage_manager import STAGE_MANAGER


class ProgressiveObstacleManager:
    """
    Gestionnaire intelligent des obstacles selon l'étape d'apprentissage.
    Génère des environnements appropriés pour chaque phase du curriculum.
    """
    
    
    def __init__(self):
        pass
    
    
    def generate_stage_environment(self, stage_nb: int, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        """
        Génère un environnement d'obstacles adapté à l'étape courante.
        
        Args:
            stage_nb: Numéro d'étape (1, 2, ou 3)
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            
        Returns:
            Masque des obstacles [H, W]
        """
        
        stage = STAGE_MANAGER.get_stage(stage_nb)
        
        return stage.generate_environment(size, source_pos)
    
    
    def get_difficulty_metrics(self, stage_nb: int, obstacle_mask: torch.Tensor) -> Dict[str, float]:
        """
        Calcule des métriques de difficulté pour l'environnement généré.
        
        Returns:
            Dictionnaire avec les métriques de complexité
        """
        H, W = obstacle_mask.shape
        total_cells = H * W
        obstacle_cells = obstacle_mask.sum().item()
        
        metrics = {
            'stage_nb':         stage_nb,
            'obstacle_ratio':   obstacle_cells / total_cells,
            'free_cells':       total_cells - obstacle_cells,
            'complexity_score': stage_nb * (obstacle_cells / total_cells)
        }
        
        return metrics
