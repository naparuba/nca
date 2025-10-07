"""
Visualisations spécifiques pour le stage avec obstacle unique (single_obstacle).
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Any


class SingleObstacleVisualizer:
    """
    Visualisations spécialisées pour le stage avec un obstacle unique.
    
    Ce visualiseur ne connaît PAS le numéro du stage.
    Il est associé automatiquement via le slug 'single_obstacle'.
    """
    
    @staticmethod
    def create_visualizations(stage_dir: Path, target_seq: List[torch.Tensor],
                             nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                             source_mask: torch.Tensor, source_intensity: float,
                             vis_seed: int, intensity_history: Optional[Any] = None):
        """
        Crée les visualisations complètes pour ce stage.
        
        Args:
            stage_dir: Répertoire où sauvegarder les visualisations
            target_seq: Séquence cible
            nca_seq: Séquence prédite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source
            source_intensity: Intensité de la source
            vis_seed: Graine de visualisation
            intensity_history: Non utilisé pour ce stage
        """
        print(f"  📊 Création des visualisations pour {stage_dir.name}")
        # TODO: Implémenter les visualisations spécifiques aux obstacles
        pass

