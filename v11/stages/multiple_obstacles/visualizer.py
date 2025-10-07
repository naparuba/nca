"""
Visualisations spécifiques pour le stage avec obstacles multiples (multiple_obstacles).
"""

import torch
from pathlib import Path
from typing import List, Optional, Any


class MultipleObstaclesVisualizer:
    """
    Visualisations spécialisées pour le stage avec obstacles multiples.
    
    Ce visualiseur ne connaît PAS le numéro du stage.
    Il est associé automatiquement via le slug 'multiple_obstacles'.
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
        # TODO: Implémenter les visualisations spécifiques aux obstacles multiples
        pass

