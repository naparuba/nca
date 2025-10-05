"""
Classe de base pour les visualiseurs spécialisés.
Définit l'interface commune que chaque visualiseur doit implémenter.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch


class BaseVisualizer:
    """
    Classe de base abstraite pour tous les visualiseurs de stages.
    Définit l'interface commune que chaque visualiseur doit implémenter.
    """
    
    def __init__(self):
        """Initialisation de base du visualiseur."""
        pass
    
    def create_visualizations(self, output_dir: Path,
                             target_sequence: List[torch.Tensor],
                             nca_sequence: List[torch.Tensor],
                             obstacle_mask: torch.Tensor,
                             source_mask: torch.Tensor,
                             initial_intensity: float,
                             seed: int,
                             temporal_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Méthode principale pour créer toutes les visualisations pour ce stage.
        
        Args:
            output_dir: Répertoire de sortie
            target_sequence: Séquence cible de diffusion
            nca_sequence: Séquence prédite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source
            initial_intensity: Intensité initiale de la source
            seed: Graine aléatoire pour reproductibilité
            temporal_data: Données temporelles additionnelles (optionnel)
        """
        raise NotImplementedError("Les visualiseurs spécialisés doivent implémenter cette méthode")
    
    def _convert_to_numpy(self, tensor_sequence: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Utilitaire pour convertir une séquence de tenseurs en arrays NumPy.
        
        Args:
            tensor_sequence: Liste de tenseurs PyTorch
            
        Returns:
            Liste d'arrays NumPy
        """
        return [t.detach().cpu().numpy() for t in tensor_sequence]
