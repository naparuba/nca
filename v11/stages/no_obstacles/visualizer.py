"""
Visualisations spécifiques pour le stage sans obstacles (no_obstacles).
Ce module contient les visualisations personnalisées pour le stage de base.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class NoObstaclesVisualizer:
    """
    Visualisations spécialisées pour le stage sans obstacles.
    
    Ce visualiseur ne connaît PAS le numéro du stage.
    Il est associé automatiquement via le slug 'no_obstacles'.
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
            intensity_history: Non utilisé pour ce stage, inclus pour compatibilité
        """
        # Crée les animations standard
        NoObstaclesVisualizer._create_animations(
            stage_dir, target_seq, nca_seq, obstacle_mask, source_intensity
        )
        
        # Crée les graphiques de convergence
        NoObstaclesVisualizer._create_convergence_plot(
            stage_dir, target_seq, nca_seq, vis_seed
        )
        
        # Visualisations spécifiques à ce stage
        NoObstaclesVisualizer._create_diffusion_pattern_plot(
            stage_dir, nca_seq, source_mask
        )
    
    @staticmethod
    def _create_animations(stage_dir: Path, target_seq: List[torch.Tensor],
                          nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                          source_intensity: float):
        """Crée les animations de comparaison."""
        print(f"  📊 Création des animations pour {stage_dir.name}")
        # TODO: Implémenter la création d'animations
        # Cette méthode sera complétée selon les besoins
        pass
    
    @staticmethod
    def _create_convergence_plot(stage_dir: Path, target_seq: List[torch.Tensor],
                                nca_seq: List[torch.Tensor], vis_seed: int):
        """Crée le graphique de convergence."""
        print(f"  📈 Création du graphique de convergence")
        # TODO: Implémenter le graphique de convergence
        pass
    
    @staticmethod
    def _create_diffusion_pattern_plot(stage_dir: Path, nca_seq: List[torch.Tensor],
                                      source_mask: torch.Tensor):
        """
        Crée une visualisation du pattern de diffusion.
        Spécifique à ce stage pour montrer la diffusion pure.
        """
        print(f"  🎨 Création du pattern de diffusion")
        # TODO: Implémenter la visualisation du pattern
        pass

