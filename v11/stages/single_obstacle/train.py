"""
Stage : Apprentissage avec un obstacle unique (single_obstacle).
Introduction progressive des obstacles dans l'environnement.

Ce stage ne connaît PAS son numéro dans la séquence.
Il est identifié uniquement par son slug 'single_obstacle'.
"""

import torch
import random
from typing import Dict, Any, List, Optional, Tuple
from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class SingleObstacleConfig(StageConfig):
    """Configuration pour le stage avec un obstacle unique."""
    
    def __init__(self):
        super().__init__(
            name="single_obstacle",  # Slug unique - identifiant du stage
            description="Apprentissage du contournement d'un obstacle unique",
            epochs_ratio=0.2,
            convergence_threshold=0.0002,
            learning_rate_multiplier=0.8,
            min_obstacles=1,
            max_obstacles=1
        )


class SingleObstacleStage(BaseStage):
    """
    Stage d'apprentissage avec un obstacle unique.
    Introduction du concept de contournement d'obstacles.
    
    Ce stage apprend au NCA à contourner un seul obstacle
    placé aléatoirement dans l'environnement.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = SingleObstacleConfig()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement avec un seul obstacle.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles avec un obstacle unique
            
        Raises:
            RuntimeError: Si impossible de générer un environnement valide
        """
        env_generator = EnvironmentGenerator(self.device)
        return env_generator.generate_single_obstacle_environment(
            size, source_pos,
            min_obstacle_size=self.min_obstacle_size,
            max_obstacle_size=self.max_obstacle_size,
            seed=seed
        )
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour ce stage.
        
        Args:
            global_config: Configuration globale du système
            
        Returns:
            Paramètres d'entraînement optimisés pour l'apprentissage avec obstacles
        """
        return {
            'cache_size': 200,  # Cache plus grand pour la variété d'obstacles
            'use_cache': True,
            'shuffle_frequency': 15,  # Mélange plus fréquent
            'source_intensity': global_config.SOURCE_INTENSITY,
            'validation_frequency': 8,  # Validation plus fréquente
            'obstacle_validation': True,  # Validation spéciale des obstacles
        }
    
    def validate_convergence(self, recent_losses: List[float],
                           epoch_in_stage: int) -> bool:
        """
        Critères de convergence pour ce stage.
        Doit apprendre à contourner efficacement les obstacles.
        
        Args:
            recent_losses: Pertes récentes
            epoch_in_stage: Époque courante dans le stage
            
        Returns:
            True si le stage a convergé
        """
        # Minimum 20 époques car plus complexe que no_obstacles
        if epoch_in_stage < 20 or len(recent_losses) < 12:
            return False
        
        # Convergence avec critères adaptés aux obstacles
        avg_recent_loss = sum(recent_losses[-12:]) / 12
        converged = avg_recent_loss < self.config.convergence_threshold
        
        # Stabilité sur une fenêtre plus longue
        if len(recent_losses) >= 8:
            last_8 = recent_losses[-8:]
            variance = sum((x - sum(last_8)/8)**2 for x in last_8) / 8
            stable = variance < 0.0015  # Seuil légèrement plus permissif
        else:
            stable = False
        
        # Critère d'amélioration continue
        if len(recent_losses) >= 15:
            improvement = recent_losses[-15] - recent_losses[-1]
            improving = improvement > 0.00005  # Amélioration minimale requise
        else:
            improving = True
        
        return converged and stable and improving
    
    def get_learning_rate_schedule(self, epoch_in_stage: int,
                                 max_epochs: int, base_lr: float) -> float:
        """
        Schedule LR spécialisé pour ce stage.
        Décroissance plus graduelle pour l'apprentissage d'obstacles.
        
        Args:
            epoch_in_stage: Époque courante dans ce stage
            max_epochs: Nombre maximum d'époques
            base_lr: Learning rate de base
            
        Returns:
            Learning rate ajusté
        """
        import numpy as np
        
        stage_lr = base_lr * self.config.learning_rate_multiplier
        progress = epoch_in_stage / max_epochs
        
        # Phase de warm-up initial (10% des époques)
        if progress < 0.1:
            warmup_factor = progress / 0.1
            final_lr = stage_lr * (0.5 + 0.5 * warmup_factor)
        else:
            # Décroissance cosine standard après warm-up
            adjusted_progress = (progress - 0.1) / 0.9
            cos_factor = 0.5 * (1 + np.cos(np.pi * adjusted_progress))
            final_lr = stage_lr * (0.15 + 0.85 * cos_factor)
        
        return final_lr
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Poids des pertes pour ce stage.
        
        Returns:
            Dictionnaire avec les poids de chaque composante de perte
        """
        return {
            'mse': 1.0,
            'convergence': 1.5,
            'stability': 1.5,
            'obstacle_penalty': 2.0,  # Pénalité pour traverser les obstacles
        }

