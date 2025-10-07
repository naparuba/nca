"""
Stage : Apprentissage avec obstacles multiples (multiple_obstacles).
Gestion de la complexité avec plusieurs obstacles simultanés.

Ce stage ne connaît PAS son numéro dans la séquence.
Il est identifié uniquement par son slug 'multiple_obstacles'.
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class MultipleObstaclesConfig(StageConfig):
    """Configuration pour le stage avec obstacles multiples."""
    
    def __init__(self):
        super().__init__(
            name="multiple_obstacles",  # Slug unique - identifiant du stage
            description="Apprentissage avec gestion de multiples obstacles complexes",
            epochs_ratio=0.2,
            convergence_threshold=0.0002,
            learning_rate_multiplier=0.6,
            min_obstacles=2,
            max_obstacles=4
        )


class MultipleObstaclesStage(BaseStage):
    """
    Stage d'apprentissage avec obstacles multiples.
    Gestion de scénarios complexes avec plusieurs obstacles.
    
    Ce stage apprend au NCA à naviguer dans un environnement
    contenant 2 à 4 obstacles placés aléatoirement.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = MultipleObstaclesConfig()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.placement_attempts = 50
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement avec obstacles multiples.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles avec obstacles multiples
            
        Raises:
            RuntimeError: Si impossible de générer un environnement valide
        """
        env_generator = EnvironmentGenerator(self.device)
        return env_generator.generate_complex_environment(
            size, source_pos,
            min_obstacles=self.config.min_obstacles,
            max_obstacles=self.config.max_obstacles,
            min_obstacle_size=self.min_obstacle_size,
            max_obstacle_size=self.max_obstacle_size,
            placement_attempts=self.placement_attempts,
            seed=seed
        )
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour ce stage.
        
        Args:
            global_config: Configuration globale du système
            
        Returns:
            Paramètres d'entraînement optimisés
        """
        return {
            'cache_size': 250,
            'use_cache': True,
            'shuffle_frequency': 12,
            'source_intensity': global_config.SOURCE_INTENSITY,
            'validation_frequency': 10,
            'obstacle_validation': True,
            'complexity_aware': True,
        }
    
    def validate_convergence(self, recent_losses: List[float],
                           epoch_in_stage: int) -> bool:
        """
        Critères de convergence pour ce stage.
        
        Args:
            recent_losses: Pertes récentes
            epoch_in_stage: Époque courante dans le stage
            
        Returns:
            True si le stage a convergé
        """
        if epoch_in_stage < 25 or len(recent_losses) < 15:
            return False
        
        avg_recent_loss = sum(recent_losses[-15:]) / 15
        converged = avg_recent_loss < self.config.convergence_threshold
        
        if len(recent_losses) >= 10:
            last_10 = recent_losses[-10:]
            variance = sum((x - sum(last_10)/10)**2 for x in last_10) / 10
            stable = variance < 0.002
        else:
            stable = False
        
        return converged and stable
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Poids des pertes pour ce stage.
        
        Returns:
            Dictionnaire avec les poids de chaque composante de perte
        """
        return {
            'mse': 1.0,
            'convergence': 1.5,
            'stability': 2.0,
            'obstacle_penalty': 2.5,
            'complexity_bonus': 0.5,
        }

