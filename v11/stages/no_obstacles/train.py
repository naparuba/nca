"""
Stage : Apprentissage sans obstacles (no_obstacles).
Stage de base pour l'apprentissage de la diffusion pure.

Ce stage ne connaît PAS son numéro dans la séquence.
Il est identifié uniquement par son slug 'no_obstacles'.
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class NoObstaclesConfig(StageConfig):
    """Configuration pour le stage sans obstacles."""
    
    def __init__(self):
        super().__init__(
            name="no_obstacles",  # Slug unique - identifiant du stage
            description="Apprentissage de base de la diffusion sans obstacles",
            epochs_ratio=0.2,
            convergence_threshold=0.0002,
            learning_rate_multiplier=1.0,
            min_obstacles=0,
            max_obstacles=0
        )


class NoObstaclesStage(BaseStage):
    """
    Stage d'apprentissage sans obstacles.
    Le plus simple des stages, sert de base pour les suivants.
    
    Ce stage apprend au NCA à diffuser la chaleur/lumière
    dans un environnement vide, sans aucun obstacle.
    """
    
    def __init__(self, device: str = "cpu"):
        config = NoObstaclesConfig()
        super().__init__(config, device)
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement vide (aucun obstacle).
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (non utilisée ici)
            seed: Graine pour la reproductibilité (non utilisée ici)
            
        Returns:
            Masque d'obstacles vide (tous les pixels à False)
        """
        env_generator = EnvironmentGenerator(self.device)
        return env_generator.generate_empty_environment(size)
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour ce stage.
        
        Args:
            global_config: Configuration globale du système
            
        Returns:
            Paramètres d'entraînement optimisés pour l'apprentissage de base
        """
        return {
            'cache_size': 150,  # Cache relativement petit pour stage simple
            'use_cache': True,
            'shuffle_frequency': 20,  # Mélange du cache toutes les 20 époques
            'source_intensity': global_config.SOURCE_INTENSITY,
            'validation_frequency': 10,  # Validation toutes les 10 époques
        }
    
    def validate_convergence(self, recent_losses: List[float],
                           epoch_in_stage: int) -> bool:
        """
        Critères de convergence pour ce stage.
        Convergence stricte requise car c'est la base.
        
        Args:
            recent_losses: Pertes récentes
            epoch_in_stage: Époque courante dans le stage
            
        Returns:
            True si le stage a convergé
        """
        # Minimum 15 époques pour stabiliser
        if epoch_in_stage < 15 or len(recent_losses) < 10:
            return False
        
        # Convergence : moyenne des 10 dernières pertes sous le seuil
        avg_recent_loss = sum(recent_losses[-10:]) / 10
        converged = avg_recent_loss < self.config.convergence_threshold
        
        # Stabilité : variance faible sur les 5 dernières pertes
        if len(recent_losses) >= 5:
            last_5 = recent_losses[-5:]
            variance = sum((x - sum(last_5)/5)**2 for x in last_5) / 5
            stable = variance < 0.001
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
            'convergence': 2.0,  # Accent sur la convergence
            'stability': 1.0
        }
    
    def post_epoch_hook(self, epoch_in_stage: int, loss: float,
                       metrics: Dict[str, Any]) -> None:
        """
        Hook appelé après chaque époque d'entraînement.
        
        Args:
            epoch_in_stage: Numéro d'époque dans ce stage
            loss: Valeur de la perte
            metrics: Métriques additionnelles
        """
        super().post_epoch_hook(epoch_in_stage, loss, metrics)
        
        # Métriques spécifiques à ce stage
        stage_metrics = {
            'convergence_progress': max(0, 1 - loss / self.config.convergence_threshold),
            'stability_score': self._calculate_stability_score()
        }
        
        for key, value in stage_metrics.items():
            if key not in self.training_history['metrics']:
                self.training_history['metrics'][key] = []
            self.training_history['metrics'][key].append(value)
    
    def _calculate_stability_score(self) -> float:
        """
        Calcule un score de stabilité basé sur les pertes récentes.
        
        Returns:
            Score entre 0 et 1 (1 = très stable)
        """
        if len(self.training_history['losses']) < 5:
            return 0.0
        
        recent_losses = self.training_history['losses'][-5:]
        mean_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((x - mean_loss)**2 for x in recent_losses) / len(recent_losses)
        
        # Score inversement proportionnel à la variance
        return max(0, 1 - variance * 1000)

