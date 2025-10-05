"""
Stage 2 : Apprentissage avec obstacles simples.
Introduction progressive des obstacles dans l'environnement.
"""

import torch
import random
from typing import Dict, Any, List, Optional, Tuple
from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class Stage2Config(StageConfig):
    """Configuration spécialisée pour le Stage 2."""
    
    def __init__(self):
        super().__init__(
            stage_id=2,
            name="Un obstacle",
            description="Apprentissage du contournement d'un obstacle unique",
            epochs_ratio=0.3,
            convergence_threshold=0.0002,
            learning_rate_multiplier=0.8,  # LR légèrement réduit
            min_obstacles=1,
            max_obstacles=1
        )


class Stage2(BaseStage):
    """
    Stage 2 : Apprentissage avec un obstacle unique.
    Introduction du concept de contournement d'obstacles.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = Stage2Config()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement avec un seul obstacle.
        Utilise l'EnvironmentGenerator pour la génération d'obstacles.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles avec un obstacle unique
            
        Raises:
            RuntimeError: Si impossible de générer un environnement valide
        """
        # Utilisation de l'EnvironmentGenerator pour factorisation
        env_generator = EnvironmentGenerator(self.device)
        return env_generator.generate_single_obstacle_environment(
            size, source_pos,
            min_obstacle_size=self.min_obstacle_size,
            max_obstacle_size=self.max_obstacle_size,
            seed=seed
        )
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour le Stage 2.
        
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
        Critères de convergence pour le Stage 2.
        Doit apprendre à contourner efficacement les obstacles.
        
        Args:
            recent_losses: Pertes récentes
            epoch_in_stage: Époque courante dans le stage
            
        Returns:
            True si convergé
        """
        # Minimum 20 époques car plus complexe que Stage 1
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
        Schedule LR spécialisé pour le Stage 2.
        Décroissance plus graduelle pour l'apprentissage d'obstacles.
        """
        import numpy as np
        
        stage_lr = base_lr * self.config.learning_rate_multiplier
        
        # Décroissance cosine modifiée pour obstacles
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
        """Poids des pertes pour le Stage 2."""
        return {
            'mse': 1.0,
            'convergence': 1.5,
            'stability': 1.5,
            'adaptation': 1.0,  # Nouveau : adaptation aux obstacles
        }
    
    def post_epoch_hook(self, epoch_in_stage: int, loss: float,
                       metrics: Dict[str, Any]) -> None:
        """Hook post-époque pour le Stage 2."""
        super().post_epoch_hook(epoch_in_stage, loss, metrics)
        
        # Métriques spécifiques au Stage 2
        stage_metrics = {
            'convergence_progress': max(0, 1 - loss / self.config.convergence_threshold),
            'adaptation_score': self._calculate_adaptation_score(),
            'obstacle_handling_efficiency': self._calculate_obstacle_efficiency()
        }
        
        for key, value in stage_metrics.items():
            if key not in self.training_history['metrics']:
                self.training_history['metrics'][key] = []
            self.training_history['metrics'][key].append(value)
    
    def _calculate_adaptation_score(self) -> float:
        """Score d'adaptation basé sur la vitesse d'amélioration."""
        if len(self.training_history['losses']) < 10:
            return 0.0
        
        # Compare les 5 premières et 5 dernières pertes de l'historique récent
        recent_losses = self.training_history['losses'][-10:]
        first_half = sum(recent_losses[:5]) / 5
        second_half = sum(recent_losses[5:]) / 5
        
        improvement_ratio = max(0, (first_half - second_half) / first_half)
        return min(1.0, improvement_ratio * 5)  # Normalisé
    
    def _calculate_obstacle_efficiency(self) -> float:
        """Efficacité de gestion des obstacles (métrique personnalisée)."""
        if len(self.training_history['losses']) < 5:
            return 0.0
        
        # Basé sur la consistance des pertes récentes
        recent_losses = self.training_history['losses'][-5:]
        mean_loss = sum(recent_losses) / len(recent_losses)
        
        # Plus la perte est faible, plus l'efficacité est haute
        efficiency = max(0, 1 - mean_loss / (self.config.convergence_threshold * 10))
        return min(1.0, efficiency)
