"""
Stage 3 : Apprentissage avec obstacles multiples.
Gestion de la complexité avec plusieurs obstacles simultanés.
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class Stage3Config(StageConfig):
    """Configuration spécialisée pour le Stage 3."""
    
    def __init__(self):
        super().__init__(
            stage_id=3,
            name="Obstacles multiples",
            description="Apprentissage avec gestion de multiples obstacles complexes",
            epochs_ratio=0.167,  # Ajusté pour répartition équilibrée entre 6 stages
            convergence_threshold=0.0002,
            learning_rate_multiplier=0.6,  # LR encore plus réduit
            min_obstacles=2,
            max_obstacles=4
        )


class Stage3(BaseStage):
    """
    Stage 3 : Apprentissage avec obstacles multiples.
    Gestion de scénarios complexes avec plusieurs obstacles.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = Stage3Config()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.placement_attempts = 50  # Tentatives de placement par obstacle
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement avec obstacles multiples.
        Utilise l'EnvironmentGenerator pour factorisation.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles avec obstacles multiples
            
        Raises:
            RuntimeError: Si impossible de générer un environnement valide
        """
        # Utilisation de l'EnvironmentGenerator pour factorisation
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
    
    def _is_valid_obstacle_position(self, i: int, j: int, obstacle_size: int,
                                  source_pos: Tuple[int, int],
                                  placed_obstacles: List[Tuple[int, int, int]]) -> bool:
        """
        Valide qu'une position d'obstacle est acceptable.
        
        Args:
            i, j: Position proposée
            obstacle_size: Taille de l'obstacle
            source_pos: Position de la source
            placed_obstacles: Obstacles déjà placés
            
        Returns:
            True si la position est valide
        """
        source_i, source_j = source_pos
        
        # 1. Pas de chevauchement avec source
        if (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
            return False
        
        # 2. Pas de chevauchement avec obstacles existants
        for obs_i, obs_j, obs_size in placed_obstacles:
            if (i < obs_i + obs_size and i + obstacle_size > obs_i and
                j < obs_j + obs_size and j + obstacle_size > obs_j):
                return False
        
        # 3. Distance minimale de la source pour éviter l'encerclement
        source_distance = max(abs(i + obstacle_size//2 - source_i),
                            abs(j + obstacle_size//2 - source_j))
        if source_distance < 3:  # Distance minimale
            return False
        
        return True
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour le Stage 3.
        
        Returns:
            Paramètres d'entraînement optimisés pour les obstacles multiples
        """
        return {
            'cache_size': 250,  # Cache le plus grand pour la variété maximale
            'use_cache': True,
            'shuffle_frequency': 12,  # Mélange fréquent pour la diversité
            'source_intensity': global_config.SOURCE_INTENSITY,
            'validation_frequency': 6,  # Validation très fréquente
            'obstacle_validation': True,
            'complexity_validation': True,  # Validation spéciale de complexité
            'min_connectivity_ratio': 0.4,  # Ratio de connectivité plus permissif
        }
    
    def validate_convergence(self, recent_losses: List[float],
                           epoch_in_stage: int) -> bool:
        """
        Critères de convergence pour le Stage 3.
        Le plus strict car c'est le stage le plus complexe.
        
        Args:
            recent_losses: Pertes récentes
            epoch_in_stage: Époque courante dans le stage
            
        Returns:
            True si convergé
        """
        # Minimum 25 époques pour la complexité maximale
        if epoch_in_stage < 25 or len(recent_losses) < 15:
            return False
        
        # Convergence sur une fenêtre plus large
        avg_recent_loss = sum(recent_losses[-15:]) / 15
        converged = avg_recent_loss < self.config.convergence_threshold
        
        # Stabilité stricte sur 10 pertes
        if len(recent_losses) >= 10:
            last_10 = recent_losses[-10:]
            variance = sum((x - sum(last_10)/10)**2 for x in last_10) / 10
            stable = variance < 0.002  # Seuil plus permissif pour la complexité
        else:
            stable = False
        
        # Critère de robustesse : pas de remontée récente
        if len(recent_losses) >= 20:
            recent_trend = sum(recent_losses[-5:]) / 5
            older_trend = sum(recent_losses[-20:-15]) / 5
            robust = recent_trend <= older_trend * 1.05  # Max 5% de remontée
        else:
            robust = True
        
        return converged and stable and robust
    
    def get_learning_rate_schedule(self, epoch_in_stage: int,
                                 max_epochs: int, base_lr: float) -> float:
        """
        Schedule LR spécialisé pour le Stage 3.
        Approche très graduelle avec phases multiples.
        """
        import numpy as np
        
        stage_lr = base_lr * self.config.learning_rate_multiplier
        progress = epoch_in_stage / max_epochs
        
        # Phase 1 : Warm-up étendu (15% des époques)
        if progress < 0.15:
            warmup_factor = progress / 0.15
            final_lr = stage_lr * (0.3 + 0.7 * warmup_factor)
        # Phase 2 : Plateau d'apprentissage (35% des époques)
        elif progress < 0.5:
            final_lr = stage_lr * 0.9
        # Phase 3 : Décroissance cosine graduelle
        else:
            adjusted_progress = (progress - 0.5) / 0.5
            cos_factor = 0.5 * (1 + np.cos(np.pi * adjusted_progress))
            final_lr = stage_lr * (0.2 + 0.7 * cos_factor)
        
        return final_lr
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Poids des pertes pour le Stage 3."""
        return {
            'mse': 1.0,
            'convergence': 1.0,
            'stability': 2.0,  # Accent sur la stabilité
            'robustness': 1.5,  # Nouveau : robustesse aux obstacles multiples
            'complexity_handling': 1.2,  # Nouveau : gestion de la complexité
        }
    
    def post_epoch_hook(self, epoch_in_stage: int, loss: float,
                       metrics: Dict[str, Any]) -> None:
        """Hook post-époque pour le Stage 3."""
        super().post_epoch_hook(epoch_in_stage, loss, metrics)
        
        # Métriques spécifiques au Stage 3
        stage_metrics = {
            'convergence_progress': max(0, 1 - loss / self.config.convergence_threshold),
            'robustness_score': self._calculate_robustness_score(),
            'complexity_mastery': self._calculate_complexity_mastery(),
            'stability_trend': self._calculate_stability_trend()
        }
        
        for key, value in stage_metrics.items():
            if key not in self.training_history['metrics']:
                self.training_history['metrics'][key] = []
            self.training_history['metrics'][key].append(value)
    
    def _calculate_robustness_score(self) -> float:
        """Score de robustesse basé sur la consistance des performances."""
        if len(self.training_history['losses']) < 15:
            return 0.0
        
        # Analyse de la variance sur différentes fenêtres
        recent_losses = self.training_history['losses'][-15:]
        
        # Score basé sur la diminution de la variance au fil du temps
        first_third = recent_losses[:5]
        last_third = recent_losses[-5:]
        
        var_first = sum((x - sum(first_third)/5)**2 for x in first_third) / 5
        var_last = sum((x - sum(last_third)/5)**2 for x in last_third) / 5
        
        # Robustesse = réduction de la variance
        if var_first > 0:
            robustness = max(0, 1 - var_last / var_first)
        else:
            robustness = 1.0
        
        return min(1.0, robustness)
    
    def _calculate_complexity_mastery(self) -> float:
        """Maîtrise de la complexité basée sur la progression d'apprentissage."""
        if len(self.training_history['losses']) < 10:
            return 0.0
        
        # Compare la performance actuelle au début du stage
        recent_avg = sum(self.training_history['losses'][-5:]) / 5
        initial_avg = sum(self.training_history['losses'][:5]) / 5
        
        if initial_avg > 0:
            improvement_ratio = (initial_avg - recent_avg) / initial_avg
            mastery = max(0, min(1.0, improvement_ratio * 2))  # Normalisé
        else:
            mastery = 1.0
        
        return mastery
    
    def _calculate_stability_trend(self) -> float:
        """Tendance de stabilité sur l'historique récent."""
        if len(self.training_history['losses']) < 8:
            return 0.0
        
        # Calcul de la tendance (pente) des pertes récentes
        recent_losses = self.training_history['losses'][-8:]
        n = len(recent_losses)
        
        # Régression linéaire simple pour la tendance
        x_mean = (n - 1) / 2  # Indices centrés
        y_mean = sum(recent_losses) / n
        
        numerator = sum((i - x_mean) * (recent_losses[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator > 0:
            slope = numerator / denominator
            # Stabilité = tendance à la baisse (slope négatif)
            stability = max(0, min(1.0, -slope * 1000))  # Normalisé
        else:
            stability = 0.5
        
        return stability
