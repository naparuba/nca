"""
Stage : Apprentissage avec atténuation temporelle (time_attenuation).
Gestion avancée des sources avec intensité décroissante au cours du temps.

Ce stage ne connaît PAS son numéro dans la séquence.
Il est identifié uniquement par son slug 'time_attenuation'.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class TimeAttenuationConfig(StageConfig):
    """Configuration pour le stage avec atténuation temporelle."""
    
    def __init__(self):
        super().__init__(
            name="time_attenuation",  # Slug unique - identifiant du stage
            description="Apprentissage avec sources d'intensité décroissante dans le temps",
            epochs_ratio=0.2,
            convergence_threshold=0.00001,
            learning_rate_multiplier=0.3,
            min_obstacles=1,
            max_obstacles=2
        )
        
        # Configuration spéciale pour l'atténuation temporelle
        self.min_source_intensity = 0.3
        self.max_source_intensity = 1.0
        self.initial_intensity_range = [0.5, 1.0]
        self.final_intensity_range = [0.3, 1.0]
        self.intensity_distribution = 'uniform'
        
        # Configuration de l'atténuation temporelle
        self.min_attenuation_rate = 0.002
        self.max_attenuation_rate = 0.015


class TemporalAttenuationManager:
    """
    Gestionnaire spécialisé pour l'atténuation temporelle des sources.
    Gère la diminution progressive de l'intensité de la source pendant la simulation.
    """
    
    def __init__(self, config: TimeAttenuationConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.intensity_history = []
        self.attenuation_history = []
        self.current_time_sequences = {}
    
    def sample_initial_intensity(self, epoch_progress: float) -> float:
        """
        Échantillonne une intensité initiale de source selon l'avancement de l'entraînement.
        
        Args:
            epoch_progress: Progression dans ce stage (0.0 à 1.0)
            
        Returns:
            Intensité initiale échantillonnée pour cette simulation
        """
        intensity_range = self._get_progressive_range(epoch_progress)
        min_intensity, max_intensity = intensity_range
        intensity = min_intensity + (max_intensity - min_intensity) * torch.rand(1, device=self.device).item()
        intensity = self._validate_intensity(intensity)
        self.intensity_history.append(intensity)
        return intensity
    
    def sample_attenuation_rate(self, epoch_progress: float) -> float:
        """
        Échantillonne un taux d'atténuation selon l'avancement de l'entraînement.
        Les taux plus élevés sont introduits progressivement.
        
        Args:
            epoch_progress: Progression dans ce stage (0.0 à 1.0)
            
        Returns:
            Taux d'atténuation par pas de temps
        """
        min_rate = self.config.min_attenuation_rate
        max_rate = min_rate + epoch_progress * (self.config.max_attenuation_rate - min_rate)
        
        # Échantillonnage biaisé vers les taux plus faibles (distribution triangulaire)
        rand_val = torch.rand(1, device=self.device).item()
        attenuation_rate = min_rate + (max_rate - min_rate) * np.sqrt(rand_val)
        
        self.attenuation_history.append(attenuation_rate)
        return attenuation_rate
    
    def compute_intensity_at_step(self, initial_intensity: float, attenuation_rate: float,
                                  step: int) -> float:
        """
        Calcule l'intensité de la source à un pas de temps donné.
        
        Args:
            initial_intensity: Intensité initiale de la source
            attenuation_rate: Taux d'atténuation par pas
            step: Pas de temps actuel
            
        Returns:
            Intensité atténuée au pas de temps spécifié
        """
        # Atténuation exponentielle
        attenuated_intensity = initial_intensity * np.exp(-attenuation_rate * step)
        
        # Intensité minimale pour éviter les valeurs quasi-nulles
        attenuated_intensity = max(attenuated_intensity, 0.001)
        
        return attenuated_intensity
    
    def _get_progressive_range(self, epoch_progress: float) -> Tuple[float, float]:
        """Calcule la plage d'intensité progressive selon l'avancement."""
        initial_range = self.config.initial_intensity_range
        final_range = self.config.final_intensity_range
        
        min_intensity = initial_range[0] + epoch_progress * (final_range[0] - initial_range[0])
        max_intensity = initial_range[1] + epoch_progress * (final_range[1] - initial_range[1])
        
        return (min_intensity, max_intensity)
    
    def _validate_intensity(self, intensity: float) -> float:
        """Valide et ajuste une intensité si nécessaire."""
        intensity = max(0.0, min(1.0, intensity))
        if 0.0 < intensity < 0.001:
            intensity = 0.001
        return intensity
    
    def get_attenuation_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques des intensités et atténuations utilisées."""
        if not self.intensity_history:
            return {
                'intensity': {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'attenuation': {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            }
        
        intensities = np.array(self.intensity_history)
        attenuations = np.array(self.attenuation_history)
        
        return {
            'intensity': {
                'count': len(intensities),
                'mean': float(np.mean(intensities)),
                'std': float(np.std(intensities)),
                'min': float(np.min(intensities)),
                'max': float(np.max(intensities))
            },
            'attenuation': {
                'count': len(attenuations),
                'mean': float(np.mean(attenuations)),
                'std': float(np.std(attenuations)),
                'min': float(np.min(attenuations)),
                'max': float(np.max(attenuations))
            }
        }


class TimeAttenuationStage(BaseStage):
    """
    Stage d'apprentissage avec atténuation temporelle.
    Gestion avancée des sources avec intensité décroissante dans le temps.
    
    Ce stage apprend au NCA à gérer des sources dont l'intensité
    diminue progressivement au cours de la simulation.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = TimeAttenuationConfig()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.attenuation_manager = TemporalAttenuationManager(config, device)
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement avec 1 à 2 obstacles.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles
        """
        env_generator = EnvironmentGenerator(self.device)
        return env_generator.generate_complex_environment(
            size, source_pos,
            min_obstacles=self.config.min_obstacles,
            max_obstacles=self.config.max_obstacles,
            min_obstacle_size=self.min_obstacle_size,
            max_obstacle_size=self.max_obstacle_size,
            placement_attempts=50,
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
            'cache_size': 300,
            'use_cache': True,
            'shuffle_frequency': 10,
            'source_intensity': global_config.SOURCE_INTENSITY,
            'validation_frequency': 15,
            'use_temporal_attenuation': True,
            'attenuation_manager': self.attenuation_manager,
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
        if epoch_in_stage < 35 or len(recent_losses) < 25:
            return False
        
        avg_recent_loss = sum(recent_losses[-25:]) / 25
        converged = avg_recent_loss < self.config.convergence_threshold
        
        if len(recent_losses) >= 15:
            last_15 = recent_losses[-15:]
            variance = sum((x - sum(last_15)/15)**2 for x in last_15) / 15
            stable = variance < 0.003
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
            'convergence': 1.0,
            'stability': 2.5,
            'temporal_consistency': 2.0,
        }
    
    def initialize_sequence(self, n_steps: int, progress: float = 0.0) -> None:
        """
        Initialise une séquence temporelle avec atténuation.
        
        Args:
            n_steps: Nombre de pas de temps de la séquence
            progress: Progression de l'entraînement (0.0 à 1.0)
        """
        initial_intensity = self.attenuation_manager.sample_initial_intensity(progress)
        attenuation_rate = self.attenuation_manager.sample_attenuation_rate(progress)
        
        # Stockage pour usage dans get_source_intensity_at_step
        self._current_initial_intensity = initial_intensity
        self._current_attenuation_rate = attenuation_rate
    
    def get_source_intensity_at_step(self, step: int, base_intensity: float) -> float:
        """
        Retourne l'intensité de la source atténuée à un pas de temps spécifique.
        
        Args:
            step: Pas de temps actuel
            base_intensity: Intensité de base (ignorée, utilise l'intensité initiale échantillonnée)
            
        Returns:
            Intensité de la source atténuée au pas de temps spécifié
        """
        if not hasattr(self, '_current_initial_intensity'):
            return base_intensity
        
        return self.attenuation_manager.compute_intensity_at_step(
            self._current_initial_intensity,
            self._current_attenuation_rate,
            step
        )
    
    def get_attenuation_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'atténuation utilisées durant l'entraînement.
        
        Returns:
            Statistiques d'intensité et d'atténuation
        """
        return self.attenuation_manager.get_attenuation_statistics()

