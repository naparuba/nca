"""
Stage : Apprentissage avec intensités variables (variable_intensity).
Gestion avancée avec intensités de source dynamiques.

Ce stage ne connaît PAS son numéro dans la séquence.
Il est identifié uniquement par son slug 'variable_intensity'.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class VariableIntensityConfig(StageConfig):
    """Configuration pour le stage avec intensités variables."""
    
    def __init__(self):
        super().__init__(
            name="variable_intensity",  # Slug unique - identifiant du stage
            description="Apprentissage avec intensités de source variables et obstacles",
            epochs_ratio=0.2,
            convergence_threshold=0.0002,
            learning_rate_multiplier=0.4,
            min_obstacles=1,
            max_obstacles=2
        )
        
        # Configuration spéciale pour intensités variables
        self.min_source_intensity = 0.0
        self.max_source_intensity = 1.0
        self.initial_intensity_range = [0.5, 1.0]
        self.final_intensity_range = [0.0, 1.0]
        self.intensity_distribution = 'uniform'


class IntensityManager:
    """
    Gestionnaire spécialisé pour les intensités variables.
    Complètement découplé des autres stages.
    """
    
    def __init__(self, config: VariableIntensityConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.intensity_history = []
    
    def sample_simulation_intensity(self, epoch_progress: float) -> float:
        """
        Échantillonne une intensité selon l'avancement de l'entraînement.
        
        Args:
            epoch_progress: Progression dans ce stage (0.0 à 1.0)
            
        Returns:
            Intensité échantillonnée pour cette simulation
        """
        intensity_range = self._get_progressive_range(epoch_progress)
        min_intensity, max_intensity = intensity_range
        intensity = min_intensity + (max_intensity - min_intensity) * torch.rand(1, device=self.device).item()
        intensity = self._validate_intensity(intensity)
        self.intensity_history.append(intensity)
        return intensity
    
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
    
    def get_intensity_statistics(self) -> Dict[str, float]:
        """Retourne les statistiques des intensités utilisées."""
        if not self.intensity_history:
            return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        intensities = np.array(self.intensity_history)
        return {
            'count': len(intensities),
            'mean': float(np.mean(intensities)),
            'std': float(np.std(intensities)),
            'min': float(np.min(intensities)),
            'max': float(np.max(intensities))
        }


class VariableIntensityStage(BaseStage):
    """
    Stage d'apprentissage avec intensités variables.
    Gestion avancée des variations d'intensité de la source.
    
    Ce stage apprend au NCA à s'adapter à différentes intensités
    de source, rendant le modèle plus robuste.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = VariableIntensityConfig()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.intensity_manager = IntensityManager(config, device)
    
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
            'validation_frequency': 12,
            'use_variable_intensity': True,
            'intensity_manager': self.intensity_manager,
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
        if epoch_in_stage < 30 or len(recent_losses) < 20:
            return False
        
        avg_recent_loss = sum(recent_losses[-20:]) / 20
        converged = avg_recent_loss < self.config.convergence_threshold
        
        if len(recent_losses) >= 12:
            last_12 = recent_losses[-12:]
            variance = sum((x - sum(last_12)/12)**2 for x in last_12) / 12
            stable = variance < 0.0025
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
            'stability': 2.0,
            'intensity_robustness': 1.5,
        }
    
    def get_intensity_statistics(self) -> Dict[str, float]:
        """
        Retourne les statistiques des intensités utilisées durant l'entraînement.
        
        Returns:
            Statistiques d'intensité
        """
        return self.intensity_manager.get_intensity_statistics()

