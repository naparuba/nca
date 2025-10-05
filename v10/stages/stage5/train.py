"""
Stage 5 : Apprentissage avec atténuation temporelle des sources.
Gestion avancée des sources avec intensité décroissante au cours du temps.
"""

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch

from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class Stage5Config(StageConfig):
    """Configuration spécialisée pour le Stage 5."""
    
    def __init__(self):
        super().__init__(
            stage_id=5,
            name="Atténuation Temporelle des Sources",
            description="Apprentissage avec sources d'intensité décroissante dans le temps",
            epochs_ratio=0.2,
            convergence_threshold=0.001,
            learning_rate_multiplier=0.3,  # LR encore plus réduit pour ce stage complexe
            min_obstacles=1,
            max_obstacles=2
        )
        
        # Configuration spéciale pour l'atténuation temporelle
        self.min_source_intensity = 0.3
        self.max_source_intensity = 1.0
        self.initial_intensity_range = [0.5, 1.0]  # Plage initiale restreinte
        self.final_intensity_range = [0.3, 1.0]    # Plage finale élargie
        self.intensity_distribution = 'uniform'
        
        # Configuration de l'atténuation temporelle
        self.min_attenuation_rate = 0.002  # Atténuation minimale par pas de temps
        self.max_attenuation_rate = 0.015  # Atténuation maximale par pas de temps


class TemporalAttenuationManager:
    """
    Gestionnaire spécialisé pour l'atténuation temporelle des sources du Stage 5.
    Gère la diminution progressive de l'intensité de la source pendant la simulation.
    """
    
    def __init__(self, config: Stage5Config, device: str = "cpu"):
        self.config = config
        self.device = device
        self.intensity_history = []  # Historique des intensités initiales
        self.attenuation_history = []  # Historique des taux d'atténuation
        self.current_time_sequences = {}  # Séquences d'atténuation en cours
    
    def sample_initial_intensity(self, epoch_progress: float) -> float:
        """
        Échantillonne une intensité initiale de source selon l'avancement de l'entraînement.
        
        Args:
            epoch_progress: Progression dans l'étape 5 (0.0 à 1.0)
            
        Returns:
            Intensité initiale échantillonnée pour cette simulation
        """
        intensity_range = self._get_progressive_range(epoch_progress)
        
        # Échantillonnage uniforme dans la plage progressive
        min_intensity, max_intensity = intensity_range
        intensity = min_intensity + (max_intensity - min_intensity) * torch.rand(1, device=self.device).item()
        
        # Validation et ajustement
        intensity = self._validate_intensity(intensity)
        
        # Historique pour statistiques
        self.intensity_history.append(intensity)
        
        return intensity
    
    def sample_attenuation_rate(self, epoch_progress: float) -> float:
        """
        Échantillonne un taux d'atténuation selon l'avancement de l'entraînement.
        Les taux plus élevés sont introduits progressivement.
        
        Args:
            epoch_progress: Progression dans l'étape 5 (0.0 à 1.0)
            
        Returns:
            Taux d'atténuation par pas de temps
        """
        # Élargissement progressif de la plage d'atténuation
        min_rate = self.config.min_attenuation_rate
        max_rate = min_rate + epoch_progress * (self.config.max_attenuation_rate - min_rate)
        
        # Échantillonnage biaisé vers les taux plus faibles (distribution triangulaire)
        rand_val = torch.rand(1, device=self.device).item()
        attenuation_rate = min_rate + (max_rate - min_rate) * np.sqrt(rand_val)
        
        # Historique pour statistiques
        self.attenuation_history.append(attenuation_rate)
        
        return attenuation_rate
    
    def generate_temporal_sequence(self, initial_intensity: float, attenuation_rate: float,
                                  n_steps: int) -> List[float]:
        """
        Génère une séquence d'intensités décroissantes dans le temps.
        
        Args:
            initial_intensity: Intensité initiale de la source
            attenuation_rate: Taux d'atténuation par pas de temps
            n_steps: Nombre de pas de temps dans la séquence
            
        Returns:
            Liste des intensités pour chaque pas de temps
        """
        sequence = [initial_intensity]
        
        # Génération de la séquence d'intensités décroissantes
        for step in range(1, n_steps):
            # Atténuation linéaire avec limite inférieure à 0
            new_intensity = max(0.0, sequence[-1] - attenuation_rate)
            sequence.append(new_intensity)
        
        # Stockage pour analyse et visualisation
        sequence_id = len(self.current_time_sequences)
        self.current_time_sequences[sequence_id] = {
            'initial_intensity': initial_intensity,
            'attenuation_rate': attenuation_rate,
            'sequence': sequence
        }
        
        # Limiter le nombre de séquences stockées
        if len(self.current_time_sequences) > 100:
            # Garder seulement les 50 plus récentes
            self.current_time_sequences = {k: v for k, v in list(self.current_time_sequences.items())[-50:]}
        
        return sequence
    
    def get_source_intensity_at_step(self, sequence_id: int, step: int) -> float:
        """
        Récupère l'intensité de la source à un pas de temps spécifique.
        
        Args:
            sequence_id: Identifiant de la séquence
            step: Pas de temps
            
        Returns:
            Intensité de la source à ce pas de temps
        """
        if sequence_id not in self.current_time_sequences:
            return 0.0
        
        sequence = self.current_time_sequences[sequence_id]['sequence']
        if step >= len(sequence):
            return 0.0
        
        return sequence[step]
    
    def _get_progressive_range(self, epoch_progress: float) -> Tuple[float, float]:
        """Calcule la plage d'intensité progressive selon l'avancement."""
        initial_range = self.config.initial_intensity_range
        final_range = self.config.final_intensity_range
        
        # Interpolation linéaire entre plages initiale et finale
        min_intensity = initial_range[0] + epoch_progress * (final_range[0] - initial_range[0])
        max_intensity = initial_range[1] + epoch_progress * (final_range[1] - initial_range[1])
        
        return (min_intensity, max_intensity)
    
    def _validate_intensity(self, intensity: float) -> float:
        """Valide et ajuste une intensité si nécessaire."""
        # Assure que l'intensité est dans [0.0, 1.0]
        intensity = max(0.0, min(1.0, intensity))
        
        # Évite les intensités quasi-nulles problématiques
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
            'max': float(np.max(intensities)),
            'median': float(np.median(intensities))
        }
    
    def get_attenuation_statistics(self) -> Dict[str, float]:
        """Retourne les statistiques des taux d'atténuation utilisés."""
        if not self.attenuation_history:
            return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        rates = np.array(self.attenuation_history)
        
        return {
            'count': len(rates),
            'mean': float(np.mean(rates)),
            'std': float(np.std(rates)),
            'min': float(np.min(rates)),
            'max': float(np.max(rates)),
            'median': float(np.median(rates))
        }
    
    def clear_history(self):
        """Efface l'historique pour économiser la mémoire."""
        if len(self.intensity_history) > 1000:
            self.intensity_history = self.intensity_history[-1000:]
        if len(self.attenuation_history) > 1000:
            self.attenuation_history = self.attenuation_history[-1000:]


class Stage5(BaseStage):
    """
    Stage 5 : Apprentissage avec atténuation temporelle des sources.
    Nouveau stage avec intensité décroissante au cours du temps.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = Stage5Config()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.attenuation_manager = TemporalAttenuationManager(config, device)
        self.placement_attempts = 50
        self.current_sequence = None
        self.current_sequence_id = -1
        self.current_step = 0
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement pour le stage d'atténuation temporelle.
        Utilise l'EnvironmentGenerator pour factorisation.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles pour Stage 5
            
        Raises:
            RuntimeError: Si impossible de générer un environnement valide
        """
        # Utilisation de l'EnvironmentGenerator pour factorisation
        env_generator = EnvironmentGenerator(self.device)
        return env_generator.generate_variable_intensity_environment(
            size, source_pos,
            min_obstacles=self.config.min_obstacles,
            max_obstacles=self.config.max_obstacles,
            min_obstacle_size=self.min_obstacle_size,
            max_obstacle_size=self.max_obstacle_size,
            placement_attempts=self.placement_attempts,
            seed=seed
        )
    
    def initialize_sequence(self, n_steps: int, progress: float = 0.5) -> None:
        """
        Initialise une nouvelle séquence d'atténuation temporelle.
        Surcharge la méthode de BaseStage.
        
        Args:
            n_steps: Nombre de pas de temps dans la séquence
            progress: Progression dans l'entraînement (0.0 à 1.0)
        """
        # Échantillonnage de l'intensité initiale et du taux d'atténuation
        initial_intensity = self.attenuation_manager.sample_initial_intensity(progress)
        attenuation_rate = self.attenuation_manager.sample_attenuation_rate(progress)
        
        # Génération de la séquence
        self.current_sequence = self.attenuation_manager.generate_temporal_sequence(
            initial_intensity, attenuation_rate, n_steps)
        
        # Mise à jour de l'état
        self.current_sequence_id = len(self.attenuation_manager.current_time_sequences) - 1
        self.current_step = 0
    
    def get_source_intensity_at_step(self, step: int, initial_intensity: float = None) -> float:
        """
        Récupère l'intensité de la source à un pas de temps spécifique.
        Surcharge la méthode de BaseStage.
        
        Args:
            step: Pas de temps actuel
            initial_intensity: Intensité initiale (ignorée, utilisé pour compatibilité)
            
        Returns:
            Intensité de la source pour ce pas de temps
        """
        if self.current_sequence is None or step >= len(self.current_sequence):
            return 0.0
            
        return self.current_sequence[step]
    
    def sample_source_intensity(self, epoch_progress: float) -> float:
        """
        Échantillonne l'intensité initiale de la source pour cette simulation.
        Compatible avec l'interface du Stage 4.
        
        Args:
            epoch_progress: Progression dans l'entraînement (0.0 à 1.0)
            
        Returns:
            Intensité initiale échantillonnée
        """
        return self.attenuation_manager.sample_initial_intensity(epoch_progress)
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour ce stage.
        
        Args:
            global_config: Configuration globale du système
        
        Returns:
            Paramètres d'entraînement
        """
        return {
            'batch_size': self.config.batch_size or global_config.BATCH_SIZE,
        }
    
    def validate_convergence(self, recent_losses: List[float], epoch_in_stage: int) -> bool:
        """
        Vérifie si l'entraînement du Stage 5 a convergé.
        
        Args:
            recent_losses: Historique récent des pertes
            epoch_in_stage: Époque actuelle dans ce stage
            
        Returns:
            True si le stage a convergé
        """
        if len(recent_losses) < 10:
            return False
            
        # Convergence si perte < seuil pendant 5 époques consécutives
        recent_convergence = all(
            loss < self.config.convergence_threshold
            for loss in recent_losses[-5:]
        )
        
        return recent_convergence and epoch_in_stage >= 20
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Retourne les poids pour la fonction de perte.
        
        Returns:
            Dictionnaire des poids de la perte
        """
        # Pondération plus forte pour Stage 5
        return {'mse': 1.5}
    
    def get_intensity_statistics(self) -> Dict[str, float]:
        """
        Retourne les statistiques des intensités utilisées.
        
        Returns:
            Statistiques d'intensité
        """
        return self.attenuation_manager.get_intensity_statistics()
