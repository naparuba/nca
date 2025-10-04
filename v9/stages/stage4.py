"""
Stage 4 : Apprentissage avec intensités variables.
Gestion avancée avec intensités de source dynamiques.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base_stage import BaseStage, StageConfig, StageEnvironmentValidator


class Stage4Config(StageConfig):
    """Configuration spécialisée pour le Stage 4."""
    
    def __init__(self):
        super().__init__(
            stage_id=4,
            name="Intensités variables",
            description="Apprentissage avec intensités de source variables et obstacles",
            epochs_ratio=0.2,
            convergence_threshold=0.0002,
            learning_rate_multiplier=0.4,  # LR le plus réduit
            min_obstacles=1,
            max_obstacles=2
        )
        
        # Configuration spéciale pour intensités variables
        self.min_source_intensity = 0.0
        self.max_source_intensity = 1.0
        self.initial_intensity_range = [0.5, 1.0]  # Plage initiale restreinte
        self.final_intensity_range = [0.0, 1.0]    # Plage finale complète
        self.intensity_distribution = 'uniform'


class IntensityManager:
    """
    Gestionnaire spécialisé pour les intensités variables du Stage 4.
    Complètement découplé des autres stages.
    """
    
    def __init__(self, config: Stage4Config, device: str = "cpu"):
        self.config = config
        self.device = device
        self.intensity_history = []
    
    def sample_simulation_intensity(self, epoch_progress: float) -> float:
        """
        Échantillonne une intensité selon l'avancement de l'entraînement.
        
        Args:
            epoch_progress: Progression dans l'étape 4 (0.0 à 1.0)
            
        Returns:
            Intensité échantillonnée pour cette simulation
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
    
    def clear_history(self):
        """Efface l'historique pour économiser la mémoire."""
        if len(self.intensity_history) > 1000:
            self.intensity_history = self.intensity_history[-1000:]


class Stage4(BaseStage):
    """
    Stage 4 : Apprentissage avec intensités variables.
    Le stage le plus avancé avec gestion des intensités dynamiques.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = Stage4Config()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.intensity_manager = IntensityManager(config, device)
        self.placement_attempts = 50
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement pour les intensités variables.
        Environnement modéré pour se concentrer sur l'apprentissage des intensités.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles modéré pour Stage 4
        """
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)

        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None

        # Nombre d'obstacles modéré (focus sur les intensités)
        n_obstacles = torch.randint(
            self.config.min_obstacles,
            self.config.max_obstacles + 1,
            (1,),
            generator=g,
            device=self.device
        ).item()

        source_i, source_j = source_pos
        placed_obstacles = []

        # Placement d'obstacles avec contraintes simplifiées
        for obstacle_idx in range(n_obstacles):
            obstacle_size = torch.randint(
                self.min_obstacle_size,
                self.max_obstacle_size + 1,
                (1,),
                generator=g,
                device=self.device
            ).item()

            max_pos = size - obstacle_size
            if max_pos <= 1:
                continue

            # Placement avec validation simplifiée (focus sur intensités)
            for attempt in range(self.placement_attempts):
                i = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
                j = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()

                if self._is_valid_position_stage4(i, j, obstacle_size, source_pos, placed_obstacles):
                    obstacle_mask[i:i+obstacle_size, j:j+obstacle_size] = True
                    placed_obstacles.append((i, j, obstacle_size))
                    break

        # Validation de connectivité avec seuil permissif
        if not StageEnvironmentValidator.validate_connectivity(obstacle_mask, source_pos,
                                                             min_connectivity_ratio=0.6):
            # Fallback vers environnement simple
            return self._generate_simple_environment(size, source_pos, seed)

        return obstacle_mask
    
    def _is_valid_position_stage4(self, i: int, j: int, obstacle_size: int,
                                 source_pos: Tuple[int, int],
                                 placed_obstacles: List[Tuple[int, int, int]]) -> bool:
        """Validation simplifiée pour Stage 4 (focus intensités)."""
        source_i, source_j = source_pos
        
        # 1. Pas de chevauchement avec source
        if (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
            return False
        
        # 2. Pas de chevauchement avec obstacles existants
        for obs_i, obs_j, obs_size in placed_obstacles:
            if (i < obs_i + obs_size and i + obstacle_size > obs_i and
                j < obs_j + obs_size and j + obstacle_size > obs_j):
                return False
        
        return True
    
    def _generate_simple_environment(self, size: int, source_pos: Tuple[int, int],
                                   seed: Optional[int] = None) -> torch.Tensor:
        """Environnement de fallback simple pour Stage 4."""
        # Import conditionnel pour éviter la dépendance circulaire
        from .stage2 import Stage2
        
        fallback_stage = Stage2(self.device, self.min_obstacle_size, self.max_obstacle_size)
        return fallback_stage.generate_environment(size, source_pos, seed)
    
    def sample_source_intensity(self, epoch_progress: float) -> float:
        """
        Échantillonne une intensité de source pour cette simulation.
        Interface publique du gestionnaire d'intensités.
        """
        return self.intensity_manager.sample_simulation_intensity(epoch_progress)
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour le Stage 4.
        
        Returns:
            Paramètres d'entraînement optimisés pour les intensités variables
        """
        return {
            'cache_size': 0,  # Pas de cache car intensités dynamiques
            'use_cache': False,  # Génération à la volée requise
            'shuffle_frequency': 0,  # Non applicable
            'source_intensity': None,  # Intensité variable
            'validation_frequency': 5,  # Validation très fréquente
            'obstacle_validation': True,
            'intensity_validation': True,  # Validation spéciale des intensités
            'intensity_curriculum': True,  # Curriculum d'intensités
            'min_connectivity_ratio': 0.6,  # Connectivité permissive
        }
    
    def validate_convergence(self, recent_losses: List[float],
                           epoch_in_stage: int) -> bool:
        """
        Critères de convergence pour le Stage 4.
        Adapté à la variabilité des intensités.
        
        Args:
            recent_losses: Pertes récentes
            epoch_in_stage: Époque courante dans le stage
            
        Returns:
            True si convergé
        """
        # Minimum 30 époques pour l'apprentissage des intensités variables
        if epoch_in_stage < 30 or len(recent_losses) < 20:
            return False
        
        # Convergence sur fenêtre étendue (intensités variables = plus de variance)
        avg_recent_loss = sum(recent_losses[-20:]) / 20
        converged = avg_recent_loss < self.config.convergence_threshold
        
        # Stabilité adaptée aux intensités variables
        if len(recent_losses) >= 15:
            last_15 = recent_losses[-15:]
            variance = sum((x - sum(last_15)/15)**2 for x in last_15) / 15
            stable = variance < 0.003  # Seuil plus permissif pour intensités variables
        else:
            stable = False
        
        # Critère de progression constante sur long terme
        if len(recent_losses) >= 30:
            early_period = sum(recent_losses[-30:-20]) / 10
            recent_period = sum(recent_losses[-10:]) / 10
            progressing = recent_period <= early_period * 1.1  # Max 10% de remontée
        else:
            progressing = True
        
        return converged and stable and progressing
    
    def get_learning_rate_schedule(self, epoch_in_stage: int,
                                 max_epochs: int, base_lr: float) -> float:
        """
        Schedule LR ultra-spécialisé pour le Stage 4.
        Adapté à l'apprentissage des intensités variables.
        """
        stage_lr = base_lr * self.config.learning_rate_multiplier
        progress = epoch_in_stage / max_epochs
        
        # Phase 1 : Warm-up très étendu (20% des époques)
        if progress < 0.2:
            warmup_factor = progress / 0.2
            final_lr = stage_lr * (0.2 + 0.8 * warmup_factor)
        # Phase 2 : Plateau d'adaptation (40% des époques)
        elif progress < 0.6:
            final_lr = stage_lr * 0.8
        # Phase 3 : Fine-tuning très graduel
        else:
            adjusted_progress = (progress - 0.6) / 0.4
            cos_factor = 0.5 * (1 + np.cos(np.pi * adjusted_progress))
            final_lr = stage_lr * (0.25 + 0.55 * cos_factor)
        
        return final_lr
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Poids des pertes pour le Stage 4."""
        return {
            'mse': 1.0,
            'convergence': 1.2,
            'stability': 2.5,  # Très important pour intensités variables
            'robustness': 2.0,
            'intensity_adaptation': 2.0,  # Nouveau : adaptation aux intensités
            'dynamic_learning': 1.5,  # Nouveau : apprentissage dynamique
        }
    
    def post_epoch_hook(self, epoch_in_stage: int, loss: float,
                       metrics: Dict[str, Any]) -> None:
        """Hook post-époque pour le Stage 4."""
        super().post_epoch_hook(epoch_in_stage, loss, metrics)
        
        # Métriques spécifiques au Stage 4
        intensity_stats = self.intensity_manager.get_intensity_statistics()
        
        stage_metrics = {
            'convergence_progress': max(0, 1 - loss / self.config.convergence_threshold),
            'intensity_adaptation_score': self._calculate_intensity_adaptation(),
            'dynamic_stability': self._calculate_dynamic_stability(),
            'intensity_range_mastery': self._calculate_range_mastery(epoch_in_stage),
            'current_intensity_mean': intensity_stats.get('mean', 0.0),
            'current_intensity_std': intensity_stats.get('std', 0.0),
        }
        
        for key, value in stage_metrics.items():
            if key not in self.training_history['metrics']:
                self.training_history['metrics'][key] = []
            self.training_history['metrics'][key].append(value)
    
    def _calculate_intensity_adaptation(self) -> float:
        """Score d'adaptation aux intensités variables."""
        if len(self.training_history['losses']) < 10:
            return 0.0
        
        # Analyse de la capacité à gérer différentes intensités
        recent_losses = self.training_history['losses'][-10:]
        
        # Score basé sur la stabilité malgré la variabilité des intensités
        mean_loss = sum(recent_losses) / len(recent_losses)
        max_deviation = max(abs(loss - mean_loss) for loss in recent_losses)
        
        if mean_loss > 0:
            adaptation_score = max(0, 1 - max_deviation / mean_loss)
        else:
            adaptation_score = 1.0
        
        return min(1.0, adaptation_score)
    
    def _calculate_dynamic_stability(self) -> float:
        """Stabilité dynamique adaptée aux intensités variables."""
        if len(self.training_history['losses']) < 15:
            return 0.0
        
        # Stabilité sur fenêtre glissante
        recent_losses = self.training_history['losses'][-15:]
        
        # Calcul de stabilité par segments
        segment_size = 5
        segment_variances = []
        
        for i in range(0, len(recent_losses) - segment_size + 1, segment_size):
            segment = recent_losses[i:i + segment_size]
            mean_seg = sum(segment) / len(segment)
            variance = sum((x - mean_seg)**2 for x in segment) / len(segment)
            segment_variances.append(variance)
        
        if segment_variances:
            avg_variance = sum(segment_variances) / len(segment_variances)
            stability = max(0, 1 - avg_variance * 500)  # Normalisé pour intensités variables
        else:
            stability = 0.0
        
        return min(1.0, stability)
    
    def _calculate_range_mastery(self, epoch_in_stage: int) -> float:
        """Maîtrise de la plage d'intensités selon la progression."""
        max_epochs = len(self.training_history['losses']) + epoch_in_stage
        if max_epochs == 0:
            return 0.0
        
        progress = epoch_in_stage / max(max_epochs, 1)
        
        # Maîtrise basée sur la progression et les statistiques d'intensité
        intensity_stats = self.intensity_manager.get_intensity_statistics()
        
        if intensity_stats['count'] > 0:
            # Score basé sur la couverture de la plage d'intensités
            range_coverage = intensity_stats['max'] - intensity_stats['min']
            expected_range = self.config.max_source_intensity - self.config.min_source_intensity
            
            if expected_range > 0:
                coverage_score = range_coverage / expected_range
            else:
                coverage_score = 1.0
            
            # Combiné avec la progression temporelle
            mastery = min(1.0, coverage_score * (1 + progress) / 2)
        else:
            mastery = 0.0
        
        return mastery
    
    def get_intensity_statistics(self) -> Dict[str, float]:
        """Interface publique pour les statistiques d'intensités."""
        return self.intensity_manager.get_intensity_statistics()
    
    def clear_intensity_history(self):
        """Interface publique pour nettoyer l'historique des intensités."""
        self.intensity_manager.clear_history()
