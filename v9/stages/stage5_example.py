"""
Exemple de Stage 5 : Stage personnalisé avec sources multiples.
Démontre la facilité d'ajout de nouveaux stages dans l'architecture modulaire.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from .base_stage import BaseStage, StageConfig, StageEnvironmentValidator


class Stage5Config(StageConfig):
    """Configuration pour le Stage 5 - Sources multiples."""
    
    def __init__(self):
        super().__init__(
            stage_id=5,
            name="Sources multiples",
            description="Apprentissage avec plusieurs sources de chaleur simultanées",
            epochs_ratio=0.15,  # 15% des époques totales
            convergence_threshold=0.0003,  # Seuil légèrement plus permissif
            learning_rate_multiplier=0.3,  # LR très réduit pour la complexité
            min_obstacles=1,
            max_obstacles=3
        )
        
        # Configuration spéciale pour sources multiples
        self.min_sources = 2
        self.max_sources = 4
        self.source_intensity_variation = 0.2  # Variation entre les sources


class Stage5(BaseStage):
    """
    Stage 5 : Apprentissage avec sources multiples.
    Exemple d'extension facile de l'architecture modulaire.
    """
    
    def __init__(self, device: str = "cpu", min_obstacle_size: int = 2, max_obstacle_size: int = 4):
        config = Stage5Config()
        super().__init__(config, device)
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.source_positions = []  # Stockage des positions des sources
        self.source_intensities = []  # Stockage des intensités des sources
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement avec obstacles adaptés aux sources multiples.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la première source (les autres seront générées)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles optimisé pour sources multiples
        """
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)

        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None

        # Génération des positions de sources multiples
        self._generate_multiple_sources(size, source_pos, seed)
        
        # Nombre d'obstacles adapté
        n_obstacles = torch.randint(
            self.config.min_obstacles,
            self.config.max_obstacles + 1,
            (1,),
            generator=g,
            device=self.device
        ).item()

        placed_obstacles = []

        # Placement d'obstacles en évitant toutes les sources
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

            # Placement avec validation multi-sources
            for attempt in range(100):
                i = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
                j = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()

                if self._is_valid_position_multi_source(i, j, obstacle_size, placed_obstacles):
                    obstacle_mask[i:i+obstacle_size, j:j+obstacle_size] = True
                    placed_obstacles.append((i, j, obstacle_size))
                    break

        # Validation de connectivité pour sources multiples
        connectivity_ok = True
        for source_pos in self.source_positions:
            if not StageEnvironmentValidator.validate_connectivity(obstacle_mask, source_pos,
                                                                 min_connectivity_ratio=0.3):
                connectivity_ok = False
                break

        if not connectivity_ok:
            # Fallback vers environnement plus simple
            return self._generate_simple_multi_source_environment(size)

        return obstacle_mask
    
    def _generate_multiple_sources(self, size: int, initial_source: Tuple[int, int], seed: Optional[int] = None):
        """Génère les positions et intensités des sources multiples."""
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed + 1000)  # Seed différent pour les sources
        else:
            g = None

        # Nombre de sources
        n_sources = torch.randint(
            self.config.min_sources,
            self.config.max_sources + 1,
            (1,),
            generator=g,
            device=self.device
        ).item()

        self.source_positions = [initial_source]  # Première source
        self.source_intensities = [1.0]  # Intensité de référence

        # Génération des sources supplémentaires
        for source_idx in range(1, n_sources):
            # Position éloignée des sources existantes
            for attempt in range(50):
                i = torch.randint(2, size-2, (1,), generator=g, device=self.device).item()
                j = torch.randint(2, size-2, (1,), generator=g, device=self.device).item()
                
                # Vérifier distance minimale des autres sources
                min_distance = min(
                    max(abs(i - si), abs(j - sj))
                    for si, sj in self.source_positions
                )
                
                if min_distance >= 4:  # Distance minimale de 4 cellules
                    self.source_positions.append((i, j))
                    
                    # Intensité avec variation
                    base_intensity = 1.0
                    variation = (torch.rand(1, generator=g, device=self.device).item() - 0.5) * 2
                    intensity = base_intensity + variation * self.config.source_intensity_variation
                    intensity = max(0.3, min(1.0, intensity))  # Borné entre 0.3 et 1.0
                    
                    self.source_intensities.append(intensity)
                    break
    
    def _is_valid_position_multi_source(self, i: int, j: int, obstacle_size: int,
                                      placed_obstacles: List[Tuple[int, int, int]]) -> bool:
        """Validation de position pour environnement multi-sources."""
        # 1. Pas de chevauchement avec les sources
        for source_i, source_j in self.source_positions:
            if (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
                return False
        
        # 2. Pas de chevauchement avec obstacles existants
        for obs_i, obs_j, obs_size in placed_obstacles:
            if (i < obs_i + obs_size and i + obstacle_size > obs_i and
                j < obs_j + obs_size and j + obstacle_size > obs_j):
                return False
        
        # 3. Distance minimale des sources pour éviter l'isolement
        for source_i, source_j in self.source_positions:
            distance = max(abs(i + obstacle_size//2 - source_i),
                         abs(j + obstacle_size//2 - source_j))
            if distance < 2:
                return False
        
        return True
    
    def _generate_simple_multi_source_environment(self, size: int) -> torch.Tensor:
        """Environnement de fallback simple pour sources multiples."""
        return torch.zeros((size, size), dtype=torch.bool, device=self.device)
    
    def get_multi_source_data(self) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Interface pour récupérer les données des sources multiples."""
        return self.source_positions.copy(), self.source_intensities.copy()
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour le Stage 5.
        
        Returns:
            Paramètres d'entraînement optimisés pour sources multiples
        """
        return {
            'cache_size': 0,  # Pas de cache car sources variables
            'use_cache': False,  # Génération à la volée
            'shuffle_frequency': 0,  # Non applicable
            'source_intensity': None,  # Sources multiples
            'validation_frequency': 4,  # Validation très fréquente
            'obstacle_validation': True,
            'multi_source_validation': True,  # Validation spéciale multi-sources
            'min_connectivity_ratio': 0.3,  # Plus permissif
        }
    
    def validate_convergence(self, recent_losses: List[float],
                           epoch_in_stage: int) -> bool:
        """
        Critères de convergence pour le Stage 5.
        Adapté à la complexité des sources multiples.
        """
        # Minimum 35 époques pour l'apprentissage des sources multiples
        if epoch_in_stage < 35 or len(recent_losses) < 25:
            return False
        
        # Convergence sur fenêtre très étendue
        avg_recent_loss = sum(recent_losses[-25:]) / 25
        converged = avg_recent_loss < self.config.convergence_threshold
        
        # Stabilité très permissive pour la complexité
        if len(recent_losses) >= 20:
            last_20 = recent_losses[-20:]
            variance = sum((x - sum(last_20)/20)**2 for x in last_20) / 20
            stable = variance < 0.005  # Très permissif
        else:
            stable = False
        
        # Progression à long terme
        if len(recent_losses) >= 35:
            early = sum(recent_losses[-35:-25]) / 10
            late = sum(recent_losses[-10:]) / 10
            progressing = late <= early * 1.15  # Max 15% de remontée
        else:
            progressing = True
        
        return converged and stable and progressing
    
    def get_learning_rate_schedule(self, epoch_in_stage: int,
                                 max_epochs: int, base_lr: float) -> float:
        """
        Schedule LR spécialisé pour le Stage 5.
        Très conservateur pour les sources multiples.
        """
        stage_lr = base_lr * self.config.learning_rate_multiplier
        progress = epoch_in_stage / max_epochs
        
        # Phase 1 : Warm-up très long (25% des époques)
        if progress < 0.25:
            warmup_factor = progress / 0.25
            final_lr = stage_lr * (0.1 + 0.9 * warmup_factor)
        # Phase 2 : Plateau étendu (50% des époques)
        elif progress < 0.75:
            final_lr = stage_lr * 0.7
        # Phase 3 : Décroissance très douce
        else:
            adjusted_progress = (progress - 0.75) / 0.25
            cos_factor = 0.5 * (1 + np.cos(np.pi * adjusted_progress))
            final_lr = stage_lr * (0.3 + 0.4 * cos_factor)
        
        return final_lr
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Poids des pertes pour le Stage 5."""
        return {
            'mse': 1.0,
            'convergence': 1.0,
            'stability': 3.0,  # Très important pour sources multiples
            'robustness': 2.5,
            'multi_source_coherence': 2.0,  # Nouveau : cohérence entre sources
            'spatial_distribution': 1.5,  # Nouveau : distribution spatiale
        }
    
    def post_epoch_hook(self, epoch_in_stage: int, loss: float,
                       metrics: Dict[str, Any]) -> None:
        """Hook post-époque pour le Stage 5."""
        super().post_epoch_hook(epoch_in_stage, loss, metrics)
        
        # Métriques spécifiques au Stage 5
        stage_metrics = {
            'convergence_progress': max(0, 1 - loss / self.config.convergence_threshold),
            'multi_source_mastery': self._calculate_multi_source_mastery(),
            'spatial_coherence': self._calculate_spatial_coherence(),
            'complexity_handling': self._calculate_complexity_handling(),
            'source_count': len(self.source_positions),
            'intensity_diversity': self._calculate_intensity_diversity(),
        }
        
        for key, value in stage_metrics.items():
            if key not in self.training_history['metrics']:
                self.training_history['metrics'][key] = []
            self.training_history['metrics'][key].append(value)
    
    def _calculate_multi_source_mastery(self) -> float:
        """Score de maîtrise des sources multiples."""
        if len(self.training_history['losses']) < 10:
            return 0.0
        
        # Basé sur la stabilité relative à la complexité
        recent_losses = self.training_history['losses'][-10:]
        mean_loss = sum(recent_losses) / len(recent_losses)
        
        # Score inversement proportionnel à la perte moyenne
        if self.config.convergence_threshold > 0:
            mastery = max(0, 1 - mean_loss / (self.config.convergence_threshold * 2))
        else:
            mastery = 0.5
        
        return min(1.0, mastery)
    
    def _calculate_spatial_coherence(self) -> float:
        """Cohérence spatiale basée sur la distribution des sources."""
        if len(self.source_positions) < 2:
            return 1.0  # Parfait si une seule source
        
        # Calcul de la dispersion spatiale des sources
        positions = np.array(self.source_positions)
        center = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - center) for pos in positions]
        
        # Score basé sur la distribution (ni trop regroupé, ni trop dispersé)
        mean_distance = np.mean(distances)
        optimal_distance = 5.0  # Distance optimale du centre
        
        coherence = max(0, 1 - abs(mean_distance - optimal_distance) / optimal_distance)
        return coherence
    
    def _calculate_complexity_handling(self) -> float:
        """Gestion de la complexité multi-sources."""
        if len(self.training_history['losses']) < 15:
            return 0.0
        
        # Amélioration sur long terme malgré la complexité
        early_losses = self.training_history['losses'][:5]
        recent_losses = self.training_history['losses'][-5:]
        
        early_avg = sum(early_losses) / len(early_losses)
        recent_avg = sum(recent_losses) / len(recent_losses)
        
        if early_avg > 0:
            improvement = (early_avg - recent_avg) / early_avg
            handling = max(0, min(1.0, improvement * 3))  # Normalisé
        else:
            handling = 0.5
        
        return handling
    
    def _calculate_intensity_diversity(self) -> float:
        """Diversité des intensités des sources."""
        if len(self.source_intensities) <= 1:
            return 0.0
        
        # Score basé sur la variance des intensités
        intensities = np.array(self.source_intensities)
        variance = np.var(intensities)
        
        # Normalisation (variance optimale autour de 0.02)
        optimal_variance = 0.02
        diversity = max(0, 1 - abs(variance - optimal_variance) / optimal_variance)
        
        return min(1.0, diversity)
