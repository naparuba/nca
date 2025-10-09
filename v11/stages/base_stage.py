"""
Architecture de base pour les stages modulaires du NCA.
Permet l'ajout facile de nouveaux stages sans impact sur les existants.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import torch
from dataclasses import dataclass


@dataclass
class StageConfig:
    """
    Configuration de base pour un stage d'entraînement.
    
    Le 'name' sert maintenant d'identifiant unique (slug) pour le stage.
    Plus besoin de stage_id - le découplage est total.
    """
    name: str  # Slug unique du stage (ex: 'no_obstacles', 'single_obstacle')
    description: str
    epochs_ratio: float = 0.25  # Ratio par défaut des époques totales
    convergence_threshold: float = 0.0002
    learning_rate_multiplier: float = 1.0
    batch_size: Optional[int] = None  # None = utilise la config globale
    
    # Paramètres d'obstacles par défaut
    min_obstacles: int = 0
    max_obstacles: int = 0
    
    def __post_init__(self):
        """Validation des paramètres après initialisation."""
        if self.epochs_ratio <= 0 or self.epochs_ratio > 1:
            raise ValueError(f"epochs_ratio doit être entre 0 et 1, reçu: {self.epochs_ratio}")
        if self.convergence_threshold <= 0:
            raise ValueError(f"convergence_threshold doit être > 0, reçu: {self.convergence_threshold}")
        # Validation que name est un slug valide (sans espaces, minuscules, underscores)
        if not self.name.replace('_', '').isalnum() or ' ' in self.name:
            raise ValueError(f"Le name doit être un slug valide (lettres, chiffres, underscores): {self.name}")


class BaseStage(ABC):
    """
    Classe de base abstraite pour tous les stages d'entraînement.
    Définit l'interface commune que chaque stage doit implémenter.
    """
    
    def __init__(self, config: StageConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.training_history = {
            'losses': [],
            'epochs': [],
            'lr': [],
            'metrics': {}
        }
    
    @abstractmethod
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère l'environnement d'obstacles spécifique à ce stage.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles [H, W]
        """
        pass
    
    @abstractmethod
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement spécifiques à ce stage.
        
        Args:
            global_config: Configuration globale du système
            
        Returns:
            Dictionnaire avec les paramètres d'entraînement
        """
        pass
    
    @abstractmethod
    def validate_convergence(self, recent_losses: List[float],
                           epoch_in_stage: int) -> bool:
        """
        Détermine si le stage a convergé selon ses critères spécifiques.
        
        Args:
            recent_losses: Pertes récentes
            epoch_in_stage: Époque actuelle dans le stage
            
        Returns:
            True si le stage a convergé
        """
        pass
    
    def get_learning_rate_schedule(self, epoch_in_stage: int,
                                 max_epochs: int, base_lr: float) -> float:
        """
        Calcule le learning rate pour cette époque (méthode par défaut).
        Peut être surchargée par les stages spécialisés.
        
        Args:
            epoch_in_stage: Époque courante dans le stage
            max_epochs: Nombre maximum d'époques pour ce stage
            base_lr: Learning rate de base
            
        Returns:
            Learning rate ajusté
        """
        import numpy as np
        
        stage_lr = base_lr * self.config.learning_rate_multiplier
        
        # Décroissance cosine par défaut
        cos_factor = 0.5 * (1 + np.cos(np.pi * epoch_in_stage / max_epochs))
        final_lr = stage_lr * (0.1 + 0.9 * cos_factor)
        
        return final_lr
    
    def get_loss_weights(self) -> Dict[str, float]:
        """
        Retourne les poids de pondération des pertes pour ce stage.
        Peut être surchargée par les stages spécialisés.
        """
        return {'mse': 1.0}
    
    def get_name(self) -> str:
        return self.config.name
    
    def post_epoch_hook(self, epoch_in_stage: int, loss: float,
                       metrics: Dict[str, Any]) -> None:
        """
        Hook appelé après chaque époque d'entraînement.
        Permet aux stages d'effectuer des traitements spécifiques.
        """
        self.training_history['losses'].append(loss)
        self.training_history['epochs'].append(epoch_in_stage)
        
        # Stockage des métriques personnalisées
        for key, value in metrics.items():
            if key not in self.training_history['metrics']:
                self.training_history['metrics'][key] = []
            self.training_history['metrics'][key].append(value)
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances du stage."""
        if not self.training_history['losses']:
            return {
                'stage_id': self.config.name,
                'name': self.config.name,
                'epochs_trained': 0,
                'final_loss': float('inf'),
                'converged': False
            }
        
        return {
            'stage_id': self.config.name,
            'name': self.config.name,
            'epochs_trained': len(self.training_history['losses']),
            'final_loss': self.training_history['losses'][-1],
            'converged': self.training_history['losses'][-1] < self.config.convergence_threshold,
            'training_history': self.training_history
        }
    
    def reset_training_history(self) -> None:
        """Réinitialise l'historique d'entraînement."""
        self.training_history = {
            'losses': [],
            'epochs': [],
            'lr': [],
            'metrics': {}
        }
    
    def initialize_sequence(self, n_steps: int, progress: float = 0.0) -> None:
        """
        Initialise une séquence temporelle pour la simulation ou prédiction.
        Cette méthode peut être surchargée par les stages qui ont besoin
        d'une initialisation spécifique pour les séquences temporelles.
        
        Args:
            n_steps: Nombre de pas de temps de la séquence
            progress: Progression de l'entraînement (0.0 à 1.0)
        """
        # Méthode de base - ne fait rien par défaut
        pass
    
    def get_source_intensity_at_step(self, step: int, base_intensity: float) -> float:
        """
        Retourne l'intensité de la source à un pas de temps spécifique.
        Cette méthode peut être surchargée par les stages qui ont besoin
        de faire varier l'intensité de la source au cours du temps.
        
        Args:
            step: Pas de temps actuel
            base_intensity: Intensité de base de la source
            
        Returns:
            Intensité de la source au pas de temps spécifié
        """
        # Comportement par défaut : intensité constante
        return base_intensity


class StageEnvironmentValidator:
    """
    Utilitaire pour valider les environnements générés par les stages.
    Méthodes communes de validation utilisables par tous les stages.
    """
    
    @staticmethod
    def validate_connectivity(obstacle_mask: torch.Tensor,
                            source_pos: Tuple[int, int],
                            min_connectivity_ratio: float = 0.5) -> bool:
        """
        Valide qu'un chemin de diffusion reste possible avec les obstacles.
        
        Args:
            obstacle_mask: Masque des obstacles
            source_pos: Position de la source
            min_connectivity_ratio: Ratio minimum de cellules accessibles
            
        Returns:
            True si la connectivité est suffisante
        """
        H, W = obstacle_mask.shape
        source_i, source_j = source_pos

        # Matrice de visite
        visited = torch.zeros_like(obstacle_mask, dtype=torch.bool)
        visited[obstacle_mask] = True  # Les obstacles sont "déjà visités"

        # Flood-fill depuis la source
        stack = [(source_i, source_j)]
        visited[source_i, source_j] = True
        accessible_cells = 1

        while stack:
            i, j = stack.pop()

            # Parcours des 4 voisins
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj

                if (0 <= ni < H and 0 <= nj < W and
                    not visited[ni, nj] and not obstacle_mask[ni, nj]):
                    visited[ni, nj] = True
                    stack.append((ni, nj))
                    accessible_cells += 1

        # Vérification du ratio de connectivité
        total_free_cells = (H * W) - obstacle_mask.sum().item()
        connectivity_ratio = accessible_cells / max(total_free_cells, 1)

        return connectivity_ratio >= min_connectivity_ratio
    
    @staticmethod
    def calculate_difficulty_metrics(stage_id: int,
                                   obstacle_mask: torch.Tensor) -> Dict[str, float]:
        """
        Calcule des métriques de difficulté pour l'environnement généré.
        
        Args:
            stage_id: Identifiant du stage
            obstacle_mask: Masque des obstacles
            
        Returns:
            Dictionnaire avec les métriques de complexité
        """
        H, W = obstacle_mask.shape
        total_cells = H * W
        obstacle_cells = obstacle_mask.sum().item()

        return {
            'stage_id': stage_id,
            'obstacle_ratio': obstacle_cells / total_cells,
            'free_cells': total_cells - obstacle_cells,
            'complexity_score': stage_id * (obstacle_cells / total_cells),
            'obstacle_count': obstacle_cells
        }
