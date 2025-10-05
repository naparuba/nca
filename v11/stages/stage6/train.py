"""
Stage 6 : Apprentissage avec sources multiples sans atténuation.
Gestion de plusieurs sources avec intensités constantes (sans atténuation temporelle).
"""

from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
import torch

from ..base_stage import BaseStage, StageConfig
from ..environment_generator import EnvironmentGenerator


class Stage6Config(StageConfig):
    """Configuration spécialisée pour le Stage 6 - Sources multiples sans atténuation."""
    
    def __init__(self):
        super().__init__(
            stage_id=6,
            name="Sources Multiples Sans Atténuation",
            description="Apprentissage avec plusieurs sources d'intensité constante (pas d'atténuation dans le temps)",
            epochs_ratio=0.167,  # Ajusté pour répartition équilibrée entre 6 stages
            convergence_threshold=0.0000005,  # Seuil très fin pour la précision des interactions
            learning_rate_multiplier=0.2,  # LR extrêmement réduit pour ce stage complexe
            min_obstacles=1,
            max_obstacles=2
        )
        
        # Configuration pour les sources multiples
        self.min_sources = 1
        self.max_sources = 3
        
        # Configuration d'intensité (sans atténuation)
        self.min_source_intensity = 0.3
        self.max_source_intensity = 1.0
        self.initial_intensity_range = [0.3, 1.0]
        
        # Curriculum d'apprentissage pour les sources multiples (sans atténuation)
        self.curriculum_phases = [
            # Phase 1: 1 source unique, intensités moyennes-fortes
            {"progress": 0.25, "sources": (1, 1), "intensity_range": (0.5, 1.0)},
            # Phase 2: 1-2 sources, intensités moyennes-fortes
            {"progress": 0.5, "sources": (1, 2), "intensity_range": (0.4, 1.0)},
            # Phase 3: 2 sources, intensités variables
            {"progress": 0.75, "sources": (2, 2), "intensity_range": (0.3, 1.0)},
            # Phase 4: 1-3 sources, pleine plage d'intensités
            {"progress": 1.0, "sources": (1, 3), "intensity_range": (0.3, 1.0)}
        ]


class ConstantSourceManager:
    """
    Gestionnaire spécialisé pour les sources multiples à intensité constante du Stage 6.
    Gère plusieurs sources, chacune avec sa propre intensité (sans atténuation).
    """
    
    def __init__(self, config: Stage6Config, device: str = "cpu"):
        self.config = config
        self.device = device
        self.current_phase = 0
        
        # Historiques pour analyse
        self.sources_count_history = []
        self.intensity_history = []  # Liste de listes (une par source)
        
        # Séquences temporelles actuelles
        self.current_sequences = {}  # Dictionnaire {sequence_id: {source_id: intensité}}
        self.current_sequence_counter = 0
    
    def sample_sources_count(self, epoch_progress: float) -> int:
        """
        Détermine le nombre de sources à utiliser selon la phase du curriculum.
        
        Args:
            epoch_progress: Progression dans l'entraînement (0.0 à 1.0)
            
        Returns:
            Nombre de sources à utiliser
        """
        # Détermination de la phase actuelle du curriculum
        phase = self._get_current_phase(epoch_progress)
        
        # Échantillonnage du nombre de sources
        min_sources, max_sources = phase["sources"]
        sources_count = random.randint(min_sources, max_sources)
        
        # Stockage pour statistiques
        self.sources_count_history.append(sources_count)
        
        return sources_count
    
    def sample_source_positions(self, grid_size: int, n_sources: int,
                              existing_obstacles: torch.Tensor = None,
                              min_distance: int = 3) -> List[Tuple[int, int]]:
        """
        Échantillonne les positions des sources en évitant les obstacles
        et en maintenant une distance minimale entre elles.
        
        Args:
            grid_size: Taille de la grille
            n_sources: Nombre de sources à placer
            existing_obstacles: Masque des obstacles existants
            min_distance: Distance minimale entre les sources
            
        Returns:
            Liste de tuples (i, j) pour chaque source
        """
        if existing_obstacles is None:
            available_mask = torch.ones(grid_size, grid_size, dtype=torch.bool, device=self.device)
        else:
            available_mask = ~existing_obstacles  # Inverse pour avoir les positions disponibles
        
        # Évitement des bords (marge de 1)
        available_mask[0, :] = False
        available_mask[-1, :] = False
        available_mask[:, 0] = False
        available_mask[:, -1] = False
        
        # Conversion en coordonnées
        available_coords = torch.nonzero(available_mask).tolist()
        
        if len(available_coords) < n_sources:
            raise RuntimeError(f"Impossible de placer {n_sources} sources - pas assez de positions disponibles")
        
        source_positions = []
        for _ in range(n_sources):
            if not available_coords:
                raise RuntimeError("Plus de positions disponibles pour les sources")
            
            # Sélection aléatoire d'une position
            idx = random.randint(0, len(available_coords) - 1)
            pos = tuple(available_coords[idx])
            source_positions.append(pos)
            
            # Mise à jour des positions disponibles (retrait des positions trop proches)
            new_available = []
            for coord in available_coords:
                i, j = coord
                too_close = False
                for src_i, src_j in source_positions:
                    if abs(i - src_i) + abs(j - src_j) < min_distance:  # Distance de Manhattan
                        too_close = True
                        break
                if not too_close:
                    new_available.append(coord)
            
            available_coords = new_available
        
        return source_positions
    
    def sample_source_intensities(self, epoch_progress: float, n_sources: int) -> List[float]:
        """
        Échantillonne les intensités initiales pour chaque source (sans atténuation).
        
        Args:
            epoch_progress: Progression dans l'étape 6 (0.0 à 1.0)
            n_sources: Nombre de sources
            
        Returns:
            Liste des intensités pour chaque source
        """
        phase = self._get_current_phase(epoch_progress)
        min_intensity, max_intensity = phase["intensity_range"]
        
        # Échantillonnage des intensités constantes pour chaque source
        intensities = []
        for _ in range(n_sources):
            # Distribution uniforme pour chaque source
            intensity = min_intensity + (max_intensity - min_intensity) * torch.rand(1, device=self.device).item()
            
            # Validation et ajustement
            intensity = max(self.config.min_source_intensity, min(self.config.max_source_intensity, intensity))
            intensities.append(intensity)
        
        # Historique pour statistiques
        self.intensity_history.append(intensities)
        
        return intensities
    
    def get_source_intensities_sequence(self, source_intensities: List[float],
                                      sequence_length: int) -> Dict[int, List[float]]:
        """
        Crée des séquences d'intensités constantes pour chaque source.
        
        Args:
            source_intensities: Intensités initiales pour chaque source
            sequence_length: Longueur de la séquence temporelle
            
        Returns:
            Dictionnaire {source_id: list d'intensités constantes}
        """
        # Création d'une séquence avec intensité constante pour chaque source
        sequence_id = self.current_sequence_counter
        self.current_sequence_counter += 1
        
        sequences = {}
        for i, intensity in enumerate(source_intensities):
            # Création d'une séquence d'intensité constante (pas d'atténuation)
            sequences[i] = [intensity] * sequence_length
            
        # Mémorisation de la séquence courante
        self.current_sequences[sequence_id] = sequences
        
        return sequences
    
    def _get_current_phase(self, epoch_progress: float) -> Dict[str, Any]:
        """
        Détermine la phase actuelle du curriculum en fonction de la progression.
        
        Args:
            epoch_progress: Progression dans l'entraînement (0.0 à 1.0)
            
        Returns:
            Dictionnaire de configuration de la phase
        """
        for i, phase in enumerate(self.config.curriculum_phases):
            if epoch_progress <= phase["progress"]:
                if i != self.current_phase:
                    self.current_phase = i
                return phase
        
        # Fallback au cas où (ne devrait pas arriver si progress va jusqu'à 1.0)
        return self.config.curriculum_phases[-1]


class Stage6(BaseStage):
    """
    Stage 6 : Apprentissage avec sources multiples sans atténuation.
    """
    
    def __init__(self, device: str = "cpu"):
        config = Stage6Config()
        super().__init__(config, device)
        
        # Composants spécialisés pour le Stage 6
        self.env_generator = EnvironmentGenerator(device)
        self.source_manager = ConstantSourceManager(config, device)
        
        # Métriques spécifiques au stage
        self.training_history['metrics']['sources_count'] = []
        self.training_history['metrics']['source_intensities'] = []
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int],
                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère l'environnement d'obstacles pour ce stage.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source principale (ignorée car sources multiples)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles [H, W]
        """
        # Nombre d'obstacles variable selon la configuration du stage
        min_obstacles = self.config.min_obstacles
        max_obstacles = self.config.max_obstacles
        
        return self.env_generator.generate_complex_environment(
            size=size,
            source_pos=source_pos,  # Position de référence (sera remplacée par multiple)
            min_obstacles=min_obstacles,
            max_obstacles=max_obstacles,
            min_obstacle_size=2,
            max_obstacle_size=4,
            seed=seed
        )
    
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]:
        """
        Prépare les données d'entraînement spécifiques à ce stage.
        
        Args:
            global_config: Configuration globale du système
            
        Returns:
            Dictionnaire avec les paramètres d'entraînement
        """
        # Conversion des époques totales à celles pour ce stage
        total_epochs = global_config.TOTAL_EPOCHS
        stage_epochs = int(total_epochs * self.config.epochs_ratio)
        
        # Taille de batch spécifique au stage ou globale
        batch_size = self.config.batch_size if self.config.batch_size else global_config.BATCH_SIZE
        
        # Calcul du learning rate ajusté
        learning_rate = global_config.LEARNING_RATE * self.config.learning_rate_multiplier
        
        return {
            'stage_epochs': stage_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'supports_multiple_sources': True,  # Indicateur pour le simulateur
            'use_constant_sources': True        # Indicateur pour le simulateur: sources sans atténuation
        }
    
    def validate_convergence(self, recent_losses: List[float], epoch_in_stage: int) -> bool:
        """
        Détermine si le stage a convergé selon ses critères spécifiques.
        
        Args:
            recent_losses: Pertes récentes
            epoch_in_stage: Époque actuelle dans le stage
            
        Returns:
            True si le stage a convergé
        """
        # Vérification du nombre minimum d'époques (au moins 20% du stage)
        min_epochs = int(self.config.epochs_ratio * 0.2)
        if epoch_in_stage < min_epochs:
            return False
            
        # Vérification de la variation des pertes récentes
        if len(recent_losses) >= 10:
            loss_std = np.std(recent_losses[-10:])
            if loss_std < self.config.convergence_threshold:
                # Convergence détectée
                return True
                
        return False
    
    def prepare_simulation_data(self, epoch_in_stage: int, stage_epochs: int,
                              grid_size: int, obstacles: torch.Tensor) -> Dict[str, Any]:
        """
        Prépare les données de simulation spécifiques pour ce stage.
        
        Args:
            epoch_in_stage: Époque actuelle dans le stage
            stage_epochs: Nombre total d'époques pour ce stage
            grid_size: Taille de la grille
            obstacles: Masque des obstacles
            
        Returns:
            Dictionnaire avec les paramètres de simulation
        """
        # Calcul de la progression dans ce stage (0.0 à 1.0)
        epoch_progress = epoch_in_stage / stage_epochs
        
        # Détermination du nombre de sources
        n_sources = self.source_manager.sample_sources_count(epoch_progress)
        
        # Placement des sources
        source_positions = self.source_manager.sample_source_positions(
            grid_size=grid_size,
            n_sources=n_sources,
            existing_obstacles=obstacles
        )
        
        # Détermination des intensités constantes (sans atténuation)
        source_intensities = self.source_manager.sample_source_intensities(
            epoch_progress=epoch_progress,
            n_sources=n_sources
        )
        
        # Métriques pour historique
        self.training_history['metrics']['sources_count'].append(n_sources)
        self.training_history['metrics']['source_intensities'].append(source_intensities)
        
        return {
            'source_positions': source_positions,
            'source_intensities': source_intensities,
            'use_constant_sources': True  # Pas d'atténuation
        }
