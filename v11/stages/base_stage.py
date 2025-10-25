import json
from abc import ABC
from pathlib import Path
from typing import Tuple, List

import torch
from config import CONFIG
from reality_world import RealityWorld
from simulation_temporal_sequence import SimulationTemporalSequence
from torch.nn import functional as F


class REALITY_LAYER:
    TEMPERATURE = 0
    OBSTACLE = 1


ALL_REALITY_LAYERS = [REALITY_LAYER.TEMPERATURE, REALITY_LAYER.OBSTACLE]


class BaseStage(ABC):
    NAME = 'UNSET'
    DISPLAY_NAME = 'Étape non définie'
    
    COLOR = 'black'
    
    
    def __init__(self):
        self._stage_nb = -1  # Numéro de l'étape, à définir dans les sous-classes
        
        # Metrics
        self._metrics_epochs_trained = 0
        self._metrics_loss_history = []
        self._metrics_stage_lrs = []
        
        # CACHE
        self._reality_temporal_sequences_for_training = []  # type: List[SimulationTemporalSequence]
        self._reality_temporal_sequences_for_training_current_indices = 0
        
        # Evaluation part
        self._reality_temporal_sequences_for_evaluation = []  # type: List[SimulationTemporalSequence]
        
        # step player
        self._kernel_avg_3x3 = torch.ones((1, 1, 3, 3), device=CONFIG.DEVICE) / 9.0  # Average 3x3
    
    
    def get_name(self):
        return self.NAME
    
    
    def get_display_name(self):
        return self.DISPLAY_NAME
    
    
    def get_color(self):
        return self.COLOR
    
    
    def set_stage_nb(self, stage_nb: int):
        self._stage_nb = stage_nb
    
    
    def set_metrics(self, epochs_trained: int, loss_history: list, stage_lrs: list):
        self._metrics_epochs_trained = epochs_trained
        self._metrics_loss_history = loss_history
        self._metrics_stage_lrs = stage_lrs
    
    
    def get_stage_nb(self):
        return self._stage_nb
    
    
    def generate_obstacles(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        """
                Génère un environnement d'obstacles adapté à l'étape courante.

                Args:
                    size: Taille de la grille
                    source_pos: Position de la source (i, j)

                Returns:
                    Masque des obstacles [H, W]
                """
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def _get_stage_dir(self):
        # type: () -> Path
        stage_dir = Path(CONFIG.OUTPUT_DIR) / f"stage_{self.get_stage_nb()}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    
    
    def get_metrics(self):
        """Retourne les métriques de l'étape."""
        
        return {
            'stage_nb':       self.get_stage_nb(),
            'epochs_trained': self._metrics_epochs_trained,
            'loss_history':   self._metrics_loss_history,
        }
    
    
    def get_loss_history(self):
        return self._metrics_loss_history
    
    
    def get_metrics_epochs_trained(self):
        return self._metrics_epochs_trained
    
    
    def get_metrics_lrs(self):
        return self._metrics_stage_lrs
    
    
    def save_stage_checkpoint(self, model_state_dict, optimizer_state_dict):
        # Type: (Dict, Dict) -> None
        """Sauvegarde le checkpoint d'une étape."""
        
        stage_dir = self._get_stage_dir()
        
        metrics = self.get_metrics()
        
        # Sauvegarde du modèle
        model_path = stage_dir / "model_checkpoint.pth"
        torch.save({
            'model_state_dict':     model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'stage_nb':             self.get_stage_nb(),
            'metrics':              metrics,
            'config':               CONFIG.__dict__
        }, model_path)
        
        # Sauvegarde des métriques en JSON
        metrics_path = stage_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"💾 Checkpoint étape {self.get_stage_nb()} sauvegardé: {stage_dir}")
    
    
    def get_sequences_for_training(self):
        # type: (BaseStage) -> SimulationTemporalSequence
        """Récupère un échantillon pour l'étape spécifiée."""
        stage_nb = self.get_stage_nb()
        if not self._reality_temporal_sequences_for_training:
            raise Exception("Le cache de séquences n'a pas été généré pour l'étape {stage_nb}.")
        
        # Récupère l'échantillon courant et avance l'index
        sequence = self._reality_temporal_sequences_for_training[
            self._reality_temporal_sequences_for_training_current_indices]  # type: SimulationTemporalSequence
        self._reality_temporal_sequences_for_training_current_indices = (self._reality_temporal_sequences_for_training_current_indices + 1) % len(
                self._reality_temporal_sequences_for_training)
        
        return sequence
    
    
    def generate_reality_sequences_for_training(self):
        cache_size = CONFIG.NB_EPOCHS_BY_STAGE
        print(f"🎯 Génération de {cache_size} séquences pour l'étape {self.get_stage_nb()}...", end='', flush=True)
        
        # sequences = []  # :Type: List[Sequence]
        for i in range(cache_size):
            if i % 50 == 0:
                print(f"\r   Étape {self.get_stage_nb()}: {i}/{cache_size}                                 ", end='', flush=True)
            
            simulation_temporal_sequence = self.generate_simulation_temporal_sequence(n_steps=CONFIG.NCA_STEPS, size=CONFIG.GRID_SIZE)
            
            self._reality_temporal_sequences_for_training.append(simulation_temporal_sequence)
        
        print(f"\r✅ Cache étape {self.get_stage_nb()} créé ({cache_size} séquences)")
    
    
    def generate_reality_sequences_for_evaluation(self):
        print(f"Génération de {CONFIG.NB_EPOCHS_FOR_EVALUATION} séquences pour Evaluation de {self.get_stage_nb()}...", end='', flush=True)
        
        # sequences = []  # :Type: List[Sequence]
        for i in range(CONFIG.NB_EPOCHS_FOR_EVALUATION):
            simulation_temporal_sequence = self.generate_simulation_temporal_sequence(n_steps=CONFIG.NCA_STEPS, size=CONFIG.GRID_SIZE)
            self._reality_temporal_sequences_for_evaluation.append(simulation_temporal_sequence)
    
    
    def get_sequence_for_evaluation(self):
        return self._reality_temporal_sequences_for_evaluation.pop(0)
    
    def generate_simulation_temporal_sequence(self, n_steps, size):
        # type: (int, int) -> SimulationTemporalSequence
        """
        Génère une séquence adaptée à l'étape d'apprentissage courante.

        Args:
            n_steps: Nombre d'étapes de simulation
            size: Taille de la grille

        Returns :
            (séquence, masque_source, masque_obstacles)
        """
        # Position aléatoire de la source
        g = torch.Generator(device=CONFIG.DEVICE)
        g.manual_seed(CONFIG.SEED)
        i0 = torch.randint(2, size - 2, (1,), generator=g, device=CONFIG.DEVICE).item()
        j0 = torch.randint(2, size - 2, (1,), generator=g, device=CONFIG.DEVICE).item()
        
        # Génération d'obstacles selon l'étape
        obstacle_mask = self.generate_obstacles(size, (i0, j0))
        
        # Initialisation
        grid = torch.zeros((len(ALL_REALITY_LAYERS), size, size), device=CONFIG.DEVICE)  # 2 layers: temperature, obstacle sur grille 16x16
        grid[REALITY_LAYER.TEMPERATURE, i0, j0] = CONFIG.SOURCE_INTENSITY  # force la source dans la chaleur
        # et on set les obstacles
        grid[REALITY_LAYER.OBSTACLE, obstacle_mask] = CONFIG.OBSTACLE_FULL_BLOCK_VALUE
        # ou grid[REALITY_LAYER.OBSTACLE, :, :] = obstacle_mask.float() ?
        
        source_mask = torch.zeros_like(grid[REALITY_LAYER.TEMPERATURE], dtype=torch.bool)
        source_mask[i0, j0] = True
        
        # S'assurer que la source n'est pas dans un obstacle
        if obstacle_mask[i0, j0]:
            obstacle_mask[i0, j0] = False
        
        # Simulation temporelle
        reality_worlds = [RealityWorld(grid.clone())]
        for _ in range(n_steps):
            grid = self._play_diffusion_step(grid, source_mask, obstacle_mask)
            reality_worlds.append(RealityWorld(grid.clone()))
        
        sequence = SimulationTemporalSequence(reality_worlds, source_mask, obstacle_mask)
        return sequence
    
    
    # Un pas de diffusion de chaleur avec obstacles
    def _play_diffusion_step(self, grid, source_mask, obstacle_mask):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        
        # On applique la diffusion uniquement sur la couche température
        x = grid[REALITY_LAYER.TEMPERATURE].unsqueeze(0).unsqueeze(0)  # Que la couche de température, shape (1, 1, H, W)
        new_grid_heat = F.conv2d(x, self._kernel_avg_3x3, padding=1).squeeze()
        
        # On clone la grille complète pour créer le nouvel état
        new_grid = grid.clone()
        
        # On met à jour la couche température avec la diffusion
        new_grid[REALITY_LAYER.TEMPERATURE] = new_grid_heat
        
        # IMPORTANT : On force la source à conserver son intensité (contrainte physique stricte)
        # Sans cela, la source s'atténue progressivement avec la diffusion
        new_grid[REALITY_LAYER.TEMPERATURE][source_mask] = CONFIG.SOURCE_INTENSITY
        
        # Les obstacles restent à 0 dans la couche température (pas de chaleur dans les obstacles)
        new_grid[REALITY_LAYER.TEMPERATURE][obstacle_mask] = 0.0
        
        # Clipping pour limiter la diffusion infinie :
        # On remet à 0.0 les zones où la température est descendue en dessous de 1% de l'intensité maximale de la source
        # Cela évite d'avoir des valeurs infinitésimales qui s'étalent indéfiniment sur toute la grille
        # et permet de maintenir des frontières nettes pour la zone d'influence thermique
        threshold = CONFIG.SOURCE_INTENSITY * 0.01
        new_grid[REALITY_LAYER.TEMPERATURE][new_grid[REALITY_LAYER.TEMPERATURE] < threshold] = 0.0
        
        # La couche obstacles (REALITY_LAYER.OBSTACLE) reste inchangée (les obstacles ne bougent pas)
        
        return new_grid
