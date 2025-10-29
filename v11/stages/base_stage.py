import json
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from config import CONFIG
from reality_world import RealityWorld
from simulation_temporal_sequence import SimulationTemporalSequence
from torch.nn import functional as F
from torched import get_MSELoss, get_matrix_boolean

if TYPE_CHECKING:
    from typing import Tuple, List, Dict
    from nca_model import NCA


class REALITY_LAYER:
    TEMPERATURE = 0
    OBSTACLE = 1
    HEAT_SOURCES = 2


ALL_REALITY_LAYERS = [REALITY_LAYER.TEMPERATURE, REALITY_LAYER.OBSTACLE, REALITY_LAYER.HEAT_SOURCES]


class BaseStage(ABC):
    NAME = 'UNSET'
    DISPLAY_NAME = 'Étape non définie'
    
    COLOR = 'black'
    
    # Obstacles
    # et donner uen vraie taille random
    MIN_OBSTACLE_SIZE = 0
    MAX_OBSTACLE_SIZE = 0
    
    MIN_OBSTACLE_NB = 0
    MAX_OBSTACLE_NB = 0
    
    
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
        
        self._loss_fn = get_MSELoss()
    
    
    def get_name(self):
        return self.NAME
    
    
    def get_display_name(self):
        return self.DISPLAY_NAME
    
    
    def get_color(self):
        return self.COLOR
    
    
    def set_stage_nb(self, stage_nb: int):
        self._stage_nb = stage_nb
    
    
    def _set_metrics(self, epochs_trained: int, loss_history: list, stage_lrs: list):
        self._metrics_epochs_trained = epochs_trained
        self._metrics_loss_history = loss_history
        self._metrics_stage_lrs = stage_lrs
    
    
    def get_stage_nb(self):
        return self._stage_nb
    
    
    # Valide qu'un chemin de diffusion reste possible avec les obstacles.
    # Utilise un algorithme de flood-fill simplifié.
    def _validate_connectivity(self, obstacle_mask, source_pos):
        # type: (torch.Tensor, Tuple[int, int]) -> bool
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
        
        # Au moins 50% de la grille doit être accessible pour une bonne diffusion
        total_free_cells = (H * W) - obstacle_mask.sum().item()
        connectivity_ratio = accessible_cells / max(total_free_cells, 1)
        
        return connectivity_ratio >= 0.5
    
    
    # Le stage va définir son nombre et la taille des obstacles
    def generate_obstacles(self, size, source_pos):
        # type: (int, Tuple[int, int]) -> torch.Tensor
        
        obstacle_mask = get_matrix_boolean((size, size), fill_value=False)
        
        g = torch.Generator(device=CONFIG.DEVICE)
        g.manual_seed(CONFIG.SEED)
        
        n_obstacles = torch.randint(self.MIN_OBSTACLE_NB, self.MAX_OBSTACLE_NB + 1, (1,), generator=g, device=CONFIG.DEVICE).item()
        
        source_i, source_j = source_pos
        placed_obstacles = []
        
        for obstacle_idx in range(n_obstacles):
            obstacle_size = torch.randint(self.MIN_OBSTACLE_SIZE, self.MAX_OBSTACLE_SIZE + 1, (1,), generator=g, device=CONFIG.DEVICE).item()
            
            max_pos = size - obstacle_size
            if max_pos <= 1:
                continue
            
            for attempt in range(50):
                i = torch.randint(1, max_pos, (1,), generator=g, device=CONFIG.DEVICE).item()
                j = torch.randint(1, max_pos, (1,), generator=g, device=CONFIG.DEVICE).item()
                
                # Vérifications multiples pour étape 3
                valid_position = True
                
                # 1. Pas de chevauchement avec source
                if i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size:
                    valid_position = False
                
                # 2. Pas de chevauchement avec obstacles existants
                for obs_i, obs_j, obs_size in placed_obstacles:
                    if (i < obs_i + obs_size and i + obstacle_size > obs_i and
                            j < obs_j + obs_size and j + obstacle_size > obs_j):
                        valid_position = False
                        break
                
                if valid_position:
                    obstacle_mask[i:i + obstacle_size, j:j + obstacle_size] = True
                    placed_obstacles.append((i, j, obstacle_size))
                    break
        
        # Validation finale de connectivité
        if not self._validate_connectivity(obstacle_mask, source_pos):
            raise Exception("⚠️  Connectivité non garantie - génération d'un environnement plus simple")
        
        return obstacle_mask
    
    
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
    
    
    def get_metrics_lrs(self):
        return self._metrics_stage_lrs
    
    
    def _save_stage_checkpoint(self, model_state_dict, optimizer_state_dict):
        # type: (Dict, Dict) -> None
        
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
    
    
    # Récupère un échantillon pour l'étape spécifiée
    def get_sequences_for_training(self):
        # type: (BaseStage) -> SimulationTemporalSequence
        
        if not self._reality_temporal_sequences_for_training:
            raise Exception(f"Le cache de séquences n'a pas été généré pour l'étape {self.get_stage_nb()}.")
        
        # Récupère l'échantillon courant et avance l'index
        sequence = self._reality_temporal_sequences_for_training[self._reality_temporal_sequences_for_training_current_indices]
        
        self._reality_temporal_sequences_for_training_current_indices = (self._reality_temporal_sequences_for_training_current_indices + 1) % len(
                self._reality_temporal_sequences_for_training)
        
        return sequence
    
    
    def generate_reality_sequences_for_training(self):
        cache_size = CONFIG.NB_EPOCHS_BY_STAGE
        print(f"🎯 Génération de {cache_size} séquences pour l'étape {self.get_stage_nb()}...", end='', flush=True)
        
        for i in range(cache_size):
            simulation_temporal_sequence = self.generate_simulation_temporal_sequence(n_steps=CONFIG.NCA_STEPS, size=CONFIG.GRID_SIZE)
            self._reality_temporal_sequences_for_training.append(simulation_temporal_sequence)
        
        print(f"\r✅ Cache étape {self.get_stage_nb()} créé ({cache_size} séquences)")
    
    
    def generate_reality_sequences_for_evaluation(self):
        print(f"Génération de {CONFIG.NB_EPOCHS_FOR_EVALUATION} séquences pour Evaluation de {self.get_stage_nb()}...", end='', flush=True)
        
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
        # 3 layers: temperature, obstacle sur grille 16x16, source 16x16
        grid = torch.zeros((len(ALL_REALITY_LAYERS), size, size), device=CONFIG.DEVICE)
        
        grid[REALITY_LAYER.TEMPERATURE, i0, j0] = CONFIG.SOURCE_INTENSITY  # force la source dans la chaleur
        # et on set les obstacles
        grid[REALITY_LAYER.OBSTACLE, obstacle_mask] = CONFIG.OBSTACLE_FULL_BLOCK_VALUE
        
        # TODO: Gestion d'une seule source pour l'instant
        source_mask = torch.zeros_like(grid[REALITY_LAYER.TEMPERATURE], dtype=torch.bool)
        source_mask[i0, j0] = True
        
        grid[REALITY_LAYER.HEAT_SOURCES, source_mask] = CONFIG.SOURCE_INTENSITY  # couche des sources
        
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
    # NOTE: on ne diffuse que les températures, les obstacles restent fixes, les sources aussi
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
        # Pareil pour les sources (REALITY_LAYER.HEAT_SOURCES)
        
        return new_grid
    
    
    def _train_step(self, model, sequence, optimizer):
        # type: (NCA, SimulationTemporalSequence, torch.optim.Optimizer) -> float
        
        """
        Un pas d'entraînement adapté à l'étape courante.

        Le modèle apprend à respecter les contraintes des obstacles via une pénalité forte
        dans la fonction de perte, plutôt que par forçage explicite après prédiction.

        Args:
            sequence: Sequence d'entraînement
        Returns:
            Perte pour ce pas
        """
        
        optimizer.zero_grad()
        
        reality_worlds = sequence.get_reality_worlds()
        source_mask = sequence.get_source_mask()
        obstacle_mask = sequence.get_obstacle_mask()
        
        # Initialisation
        grid_pred = torch.zeros_like(reality_worlds[0].get_as_tensor())
        
        # TODO: bruiter la température initiale?
        # g = torch.Generator(device=DEVICE)
        # g.manual_seed(CONFIG.SEED)
        # grid_pred[REALITY_LAYER.TEMPERATURE] = torch.rand_like(
        #        grid_pred[REALITY_LAYER.TEMPERATURE], device=DEVICE
        # ) * 0.05  # Petite valeur initiale aléatoire pour la température
        
        # grid_pred[REALITY_LAYER.TEMPERATURE][source_mask] = CONFIG.SOURCE_INTENSITY  # Set Les sources, on a une chaleur dès le départ  # TODO: need to init or not?
        grid_pred[REALITY_LAYER.OBSTACLE][obstacle_mask] = CONFIG.OBSTACLE_FULL_BLOCK_VALUE  # Set les obstacles
        grid_pred[REALITY_LAYER.HEAT_SOURCES][source_mask] = CONFIG.SOURCE_INTENSITY  # Set les sources
        
        total_loss = torch.tensor(0.0, device=CONFIG.DEVICE)
        
        # Déroulement temporel
        for t_step in range(CONFIG.NCA_STEPS):
            target = reality_worlds[t_step + 1].get_as_tensor()
            grid_pred = model.run_step(grid_pred)  # , source_mask)
            
            # Perte standard sur la prédiction globale de la température
            step_loss = self._loss_fn(grid_pred, target)
            
            total_loss = total_loss + step_loss
        
        avg_loss = total_loss / CONFIG.NCA_STEPS
        
        # combined_loss = avg_loss
        combined_loss_float = avg_loss.item()
        
        # Backpropagation avec clipping
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if combined_loss_float == float('nan'):
            raise ValueError("NaN loss encountered during training step.")
        
        # On retourne la perte combinée pour le monitoring
        return combined_loss_float
    
    
    def _train_full_sequence(self, model, optimizer):
        # type: (NCA, torch.optim.Optimizer) -> List[float]
        
        epoch_losses = []  # type: List[float]
        
        # Entraînement par batch
        for _ in range(CONFIG.BATCH_SIZE):
            sequence = self.get_sequences_for_training()
            loss = self._train_step(model, sequence, optimizer)  # type: float
            epoch_losses.append(loss)
        
        return epoch_losses
    
    
    # Ajuste le learning rate selon l'étape et la progression.
    def _adjust_learning_rate(self, epoch_in_stage, optimizer):
        # type: (int, torch.optim.Optimizer) -> None
        
        from stage_manager import STAGE_MANAGER
        
        base_lr = CONFIG.LEARNING_RATE
        
        # Réduction progressive par étape, from 1.0 -> 0.6
        stage_lr = base_lr * (1.0 - ((self.get_stage_nb() - 1) / (len(STAGE_MANAGER.get_stages()) - 1)) * 0.4)
        
        # Décroissance cosine au sein de l'étape
        cos_factor = 0.5 * (1 + np.cos(np.pi * epoch_in_stage / CONFIG.NB_EPOCHS_BY_STAGE))
        final_lr = stage_lr * (0.1 + 0.9 * cos_factor)  # Ne descends pas sous 10% du LR de base
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = final_lr
    
    
    # Entraînement complet
    def train(self, model, optimizer):
        # type: (NCA, torch.optim.Optimizer) -> None
        
        max_epochs = CONFIG.NB_EPOCHS_BY_STAGE
        stage_nb = self.get_stage_nb()
        print(f"\n🎯 === ÉTAPE {stage_nb} - DÉBUT ===")
        print(f"📋 {self.get_display_name()}")
        print(f"⏱️  Maximum {max_epochs} époques")
        
        # Initialisation du cache pour cette étape
        self.generate_reality_sequences_for_training()
        
        # Métriques de l'étape
        stage_losses = []
        stage_lrs = []
        epoch_in_stage = 0
        
        # Boucle d'entraînement de l'étape
        for epoch_in_stage in range(max_epochs):
            
            # Ajustement du learning rate si curriculum activé
            self._adjust_learning_rate(epoch_in_stage, optimizer)
            
            # Entraînement par batch
            epoch_losses = self._train_full_sequence(model, optimizer)
            
            # Statistiques de l'époque
            avg_epoch_loss = np.mean(epoch_losses)
            stage_losses.append(avg_epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']
            stage_lrs.append(current_lr)
            
            # Affichage périodique
            if epoch_in_stage % 50 == 0 or epoch_in_stage == max_epochs - 1:
                print(f"  Époque {epoch_in_stage:3d}/{max_epochs - 1} | Loss: {avg_epoch_loss:.6f} | LR: {current_lr:.2e}")
        
        self._set_metrics(epoch_in_stage + 1, stage_losses, stage_lrs)
        
        print(f"📊 Époques entraînées: {epoch_in_stage + 1}/{max_epochs}")
        
        # Sauvegarde du checkpoint d'étape
        self._save_stage_checkpoint(model.state_dict(), optimizer.state_dict())
