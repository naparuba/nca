import json
import os
# HACK for imports
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# Hack for imports
sys.path.append(os.path.dirname(__file__))

from config import CONFIG
from torching import DEVICE

# =============================================================================
# Configuration et initialisation modulaire
# =============================================================================


SEED = CONFIG.SEED
VISUALIZATION_SEED = CONFIG.VISUALIZATION_SEED
STAGNATION_THRESHOLD = CONFIG.STAGNATION_THRESHOLD
STAGNATION_PATIENCE = CONFIG.STAGNATION_PATIENCE

# Initialisation
torch.manual_seed(SEED)
np.random.seed(SEED)

# Création du répertoire de sortie avec seed
CONFIG.OUTPUT_DIR = f"outputs"
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Seed: {SEED}")
print(f"Répertoire de sortie: {CONFIG.OUTPUT_DIR}")

from stage_manager import STAGE_MANAGER


# =============================================================================
# Gestionnaire d'obstacles progressifs
# =============================================================================

class ProgressiveObstacleManager:
    """
    Gestionnaire intelligent des obstacles selon l'étape d'apprentissage.
    Génère des environnements appropriés pour chaque phase du curriculum.
    """
    
    
    def __init__(self):
        pass
    
    
    def generate_stage_environment(self, stage_nb: int, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        """
        Génère un environnement d'obstacles adapté à l'étape courante.
        
        Args:
            stage_nb: Numéro d'étape (1, 2, ou 3)
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            
        Returns:
            Masque des obstacles [H, W]
        """
        
        stage = STAGE_MANAGER.get_stage(stage_nb)
        # print(f"🎯 Génération d'environnement pour l'étape {stage_nb} ({stage.get_name()})...")
        
        return stage.generate_environment(size, source_pos)
        # if stage_nb == 1:
        #     return self._generate_stage_1_environment(size)
        # elif stage_nb == 2:
        #     return self._generate_stage_2_environment(size, source_pos)
        # elif stage_nb == 3:
        #     return self._generate_stage_3_environment(size, source_pos)
    
    
    # def _generate_stage_1_environment(self, size: int) -> torch.Tensor:
    #     """Étape 1: Aucun obstacle - grille vide pour apprentissage de base."""
    #     return torch.zeros((size, size), dtype=torch.bool, device=DEVICE)
    
    # def _generate_stage_2_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
    #     """Étape 2: Un seul obstacle pour apprentissage du contournement."""
    #     obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=DEVICE)
    #
    #     g = torch.Generator(device=DEVICE)
    #     g.manual_seed(SEED)
    #
    #     # Un seul obstacle de taille aléatoire
    #     obstacle_size = torch.randint(CONFIG.MIN_OBSTACLE_SIZE, CONFIG.MAX_OBSTACLE_SIZE + 1,
    #                                   (1,), generator=g, device=DEVICE).item()
    #
    #     # Placement en évitant la source et les bords
    #     max_pos = size - obstacle_size
    #     if max_pos <= 1:
    #         return obstacle_mask  # Grille trop petite
    #
    #     source_i, source_j = source_pos
    #
    #     for attempt in range(100):  # Plus de tentatives pour étape 2
    #         i = torch.randint(1, max_pos, (1,), generator=g, device=DEVICE).item()
    #         j = torch.randint(1, max_pos, (1,), generator=g, device=DEVICE).item()
    #
    #         # Vérifier non-chevauchement avec source
    #         if not (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
    #             obstacle_mask[i:i + obstacle_size, j:j + obstacle_size] = True
    #             break
    #
    #     return obstacle_mask
    
    # def _generate_stage_3_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
    #     """Étape 3: Obstacles multiples pour gestion de la complexité."""
    #     obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=DEVICE)
    #
    #     g = torch.Generator(device=DEVICE)
    #     g.manual_seed(SEED)
    #
    #     config = self.stage_configs[3]
    #     n_obstacles = torch.randint(config['min_obstacles'], config['max_obstacles'] + 1,
    #                                 (1,), generator=g, device=DEVICE).item()
    #
    #     source_i, source_j = source_pos
    #     placed_obstacles = []
    #
    #     for obstacle_idx in range(n_obstacles):
    #         obstacle_size = torch.randint(CONFIG.MIN_OBSTACLE_SIZE, CONFIG.MAX_OBSTACLE_SIZE + 1,
    #                                       (1,), generator=g, device=DEVICE).item()
    #
    #         max_pos = size - obstacle_size
    #         if max_pos <= 1:
    #             continue
    #
    #         for attempt in range(50):
    #             i = torch.randint(1, max_pos, (1,), generator=g, device=DEVICE).item()
    #             j = torch.randint(1, max_pos, (1,), generator=g, device=DEVICE).item()
    #
    #             # Vérifications multiples pour étape 3
    #             valid_position = True
    #
    #             # 1. Pas de chevauchement avec source
    #             if i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size:
    #                 valid_position = False
    #
    #             # 2. Pas de chevauchement avec obstacles existants
    #             for obs_i, obs_j, obs_size in placed_obstacles:
    #                 if (i < obs_i + obs_size and i + obstacle_size > obs_i and
    #                         j < obs_j + obs_size and j + obstacle_size > obs_j):
    #                     valid_position = False
    #                     break
    #
    #             if valid_position:
    #                 obstacle_mask[i:i + obstacle_size, j:j + obstacle_size] = True
    #                 placed_obstacles.append((i, j, obstacle_size))
    #                 break
    #
    #     # Validation finale de connectivité
    #     if not self._validate_connectivity(obstacle_mask, source_pos):
    #         print("⚠️  Connectivité non garantie - génération d'un environnement plus simple")
    #         return self._generate_stage_2_environment(size, source_pos)
    #
    #     return obstacle_mask
    
    # def _validate_connectivity(self, obstacle_mask: torch.Tensor, source_pos: Tuple[int, int]) -> bool:
    #     """
    #     Valide qu'un chemin de diffusion reste possible avec les obstacles.
    #     Utilise un algorithme de flood-fill simplifié.
    #     """
    #     H, W = obstacle_mask.shape
    #     source_i, source_j = source_pos
    #
    #     # Matrice de visite
    #     visited = torch.zeros_like(obstacle_mask, dtype=torch.bool)
    #     visited[obstacle_mask] = True  # Les obstacles sont "déjà visités"
    #
    #     # Flood-fill depuis la source
    #     stack = [(source_i, source_j)]
    #     visited[source_i, source_j] = True
    #     accessible_cells = 1
    #
    #     while stack:
    #         i, j = stack.pop()
    #
    #         # Parcours des 4 voisins
    #         for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    #             ni, nj = i + di, j + dj
    #
    #             if (0 <= ni < H and 0 <= nj < W and
    #                     not visited[ni, nj] and not obstacle_mask[ni, nj]):
    #                 visited[ni, nj] = True
    #                 stack.append((ni, nj))
    #                 accessible_cells += 1
    #
    #     # Au moins 50% de la grille doit être accessible pour une bonne diffusion
    #     total_free_cells = (H * W) - obstacle_mask.sum().item()
    #     connectivity_ratio = accessible_cells / max(total_free_cells, 1)
    #
    #     return connectivity_ratio >= 0.5
    
    def get_difficulty_metrics(self, stage_nb: int, obstacle_mask: torch.Tensor) -> Dict[str, float]:
        """
        Calcule des métriques de difficulté pour l'environnement généré.
        
        Returns:
            Dictionnaire avec les métriques de complexité
        """
        H, W = obstacle_mask.shape
        total_cells = H * W
        obstacle_cells = obstacle_mask.sum().item()
        
        metrics = {
            'stage_nb':         stage_nb,
            'obstacle_ratio':   obstacle_cells / total_cells,
            'free_cells':       total_cells - obstacle_cells,
            'complexity_score': stage_nb * (obstacle_cells / total_cells)
        }
        
        return metrics


# =============================================================================
# Simulateur de diffusion
# =============================================================================

class DiffusionSimulator:
    """
    Simulateur de diffusion de chaleur adapté pour l'apprentissage modulaire.
    Utilise le gestionnaire d'obstacles progressifs.
    """
    
    
    def __init__(self):
        self.kernel = torch.ones((1, 1, 3, 3), device=DEVICE) / 9.0
        self.obstacle_manager = ProgressiveObstacleManager()
    
    
    def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor) -> torch.Tensor:
        """Un pas de diffusion de chaleur avec obstacles."""
        x = grid.unsqueeze(0).unsqueeze(0)
        new_grid = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)
        
        # Contraintes
        new_grid[obstacle_mask] = 0.0
        new_grid[source_mask] = grid[source_mask]
        
        return new_grid
    
    
    def generate_stage_sequence(self, stage_nb: int, n_steps: int, size: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Génère une séquence adaptée à l'étape d'apprentissage courante.
        
        Args:
            stage_nb: Étape d'apprentissage (1, 2, ou 3)
            n_steps: Nombre d'étapes de simulation
            size: Taille de la grille
            
        Returns:
            (séquence, masque_source, masque_obstacles)
        """
        # Position aléatoire de la source
        g = torch.Generator(device=DEVICE)
        g.manual_seed(SEED)
        i0 = torch.randint(2, size - 2, (1,), generator=g, device=DEVICE).item()
        j0 = torch.randint(2, size - 2, (1,), generator=g, device=DEVICE).item()
        
        # Génération d'obstacles selon l'étape
        obstacle_mask = self.obstacle_manager.generate_stage_environment(stage_nb, size, (i0, j0))
        
        # Initialisation
        grid = torch.zeros((size, size), device=DEVICE)
        grid[i0, j0] = CONFIG.SOURCE_INTENSITY
        
        source_mask = torch.zeros_like(grid, dtype=torch.bool)
        source_mask[i0, j0] = True
        
        # S'assurer que la source n'est pas dans un obstacle
        if obstacle_mask[i0, j0]:
            obstacle_mask[i0, j0] = False
        
        # Simulation temporelle
        sequence = [grid.clone()]
        for _ in range(n_steps):
            grid = self.step(grid, source_mask, obstacle_mask)
            sequence.append(grid.clone())
        
        return sequence, source_mask, obstacle_mask


# Instance globale du simulateur modulaire
simulator = DiffusionSimulator()


# =============================================================================
# Modèle NCA
# =============================================================================

class ImprovedNCA(nn.Module):
    """
    Neural Cellular Automaton optimisé pour l'apprentissage modulaire.
    Architecture identique à v6 mais avec support étendu pour le curriculum.
    """
    
    
    def __init__(self, input_size: int):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = CONFIG.HIDDEN_SIZE
        self.n_layers = CONFIG.N_LAYERS
        
        # Architecture profonde avec normalisation
        layers = []
        current_size = input_size
        
        for i in range(self.n_layers):
            layers.append(nn.Linear(current_size, self.hidden_size))
            layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_size = self.hidden_size
        
        # Couche de sortie stabilisée
        layers.append(nn.Linear(self.hidden_size, 1))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.delta_scale = 0.1
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec scaling des deltas."""
        delta = self.network(x)
        return delta * self.delta_scale


# =============================================================================
# Updaters NCA
# =============================================================================

class OptimizedNCAUpdater:
    """
    Updater optimisé avec extraction vectorisée des patches.
    """
    
    
    def __init__(self, model: ImprovedNCA):
        self.model = model
    
    
    def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor) -> torch.Tensor:
        """Application optimisée du NCA."""
        H, W = grid.shape
        
        # Extraction vectorisée des patches 3x3
        grid_padded = F.pad(grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        patches = F.unfold(grid_padded, kernel_size=3, stride=1)
        patches = patches.squeeze(0).transpose(0, 1)  # [H*W, 9]
        
        # Features additionnelles
        source_flat = source_mask.flatten().float().unsqueeze(1)  # [H*W, 1]
        obstacle_flat = obstacle_mask.flatten().float().unsqueeze(1)  # [H*W, 1]
        full_patches = torch.cat([patches, source_flat, obstacle_flat], dim=1)  # [H*W, 11]
        
        # Application seulement sur positions valides
        valid_mask = ~obstacle_mask.flatten()
        
        if valid_mask.any():
            valid_patches = full_patches[valid_mask]
            deltas = self.model(valid_patches)
            
            new_grid = grid.clone().flatten()
            new_grid[valid_mask] += deltas.squeeze()
            new_grid = torch.clamp(new_grid, 0.0, 1.0).reshape(H, W)
        else:
            new_grid = grid.clone()
        
        # Contraintes finales
        new_grid[obstacle_mask] = 0.0
        new_grid[source_mask] = grid[source_mask]
        
        return new_grid




# =============================================================================
# Planificateur de curriculum (NOUVEAU)
# =============================================================================


# =============================================================================
# Cache de séquences optimisé par étape
# =============================================================================

class OptimizedSequenceCache:
    """
    Cache spécialisé par étape pour l'entraînement modulaire.
    Maintient des caches séparés pour chaque étape d'apprentissage.
    """
    
    
    def __init__(self, simulator: DiffusionSimulator):
        self.simulator = simulator
        self.stage_caches = {}  # Cache par étape
        self.cache_sizes = {1: 150, 2: 200, 3: 250}  # Plus de variété pour étapes complexes
        self.current_indices = {}
    
    
    def initialize_stage_cache(self, stage_nb: int):
        """Initialise le cache pour une étape spécifique."""
        if stage_nb in self.stage_caches:
            return  # Déjà initialisé
        
        cache_size = self.cache_sizes.get(stage_nb, 200)
        print(f"🎯 Génération de {cache_size} séquences pour l'étape {stage_nb}...")
        
        sequences = []
        for i in range(cache_size):
            if i % 50 == 0:
                print(f"   Étape {stage_nb}: {i}/{cache_size}")
            
            target_seq, source_mask, obstacle_mask = self.simulator.generate_stage_sequence(
                    stage_nb=stage_nb,
                    n_steps=CONFIG.NCA_STEPS,
                    size=CONFIG.GRID_SIZE
            )
            
            sequences.append({
                'target_seq':    target_seq,
                'source_mask':   source_mask,
                'obstacle_mask': obstacle_mask,
                'stage_nb':      stage_nb
            })
        
        self.stage_caches[stage_nb] = sequences
        self.current_indices[stage_nb] = 0
        print(f"✅ Cache étape {stage_nb} créé ({cache_size} séquences)")
    
    
    def get_stage_batch(self, stage_nb: int, batch_size: int):
        """Récupère un batch pour l'étape spécifiée."""
        if stage_nb not in self.stage_caches:
            self.initialize_stage_cache(stage_nb)
        
        cache = self.stage_caches[stage_nb]
        batch = []
        
        for _ in range(batch_size):
            batch.append(cache[self.current_indices[stage_nb]])
            self.current_indices[stage_nb] = (self.current_indices[stage_nb] + 1) % len(cache)
        
        return batch
    
    
    def shuffle_stage_cache(self, stage_nb: int):
        """Mélange le cache d'une étape spécifique."""
        if stage_nb in self.stage_caches:
            import random
            random.shuffle(self.stage_caches[stage_nb])
    
    
    def clear_stage_cache(self, stage_nb: int):
        """Libère la mémoire du cache d'une étape."""
        if stage_nb in self.stage_caches:
            del self.stage_caches[stage_nb]
            del self.current_indices[stage_nb]
            print(f"🗑️  Cache étape {stage_nb} libéré")


# =============================================================================
# Entraîneur modulaire principal
# =============================================================================

class ModularTrainer:
    """
    Système d'entraînement modulaire progressif.
    Gère l'apprentissage par étapes avec transitions automatiques.
    """
    
    
    def __init__(self, model: ImprovedNCA):
        self.model = model
        
        # Choix de l'updater optimisé
        print("🚀 Utilisation de l'updater optimisé vectorisé")
        self.updater = OptimizedNCAUpdater(model)
        
        # Optimiseur et planificateur
        self.optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()
        
        # Curriculum et métriques
        from scheduler import CurriculumScheduler
        self.curriculum = CurriculumScheduler()
        
        # Cache optimisé par étape
        self.sequence_cache = OptimizedSequenceCache(simulator)
        self.use_cache = True
        
        # État d'entraînement
        self.current_stage = 1
        self.stage_histories = {stage_nb: {'losses': [], 'epochs': [], 'lr': []} for stage_nb in [1, 2, 3]}
        self.global_history = {'losses': [], 'stages': [], 'epochs': []}
        self.stage_start_epochs = {}
        self.total_epochs_trained = 0
    
    
    def train_step(self, target_sequence: List[torch.Tensor], source_mask: torch.Tensor,
                   obstacle_mask: torch.Tensor, stage_nb: int) -> float:
        """
        Un pas d'entraînement adapté à l'étape courante.

        Args:
            target_sequence: Séquence cible
            source_mask: Masque des sources
            obstacle_mask: Masque des obstacles
            stage_nb: Étape courante d'entraînement

        Returns:
            Perte pour ce pas
        """
        self.optimizer.zero_grad()
        
        # Initialisation
        grid_pred = torch.zeros_like(target_sequence[0])
        grid_pred[source_mask] = CONFIG.SOURCE_INTENSITY
        
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        # Déroulement temporel
        for t_step in range(CONFIG.NCA_STEPS):
            target = target_sequence[t_step + 1]
            grid_pred = self.updater.step(grid_pred, source_mask, obstacle_mask)
            
            # Perte pondérée selon l'étape
            step_loss = self.loss_fn(grid_pred, target)
            
            total_loss = total_loss + step_loss
        
        avg_loss = total_loss / CONFIG.NCA_STEPS
        
        # Backpropagation avec clipping
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return avg_loss.item()
    
    
    def train_stage(self, stage_nb: int, max_epochs: int) -> Dict[str, Any]:
        """
        Entraînement complet d'une étape spécifique.

        Args:
            stage_nb: Numéro d'étape (1, 2, ou 3)
            max_epochs: Nombre maximum d'époques pour cette étape

        Returns:
            Dictionnaire avec les métriques de l'étape
        """
        print(f"\n🎯 === ÉTAPE {stage_nb} - DÉBUT ===")
        stage_name = {1: "Sans obstacles", 2: "Un obstacle", 3: "Obstacles multiples"}[stage_nb]
        print(f"📋 {stage_name}")
        print(f"⏱️  Maximum {max_epochs} époques")
        
        self.current_stage = stage_nb
        self.stage_start_epochs[stage_nb] = self.total_epochs_trained
        
        # Initialisation du cache pour cette étape
        if self.use_cache:
            self.sequence_cache.initialize_stage_cache(stage_nb)
        
        # Métriques de l'étape
        stage_losses = []
        epoch_in_stage = 0
        early_stop = False
        
        # Boucle d'entraînement de l'étape
        for epoch_in_stage in range(max_epochs):
            epoch_losses = []
            
            # Ajustement du learning rate si curriculum activé
            if self.curriculum:
                self.curriculum.adjust_learning_rate(self.optimizer, stage_nb, epoch_in_stage)
            
            # Mélange périodique du cache
            if self.use_cache and epoch_in_stage % 20 == 0:
                self.sequence_cache.shuffle_stage_cache(stage_nb)
            
            # Entraînement par batch
            for _ in range(CONFIG.BATCH_SIZE):
                batch_sequences = self.sequence_cache.get_stage_batch(stage_nb, 1)
                seq_data = batch_sequences[0]
                target_seq = seq_data['target_seq']
                source_mask = seq_data['source_mask']
                obstacle_mask = seq_data['obstacle_mask']
                
                loss = self.train_step(target_seq, source_mask, obstacle_mask, stage_nb)
                epoch_losses.append(loss)
            
            # Statistiques de l'époque
            avg_epoch_loss = np.mean(epoch_losses)
            stage_losses.append(avg_epoch_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Historiques
            self.stage_histories[stage_nb]['losses'].append(avg_epoch_loss)
            self.stage_histories[stage_nb]['epochs'].append(epoch_in_stage)
            self.stage_histories[stage_nb]['lr'].append(current_lr)
            
            self.global_history['losses'].append(avg_epoch_loss)
            self.global_history['stages'].append(stage_nb)
            self.global_history['epochs'].append(self.total_epochs_trained)
            
            self.total_epochs_trained += 1
            
            # Affichage périodique
            if epoch_in_stage % 10 == 0 or epoch_in_stage == max_epochs - 1:
                print(f"  Époque {epoch_in_stage:3d}/{max_epochs - 1} | "
                      f"Loss: {avg_epoch_loss:.6f} | "
                      f"LR: {current_lr:.2e}")
            
            # Vérification de l'avancement automatique (curriculum)
            if self.curriculum.should_advance_stage(stage_nb, stage_losses):
                print(f"🎯 Convergence atteinte à l'époque {epoch_in_stage}")
                print(f"   Loss: {avg_epoch_loss:.6f}")
                early_stop = True
                break
        
        # Résumé de l'étape
        final_loss = stage_losses[-1] if stage_losses else float('inf')
        # convergence_met = final_loss < CONFIG.CONVERGENCE_THRESHOLDS.get(stage_nb, 0.05)
        
        stage_metrics = {
            'stage_nb':       stage_nb,
            'epochs_trained': epoch_in_stage + 1,
            'final_loss':     final_loss,
            'early_stopped':  early_stop,
            'loss_history':   stage_losses
        }
        
        print(f"✅ === ÉTAPE {stage_nb} - TERMINÉE ===")
        print(f"📊 Époques entraînées: {epoch_in_stage + 1}/{max_epochs}")
        print(f"📉 Perte finale: {final_loss:.6f}")
        print(f"⚡ Arrêt précoce: {'✅ OUI' if early_stop else '❌ NON'}")
        
        # Sauvegarde du checkpoint d'étape
        stage = STAGE_MANAGER.get_stage(stage_nb)
        stage.save_stage_checkpoint(stage_metrics, self.model.state_dict(), self.optimizer.state_dict())
        
        # Libération du cache de l'étape précédente pour économiser la mémoire
        if self.use_cache and stage_nb > 1:
            self.sequence_cache.clear_stage_cache(stage_nb - 1)
        
        return stage_metrics
    
    
    def train_full_curriculum(self) -> Dict[str, Any]:
        """
        Entraînement complet du curriculum en 3 étapes.

        Returns:
            Métriques complètes de l'entraînement modulaire
        """
        print(f"\n🚀 === DÉBUT ENTRAÎNEMENT MODULAIRE ===")
        print(f"🎯 Seed: {SEED}")
        print(f"📊 Époques totales prévues: {CONFIG.TOTAL_EPOCHS}")
        print(f"🔄 Étapes: {CONFIG.STAGE_1_EPOCHS} + {CONFIG.STAGE_2_EPOCHS} + {CONFIG.STAGE_3_EPOCHS}")
        
        start_time = time.time()
        self.model.train()
        
        # Entraînement séquentiel des 3 étapes
        all_stage_metrics = {}
        
        # ÉTAPE 1: Sans obstacles
        stage_1_metrics = self.train_stage(1, CONFIG.STAGE_1_EPOCHS)
        all_stage_metrics[1] = stage_1_metrics
        
        # ÉTAPE 2: Un obstacle
        stage_2_metrics = self.train_stage(2, CONFIG.STAGE_2_EPOCHS)
        all_stage_metrics[2] = stage_2_metrics
        
        # ÉTAPE 3: Obstacles multiples
        stage_3_metrics = self.train_stage(3, CONFIG.STAGE_3_EPOCHS)
        all_stage_metrics[3] = stage_3_metrics
        
        # Métriques globales
        total_time = time.time() - start_time
        total_epochs_actual = sum(metrics['epochs_trained'] for metrics in all_stage_metrics.values())
        
        global_metrics = {
            'total_epochs_planned': CONFIG.TOTAL_EPOCHS,
            'total_epochs_actual':  total_epochs_actual,
            'total_time_seconds':   total_time,
            'total_time_formatted': f"{total_time / 60:.1f} min",
            'stage_metrics':        all_stage_metrics,
            'final_loss':           stage_3_metrics['final_loss'],
            'global_history':       self.global_history,
            'stage_histories':      self.stage_histories,
            'stage_start_epochs':   self.stage_start_epochs  # AJOUT de la clé manquante
        }
        
        print(f"\n🎉 === ENTRAÎNEMENT MODULAIRE TERMINÉ ===")
        print(f"⏱️  Temps total: {total_time / 60:.1f} minutes")
        print(f"📊 Époques totales: {total_epochs_actual}/{CONFIG.TOTAL_EPOCHS}")
        print(f"📉 Perte finale: {global_metrics['final_loss']:.6f}")
        
        # Sauvegarde du modèle final et des métriques
        self.save_final_model(global_metrics)
        
        return global_metrics
    
    
    def save_final_model(self, global_metrics: Dict[str, Any]):
        """Sauvegarde le modèle final et toutes les métriques."""
        # Modèle final
        final_model_path = Path(CONFIG.OUTPUT_DIR) / "final_model.pth"
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_metrics':       global_metrics,
            'config':               CONFIG.__dict__
        }, final_model_path)
        
        # Métriques complètes
        full_metrics_path = Path(CONFIG.OUTPUT_DIR) / "complete_metrics.json"
        with open(full_metrics_path, 'w') as f:
            json.dump(global_metrics, f, indent=2, default=str)
        
        print(f"💾 Modèle final et métriques sauvegardés: {CONFIG.OUTPUT_DIR}")



def main():
    """
    Fonction principale pour l'entraînement modulaire progressif.
    Orchestre tout le processus d'apprentissage par curriculum.
    """
    print(f"\n" + "=" * 80)
    print(f"🚀 NEURAL CELLULAR AUTOMATON - APPRENTISSAGE MODULAIRE v7__")
    print(f"=" * 80)
    
    try:
        # Initialisation du modèle
        print("\n🔧 Initialisation du modèle...")
        model = ImprovedNCA(
                input_size=11,  # 9 (patch 3x3) + 1 (source) + 1 (obstacle)
        ).to(DEVICE)
        
        print(f"📊 Nombre de paramètres dans le modèle: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialisation de l'entraîneur modulaire
        print("🎯 Initialisation de l'entraîneur modulaire...")
        trainer = ModularTrainer(model)
        
        # Lancement de l'entraînement complet
        print("🚀 Lancement de l'entraînement modulaire...")
        global_metrics = trainer.train_full_curriculum()
        
        # Génération des visualisations progressives
        print("\n🎨 Génération des visualisations...")
        from visualizer import ProgressiveVisualizer
        visualizer = ProgressiveVisualizer()
        
        # Visualisation par étape avec le modèle final
        for stage_nb in [1, 2, 3]:
            visualizer.visualize_stage_results(model, stage_nb)
        
        # Résumé visuel complet du curriculum
        visualizer.create_curriculum_summary(global_metrics)
        
        # Rapport final
        print(f"\n" + "=" * 80)
        print(f"🎉 ENTRAÎNEMENT MODULAIRE TERMINÉ AVEC SUCCÈS!")
        print(f"=" * 80)
        print(f"📁 Résultats sauvegardés dans: {CONFIG.OUTPUT_DIR}")
        print(f"⏱️  Temps total: {global_metrics['total_time_formatted']}")
        print(f"📊 Époques: {global_metrics['total_epochs_actual']}/{global_metrics['total_epochs_planned']}")
        print(f"📉 Perte finale: {global_metrics['final_loss']:.6f}")
        
        # Détail par étape
        print(f"\n📋 DÉTAIL PAR ÉTAPE:")
        for stage in STAGE_MANAGER.get_stages():
            stage_data = global_metrics['stage_metrics'][stage.get_stage_nb()]
            print(f"   Étape {stage_nb} ({stage.get_display_name()}): {stage_data['final_loss']:.6f}")
        
        print(f"\n🎨 Fichiers de visualisation générés:")
        print(f"   • Animations par étape: stage_X/")
        print(f"   • Progression curriculum: curriculum_progression.png")
        print(f"   • Comparaison étapes: stage_comparison.png")
        print(f"   • Résumé performance: performance_summary.png")
        print(f"   • Métriques complètes: complete_metrics.json")
        
        return global_metrics
    
    except KeyboardInterrupt:
        print(f"\n⚠️  Entraînement interrompu par l'utilisateur")
        return None
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'entraînement:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Exécution du programme principal
    results = main()
    
    if results is not None:
        print(f"\n🎯 Programme terminé avec succès!")
        print(f"📊 Résultats disponibles dans la variable 'results'")
    else:
        print(f"\n❌ Programme terminé avec erreurs")
        exit(1)
