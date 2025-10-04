import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import os
import argparse
import json
import time
from pathlib import Path

# =============================================================================
# Configuration et initialisation modulaire
# =============================================================================

class ModularConfig:
    """
    Configuration étendue pour l'apprentissage modulaire progressif.
    Hérite des paramètres de base et ajoute la gestion des étapes.
    """
    def __init__(self, seed: int = 123):
        # Paramètres matériels de base
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SEED = seed

        # Paramètres de grille
        self.GRID_SIZE = 16
        self.SOURCE_INTENSITY = 1.0

        # Paramètres d'entraînement modulaire
        self.TOTAL_EPOCHS = 200  # Augmenté pour l'apprentissage modulaire
        self.NCA_STEPS = 20
        self.LEARNING_RATE = 1e-3
        self.BATCH_SIZE = 4

        # Configuration modulaire (NOUVEAU)
        self.MODULAR_TRAINING = True
        self.ENABLE_CURRICULUM = True
        self.ADAPTIVE_THRESHOLDS = True
        
        # Répartition des étapes (ratios des époques totales)
        self.STAGE_1_RATIO = 0.5  # 50% - Sans obstacles
        self.STAGE_2_RATIO = 0.3  # 30% - Un obstacle
        self.STAGE_3_RATIO = 0.2  # 20% - Obstacles multiples
        
        # Calcul automatique des époques par étape
        self.STAGE_1_EPOCHS = int(self.TOTAL_EPOCHS * self.STAGE_1_RATIO)
        self.STAGE_2_EPOCHS = int(self.TOTAL_EPOCHS * self.STAGE_2_RATIO)
        self.STAGE_3_EPOCHS = int(self.TOTAL_EPOCHS * self.STAGE_3_RATIO)

        # Seuils de convergence adaptatifs par étape
        self.CONVERGENCE_THRESHOLDS = {
            1: 0.01,  # Étape 1: convergence stricte
            2: 0.02,  # Étape 2: tolérance accrue
            3: 0.05   # Étape 3: tolérance maximale
        }

        # Paramètres de visualisation
        self.PREVIS_STEPS = 30
        self.POSTVIS_STEPS = 50
        self.SAVE_ANIMATIONS = True
        self.SAVE_STAGE_CHECKPOINTS = True
        self.OUTPUT_DIR = "__7__nca_outputs_modular_progressive_obstacles"

        # Paramètres du modèle
        self.HIDDEN_SIZE = 128
        self.N_LAYERS = 3

        # Paramètres d'obstacles par étape
        self.STAGE_OBSTACLE_CONFIG = {
            1: {'min_obstacles': 0, 'max_obstacles': 0},  # Pas d'obstacles
            2: {'min_obstacles': 1, 'max_obstacles': 1},  # Un seul obstacle
            3: {'min_obstacles': 2, 'max_obstacles': 4}   # 2-4 obstacles
        }
        
        self.MIN_OBSTACLE_SIZE = 2
        self.MAX_OBSTACLE_SIZE = 4

        # Optimisations (héritées de v6)
        self.USE_OPTIMIZATIONS = True
        self.USE_SEQUENCE_CACHE = True
        self.USE_VECTORIZED_PATCHES = True
        self.CACHE_SIZE = 200
        self.USE_MIXED_PRECISION = False

        # Nouveaux paramètres modulaires
        self.MAX_STAGE_RETRIES = 3  # Tentatives max par étape
        self.EARLY_STOPPING_PATIENCE = 20  # Patience pour arrêt précoce
        self.STAGE_TRANSITION_SMOOTHING = True  # Lissage des transitions

def parse_modular_arguments():
    """
    Parse les arguments étendus pour l'apprentissage modulaire.
    """
    parser = argparse.ArgumentParser(
        description='Neural Cellular Automaton - Apprentissage Modulaire Progressif',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments de base
    parser.add_argument('--seed', type=int, default=123,
                       help='Graine aléatoire pour la reproductibilité')
    parser.add_argument('--vis-seed', type=int, default=3333,
                       help='Graine pour les visualisations')
    parser.add_argument('--total-epochs', type=int, default=200,
                       help='Nombre total d\'époques d\'entraînement')
    parser.add_argument('--grid-size', type=int, default=16,
                       help='Taille de la grille')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Taille des batches')

    # Arguments modulaires (NOUVEAUX)
    parser.add_argument('--stage1-ratio', type=float, default=0.5,
                       help='Ratio d\'époques pour l\'étape 1 (sans obstacles)')
    parser.add_argument('--stage2-ratio', type=float, default=0.3,
                       help='Ratio d\'époques pour l\'étape 2 (un obstacle)')
    parser.add_argument('--stage3-ratio', type=float, default=0.2,
                       help='Ratio d\'époques pour l\'étape 3 (obstacles multiples)')
    
    parser.add_argument('--enable-curriculum', action='store_true', default=True,
                       help='Activer l\'apprentissage par curriculum')
    parser.add_argument('--adaptive-thresholds', action='store_true', default=True,
                       help='Utiliser des seuils adaptatifs pour l\'avancement')
    parser.add_argument('--max-obstacles', type=int, default=4,
                       help='Nombre maximum d\'obstacles en étape 3')
    parser.add_argument('--save-stage-checkpoints', action='store_true', default=True,
                       help='Sauvegarder les modèles à chaque étape')

    return parser.parse_args()

# Parse des arguments et configuration
args = parse_modular_arguments()
cfg = ModularConfig(seed=args.seed)

# Mise à jour depuis les arguments
cfg.TOTAL_EPOCHS = args.total_epochs
cfg.GRID_SIZE = args.grid_size
cfg.BATCH_SIZE = args.batch_size
cfg.STAGE_1_RATIO = args.stage1_ratio
cfg.STAGE_2_RATIO = args.stage2_ratio
cfg.STAGE_3_RATIO = args.stage3_ratio
cfg.ENABLE_CURRICULUM = args.enable_curriculum
cfg.ADAPTIVE_THRESHOLDS = args.adaptive_thresholds
cfg.SAVE_STAGE_CHECKPOINTS = args.save_stage_checkpoints

# Recalcul des époques par étape
cfg.STAGE_1_EPOCHS = int(cfg.TOTAL_EPOCHS * cfg.STAGE_1_RATIO)
cfg.STAGE_2_EPOCHS = int(cfg.TOTAL_EPOCHS * cfg.STAGE_2_RATIO)
cfg.STAGE_3_EPOCHS = int(cfg.TOTAL_EPOCHS * cfg.STAGE_3_RATIO)

# Configuration max obstacles étape 3
cfg.STAGE_OBSTACLE_CONFIG[3]['max_obstacles'] = args.max_obstacles

# Gestion matplotlib (héritée de v6)
def setup_matplotlib():
    """Configure matplotlib pour l'affichage interactif ou la sauvegarde."""
    try:
        matplotlib.use('Qt5Agg')
        plt.ion()
        fig, ax = plt.subplots()
        plt.close(fig)
        print("✅ Mode interactif activé")
        return True
    except:
        try:
            matplotlib.use('TkAgg')
            plt.ion()
            fig, ax = plt.subplots()
            plt.close(fig)
            print("✅ Mode interactif activé (TkAgg)")
            return True
        except:
            print("⚠️  Mode non-interactif détecté - les animations seront sauvegardées")
            matplotlib.use('Agg')
            return False

# Initialisation
interactive_mode = setup_matplotlib()
if os.name == 'nt':
    interactive_mode = False
    
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

# Création du répertoire de sortie avec seed
cfg.OUTPUT_DIR = f"__7__nca_outputs_modular_progressive_obstacles_seed_{cfg.SEED}"
if cfg.SAVE_ANIMATIONS:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(f"🎯 Configuration Modulaire NCA v7__")
print(f"Device: {cfg.DEVICE}")
print(f"Seed: {cfg.SEED}")
print(f"Mode interactif: {interactive_mode}")
print(f"Répertoire de sortie: {cfg.OUTPUT_DIR}")
print(f"Apprentissage modulaire: {cfg.MODULAR_TRAINING}")
print(f"Curriculum learning: {cfg.ENABLE_CURRICULUM}")
print(f"Étapes: {cfg.STAGE_1_EPOCHS} + {cfg.STAGE_2_EPOCHS} + {cfg.STAGE_3_EPOCHS} = {cfg.TOTAL_EPOCHS} époques")

# =============================================================================
# Gestionnaire d'obstacles progressifs
# =============================================================================

class ProgressiveObstacleManager:
    """
    Gestionnaire intelligent des obstacles selon l'étape d'apprentissage.
    Génère des environnements appropriés pour chaque phase du curriculum.
    """
    
    def __init__(self, device: str = cfg.DEVICE):
        self.device = device
        self.stage_configs = cfg.STAGE_OBSTACLE_CONFIG
        
    def generate_stage_environment(self, stage: int, size: int, source_pos: Tuple[int, int], 
                                 seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement d'obstacles adapté à l'étape courante.
        
        Args:
            stage: Numéro d'étape (1, 2, ou 3)
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles [H, W]
        """
        if stage not in self.stage_configs:
            raise ValueError(f"Étape {stage} non supportée. Étapes valides: {list(self.stage_configs.keys())}")
        
        config = self.stage_configs[stage]
        
        if stage == 1:
            return self._generate_stage_1_environment(size)
        elif stage == 2:
            return self._generate_stage_2_environment(size, source_pos, seed)
        elif stage == 3:
            return self._generate_stage_3_environment(size, source_pos, seed)
    
    def _generate_stage_1_environment(self, size: int) -> torch.Tensor:
        """Étape 1: Aucun obstacle - grille vide pour apprentissage de base."""
        return torch.zeros((size, size), dtype=torch.bool, device=self.device)
    
    def _generate_stage_2_environment(self, size: int, source_pos: Tuple[int, int], 
                                    seed: Optional[int] = None) -> torch.Tensor:
        """Étape 2: Un seul obstacle pour apprentissage du contournement."""
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)
        
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None
        
        # Un seul obstacle de taille aléatoire
        obstacle_size = torch.randint(cfg.MIN_OBSTACLE_SIZE, cfg.MAX_OBSTACLE_SIZE + 1, 
                                    (1,), generator=g, device=self.device).item()
        
        # Placement en évitant la source et les bords
        max_pos = size - obstacle_size
        if max_pos <= 1:
            return obstacle_mask  # Grille trop petite
        
        source_i, source_j = source_pos
        
        for attempt in range(100):  # Plus de tentatives pour étape 2
            i = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
            j = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
            
            # Vérifier non-chevauchement avec source
            if not (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
                obstacle_mask[i:i+obstacle_size, j:j+obstacle_size] = True
                break
        
        return obstacle_mask
    
    def _generate_stage_3_environment(self, size: int, source_pos: Tuple[int, int], 
                                    seed: Optional[int] = None) -> torch.Tensor:
        """Étape 3: Obstacles multiples pour gestion de la complexité."""
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)
        
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None
        
        config = self.stage_configs[3]
        n_obstacles = torch.randint(config['min_obstacles'], config['max_obstacles'] + 1, 
                                  (1,), generator=g, device=self.device).item()
        
        source_i, source_j = source_pos
        placed_obstacles = []
        
        for obstacle_idx in range(n_obstacles):
            obstacle_size = torch.randint(cfg.MIN_OBSTACLE_SIZE, cfg.MAX_OBSTACLE_SIZE + 1, 
                                        (1,), generator=g, device=self.device).item()
            
            max_pos = size - obstacle_size
            if max_pos <= 1:
                continue
            
            for attempt in range(50):
                i = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
                j = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
                
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
                    obstacle_mask[i:i+obstacle_size, j:j+obstacle_size] = True
                    placed_obstacles.append((i, j, obstacle_size))
                    break
        
        # Validation finale de connectivité
        if not self._validate_connectivity(obstacle_mask, source_pos):
            print("⚠️  Connectivité non garantie - génération d'un environnement plus simple")
            return self._generate_stage_2_environment(size, source_pos, seed)
        
        return obstacle_mask
    
    def _validate_connectivity(self, obstacle_mask: torch.Tensor, source_pos: Tuple[int, int]) -> bool:
        """
        Valide qu'un chemin de diffusion reste possible avec les obstacles.
        Utilise un algorithme de flood-fill simplifié.
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
        
        # Au moins 50% de la grille doit être accessible pour une bonne diffusion
        total_free_cells = (H * W) - obstacle_mask.sum().item()
        connectivity_ratio = accessible_cells / max(total_free_cells, 1)
        
        return connectivity_ratio >= 0.5
    
    def get_difficulty_metrics(self, stage: int, obstacle_mask: torch.Tensor) -> Dict[str, float]:
        """
        Calcule des métriques de difficulté pour l'environnement généré.
        
        Returns:
            Dictionnaire avec les métriques de complexité
        """
        H, W = obstacle_mask.shape
        total_cells = H * W
        obstacle_cells = obstacle_mask.sum().item()
        
        metrics = {
            'stage': stage,
            'obstacle_ratio': obstacle_cells / total_cells,
            'free_cells': total_cells - obstacle_cells,
            'complexity_score': stage * (obstacle_cells / total_cells)
        }
        
        return metrics

# =============================================================================
# Simulateur de diffusion (hérité et adapté de v6)
# =============================================================================

class DiffusionSimulator:
    """
    Simulateur de diffusion de chaleur adapté pour l'apprentissage modulaire.
    Utilise le gestionnaire d'obstacles progressifs.
    """
    
    def __init__(self, device: str = cfg.DEVICE):
        self.kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
        self.device = device
        self.obstacle_manager = ProgressiveObstacleManager(device)
    
    def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor) -> torch.Tensor:
        """Un pas de diffusion de chaleur avec obstacles."""
        x = grid.unsqueeze(0).unsqueeze(0)
        new_grid = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)
        
        # Contraintes
        new_grid[obstacle_mask] = 0.0
        new_grid[source_mask] = grid[source_mask]
        
        return new_grid
    
    def generate_stage_sequence(self, stage: int, n_steps: int, size: int, 
                              seed: Optional[int] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Génère une séquence adaptée à l'étape d'apprentissage courante.
        
        Args:
            stage: Étape d'apprentissage (1, 2, ou 3)
            n_steps: Nombre d'étapes de simulation
            size: Taille de la grille
            seed: Graine pour la reproductibilité
            
        Returns:
            (séquence, masque_source, masque_obstacles)
        """
        # Position aléatoire de la source
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
            i0 = torch.randint(2, size-2, (1,), generator=g, device=self.device).item()
            j0 = torch.randint(2, size-2, (1,), generator=g, device=self.device).item()
        else:
            i0 = torch.randint(2, size-2, (1,), device=self.device).item()
            j0 = torch.randint(2, size-2, (1,), device=self.device).item()
        
        # Génération d'obstacles selon l'étape
        obstacle_mask = self.obstacle_manager.generate_stage_environment(stage, size, (i0, j0), seed)
        
        # Initialisation
        grid = torch.zeros((size, size), device=self.device)
        grid[i0, j0] = cfg.SOURCE_INTENSITY
        
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
# Modèle NCA (hérité de v6 avec adaptations mineures)
# =============================================================================

class ImprovedNCA(nn.Module):
    """
    Neural Cellular Automaton optimisé pour l'apprentissage modulaire.
    Architecture identique à v6 mais avec support étendu pour le curriculum.
    """
    
    def __init__(self, input_size: int = 11, hidden_size: int = cfg.HIDDEN_SIZE, 
                 n_layers: int = cfg.N_LAYERS):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Architecture profonde avec normalisation
        layers = []
        current_size = input_size
        
        for i in range(n_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_size = hidden_size
        
        # Couche de sortie stabilisée
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.delta_scale = 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec scaling des deltas."""
        delta = self.network(x)
        return delta * self.delta_scale

# =============================================================================
# Updaters NCA (héritées de v6)
# =============================================================================

class OptimizedNCAUpdater:
    """
    Updater optimisé avec extraction vectorisée des patches.
    Identique à v6 pour maintenir les performances.
    """
    
    def __init__(self, model: ImprovedNCA, device: str = cfg.DEVICE):
        self.model = model
        self.device = device
    
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

class NCAUpdater:
    """
    Updater standard avec boucles Python.
    Conservé pour compatibilité et debug.
    """
    
    def __init__(self, model: ImprovedNCA, device: str = cfg.DEVICE):
        self.model = model
        self.device = device
    
    def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor) -> torch.Tensor:
        """Application standard du NCA avec boucles."""
        H, W = grid.shape
        new_grid = grid.clone()
        
        patches = []
        positions = []
        
        for i in range(1, H-1):
            for j in range(1, W-1):
                if obstacle_mask[i, j]:
                    continue
                
                patch = grid[i-1:i+2, j-1:j+2].reshape(-1)  # 9 éléments
                is_source = source_mask[i, j].float()
                is_obstacle = obstacle_mask[i, j].float()
                full_patch = torch.cat([patch, is_source.unsqueeze(0), is_obstacle.unsqueeze(0)])
                
                patches.append(full_patch)
                positions.append((i, j))
        
        if patches:
            patches_tensor = torch.stack(patches)
            deltas = self.model(patches_tensor)
            
            for idx, (i, j) in enumerate(positions):
                new_value = grid[i, j] + deltas[idx].squeeze()
                new_grid[i, j] = torch.clamp(new_value, 0.0, 1.0)
        
        # Contraintes
        new_grid[obstacle_mask] = 0.0
        new_grid[source_mask] = grid[source_mask]
        
        return new_grid

# =============================================================================
# Planificateur de curriculum (NOUVEAU)
# =============================================================================

class CurriculumScheduler:
    """
    Gestionnaire de la progression automatique entre les étapes d'apprentissage.
    Décide quand passer à l'étape suivante selon des critères adaptatifs.
    """

    def __init__(self, convergence_thresholds: Dict[int, float], patience: int = 10):
        self.thresholds = convergence_thresholds
        self.patience = patience
        self.stage_metrics_history = {stage: [] for stage in [1, 2, 3]}
        self.no_improvement_counts = {stage: 0 for stage in [1, 2, 3]}

    def should_advance_stage(self, current_stage: int, recent_losses: List[float]) -> bool:
        """
        Détermine s'il faut passer à l'étape suivante.

        Args:
            current_stage: Étape courante
            recent_losses: Pertes récentes pour évaluation

        Returns:
            True si on doit avancer à l'étape suivante
        """
        if not recent_losses or current_stage >= 3:
            return False

        # Convergence: moyenne des pertes récentes
        avg_recent_loss = np.mean(recent_losses[-5:])  # 5 dernières époques
        threshold = self.thresholds.get(current_stage, 0.05)

        # Critère principal: convergence atteinte
        converged = avg_recent_loss < threshold

        # Critère secondaire: stagnation (pas d'amélioration)
        if len(recent_losses) >= 2:
            improvement = recent_losses[-2] - recent_losses[-1]
            if improvement < 0.001:  # Amélioration négligeable
                self.no_improvement_counts[current_stage] += 1
            else:
                self.no_improvement_counts[current_stage] = 0

        stagnated = self.no_improvement_counts[current_stage] >= self.patience

        return converged or stagnated

    def adjust_learning_rate(self, optimizer, stage: int, epoch_in_stage: int):
        """Ajuste le learning rate selon l'étape et la progression."""
        base_lr = cfg.LEARNING_RATE

        # Réduction progressive par étape
        stage_multipliers = {1: 1.0, 2: 0.8, 3: 0.6}
        stage_lr = base_lr * stage_multipliers.get(stage, 0.5)

        # Décroissance cosine au sein de l'étape
        stage_epochs = {1: cfg.STAGE_1_EPOCHS, 2: cfg.STAGE_2_EPOCHS, 3: cfg.STAGE_3_EPOCHS}
        max_epochs = stage_epochs.get(stage, 50)

        cos_factor = 0.5 * (1 + np.cos(np.pi * epoch_in_stage / max_epochs))
        final_lr = stage_lr * (0.1 + 0.9 * cos_factor)  # Ne descend pas sous 10% du LR de base

        for param_group in optimizer.param_groups:
            param_group['lr'] = final_lr

    def get_stage_loss_weights(self, stage: int) -> Dict[str, float]:
        """Retourne les poids de pondération des pertes par étape."""
        weights = {
            1: {'mse': 1.0, 'convergence': 2.0, 'stability': 1.0},
            2: {'mse': 1.0, 'convergence': 1.5, 'stability': 1.5, 'adaptation': 1.0},
            3: {'mse': 1.0, 'convergence': 1.0, 'stability': 2.0, 'robustness': 1.5}
        }
        return weights.get(stage, weights[1])

# =============================================================================
# Cache de séquences optimisé par étape (NOUVEAU)
# =============================================================================

class OptimizedSequenceCache:
    """
    Cache spécialisé par étape pour l'entraînement modulaire.
    Maintient des caches séparés pour chaque étape d'apprentissage.
    """

    def __init__(self, simulator: DiffusionSimulator, device: str = cfg.DEVICE):
        self.simulator = simulator
        self.device = device
        self.stage_caches = {}  # Cache par étape
        self.cache_sizes = {1: 150, 2: 200, 3: 250}  # Plus de variété pour étapes complexes
        self.current_indices = {}

    def initialize_stage_cache(self, stage: int):
        """Initialise le cache pour une étape spécifique."""
        if stage in self.stage_caches:
            return  # Déjà initialisé

        cache_size = self.cache_sizes.get(stage, 200)
        print(f"🎯 Génération de {cache_size} séquences pour l'étape {stage}...")

        sequences = []
        for i in range(cache_size):
            if i % 50 == 0:
                print(f"   Étape {stage}: {i}/{cache_size}")

            target_seq, source_mask, obstacle_mask = self.simulator.generate_stage_sequence(
                stage=stage,
                n_steps=cfg.NCA_STEPS,
                size=cfg.GRID_SIZE
            )

            sequences.append({
                'target_seq': target_seq,
                'source_mask': source_mask,
                'obstacle_mask': obstacle_mask,
                'stage': stage
            })

        self.stage_caches[stage] = sequences
        self.current_indices[stage] = 0
        print(f"✅ Cache étape {stage} créé ({cache_size} séquences)")

    def get_stage_batch(self, stage: int, batch_size: int):
        """Récupère un batch pour l'étape spécifiée."""
        if stage not in self.stage_caches:
            self.initialize_stage_cache(stage)

        cache = self.stage_caches[stage]
        batch = []

        for _ in range(batch_size):
            batch.append(cache[self.current_indices[stage]])
            self.current_indices[stage] = (self.current_indices[stage] + 1) % len(cache)

        return batch

    def shuffle_stage_cache(self, stage: int):
        """Mélange le cache d'une étape spécifique."""
        if stage in self.stage_caches:
            import random
            random.shuffle(self.stage_caches[stage])

    def clear_stage_cache(self, stage: int):
        """Libère la mémoire du cache d'une étape."""
        if stage in self.stage_caches:
            del self.stage_caches[stage]
            del self.current_indices[stage]
            print(f"🗑️  Cache étape {stage} libéré")

# =============================================================================
# Entraîneur modulaire principal (NOUVEAU)
# =============================================================================

class ModularTrainer:
    """
    Système d'entraînement modulaire progressif.
    Gère l'apprentissage par étapes avec transitions automatiques.
    """

    def __init__(self, model: ImprovedNCA, device: str = cfg.DEVICE):
        self.model = model
        self.device = device

        # Choix de l'updater optimisé
        if cfg.USE_OPTIMIZATIONS and cfg.USE_VECTORIZED_PATCHES:
            print("🚀 Utilisation de l'updater optimisé vectorisé")
            self.updater = OptimizedNCAUpdater(model, device)
        else:
            print("⚠️  Utilisation de l'updater standard")
            self.updater = NCAUpdater(model, device)

        # Optimiseur et planificateur
        self.optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()

        # Curriculum et métriques
        if cfg.ENABLE_CURRICULUM:
            self.curriculum = CurriculumScheduler(cfg.CONVERGENCE_THRESHOLDS)
        else:
            self.curriculum = None

        # Cache optimisé par étape
        if cfg.USE_OPTIMIZATIONS and cfg.USE_SEQUENCE_CACHE:
            self.sequence_cache = OptimizedSequenceCache(simulator, device)
            self.use_cache = True
        else:
            self.use_cache = False

        # État d'entraînement
        self.current_stage = 1
        self.stage_histories = {stage: {'losses': [], 'epochs': [], 'lr': []} for stage in [1, 2, 3]}
        self.global_history = {'losses': [], 'stages': [], 'epochs': []}
        self.stage_start_epochs = {}
        self.total_epochs_trained = 0

    def train_step(self, target_sequence: List[torch.Tensor], source_mask: torch.Tensor,
                   obstacle_mask: torch.Tensor, stage: int) -> float:
        """
        Un pas d'entraînement adapté à l'étape courante.

        Args:
            target_sequence: Séquence cible
            source_mask: Masque des sources
            obstacle_mask: Masque des obstacles
            stage: Étape courante d'entraînement

        Returns:
            Perte pour ce pas
        """
        self.optimizer.zero_grad()

        # Initialisation
        grid_pred = torch.zeros_like(target_sequence[0])
        grid_pred[source_mask] = cfg.SOURCE_INTENSITY

        total_loss = torch.tensor(0.0, device=self.device)

        # Déroulement temporel
        for t_step in range(cfg.NCA_STEPS):
            target = target_sequence[t_step + 1]
            grid_pred = self.updater.step(grid_pred, source_mask, obstacle_mask)

            # Perte pondérée selon l'étape
            step_loss = self.loss_fn(grid_pred, target)
            if self.curriculum:
                weights = self.curriculum.get_stage_loss_weights(stage)
                step_loss = step_loss * weights.get('mse', 1.0)

            total_loss = total_loss + step_loss

        avg_loss = total_loss / cfg.NCA_STEPS

        # Backpropagation avec clipping
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        return avg_loss.item()

    def train_stage(self, stage: int, max_epochs: int) -> Dict[str, Any]:
        """
        Entraînement complet d'une étape spécifique.

        Args:
            stage: Numéro d'étape (1, 2, ou 3)
            max_epochs: Nombre maximum d'époques pour cette étape

        Returns:
            Dictionnaire avec les métriques de l'étape
        """
        print(f"\n🎯 === ÉTAPE {stage} - DÉBUT ===")
        stage_name = {1: "Sans obstacles", 2: "Un obstacle", 3: "Obstacles multiples"}[stage]
        print(f"📋 {stage_name}")
        print(f"⏱️  Maximum {max_epochs} époques")

        self.current_stage = stage
        self.stage_start_epochs[stage] = self.total_epochs_trained

        # Initialisation du cache pour cette étape
        if self.use_cache:
            self.sequence_cache.initialize_stage_cache(stage)

        # Métriques de l'étape
        stage_losses = []
        epoch_in_stage = 0
        early_stop = False

        # Boucle d'entraînement de l'étape
        for epoch_in_stage in range(max_epochs):
            epoch_losses = []

            # Ajustement du learning rate si curriculum activé
            if self.curriculum:
                self.curriculum.adjust_learning_rate(self.optimizer, stage, epoch_in_stage)

            # Mélange périodique du cache
            if self.use_cache and epoch_in_stage % 20 == 0:
                self.sequence_cache.shuffle_stage_cache(stage)

            # Entraînement par batch
            for _ in range(cfg.BATCH_SIZE):
                if self.use_cache:
                    batch_sequences = self.sequence_cache.get_stage_batch(stage, 1)
                    seq_data = batch_sequences[0]
                    target_seq = seq_data['target_seq']
                    source_mask = seq_data['source_mask']
                    obstacle_mask = seq_data['obstacle_mask']
                else:
                    target_seq, source_mask, obstacle_mask = simulator.generate_stage_sequence(
                        stage=stage, n_steps=cfg.NCA_STEPS, size=cfg.GRID_SIZE
                    )

                loss = self.train_step(target_seq, source_mask, obstacle_mask, stage)
                epoch_losses.append(loss)

            # Statistiques de l'époque
            avg_epoch_loss = np.mean(epoch_losses)
            stage_losses.append(avg_epoch_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Historiques
            self.stage_histories[stage]['losses'].append(avg_epoch_loss)
            self.stage_histories[stage]['epochs'].append(epoch_in_stage)
            self.stage_histories[stage]['lr'].append(current_lr)

            self.global_history['losses'].append(avg_epoch_loss)
            self.global_history['stages'].append(stage)
            self.global_history['epochs'].append(self.total_epochs_trained)

            self.total_epochs_trained += 1

            # Affichage périodique
            if epoch_in_stage % 10 == 0 or epoch_in_stage == max_epochs - 1:
                print(f"  Époque {epoch_in_stage:3d}/{max_epochs-1} | "
                      f"Loss: {avg_epoch_loss:.6f} | "
                      f"LR: {current_lr:.2e}")

            # Vérification de l'avancement automatique (curriculum)
            if (self.curriculum and cfg.ADAPTIVE_THRESHOLDS and
                epoch_in_stage >= 10):  # Minimum 10 époques par étape
                if self.curriculum.should_advance_stage(stage, stage_losses):
                    print(f"🎯 Convergence atteinte à l'époque {epoch_in_stage}")
                    print(f"   Seuil: {cfg.CONVERGENCE_THRESHOLDS[stage]:.3f}, "
                          f"Loss: {avg_epoch_loss:.6f}")
                    early_stop = True
                    break

        # Résumé de l'étape
        final_loss = stage_losses[-1] if stage_losses else float('inf')
        convergence_met = final_loss < cfg.CONVERGENCE_THRESHOLDS.get(stage, 0.05)

        stage_metrics = {
            'stage': stage,
            'epochs_trained': epoch_in_stage + 1,
            'final_loss': final_loss,
            'convergence_met': convergence_met,
            'early_stopped': early_stop,
            'loss_history': stage_losses
        }

        print(f"✅ === ÉTAPE {stage} - TERMINÉE ===")
        print(f"📊 Époques entraînées: {epoch_in_stage + 1}/{max_epochs}")
        print(f"📉 Perte finale: {final_loss:.6f}")
        print(f"🎯 Convergence: {'✅ OUI' if convergence_met else '❌ NON'}")
        print(f"⚡ Arrêt précoce: {'✅ OUI' if early_stop else '❌ NON'}")

        # Sauvegarde du checkpoint d'étape
        if cfg.SAVE_STAGE_CHECKPOINTS:
            self.save_stage_checkpoint(stage, stage_metrics)

        # Libération du cache de l'étape précédente pour économiser la mémoire
        if self.use_cache and stage > 1:
            self.sequence_cache.clear_stage_cache(stage - 1)

        return stage_metrics

    def train_full_curriculum(self) -> Dict[str, Any]:
        """
        Entraînement complet du curriculum en 3 étapes.

        Returns:
            Métriques complètes de l'entraînement modulaire
        """
        print(f"\n🚀 === DÉBUT ENTRAÎNEMENT MODULAIRE ===")
        print(f"🎯 Seed: {cfg.SEED}")
        print(f"📊 Époques totales prévues: {cfg.TOTAL_EPOCHS}")
        print(f"🔄 Étapes: {cfg.STAGE_1_EPOCHS} + {cfg.STAGE_2_EPOCHS} + {cfg.STAGE_3_EPOCHS}")
        print(f"🧠 Curriculum: {'✅ Activé' if cfg.ENABLE_CURRICULUM else '❌ Désactivé'}")

        start_time = time.time()
        self.model.train()

        # Entraînement séquentiel des 3 étapes
        all_stage_metrics = {}

        # ÉTAPE 1: Sans obstacles
        stage_1_metrics = self.train_stage(1, cfg.STAGE_1_EPOCHS)
        all_stage_metrics[1] = stage_1_metrics

        # ÉTAPE 2: Un obstacle
        stage_2_metrics = self.train_stage(2, cfg.STAGE_2_EPOCHS)
        all_stage_metrics[2] = stage_2_metrics

        # ÉTAPE 3: Obstacles multiples
        stage_3_metrics = self.train_stage(3, cfg.STAGE_3_EPOCHS)
        all_stage_metrics[3] = stage_3_metrics

        # Métriques globales
        total_time = time.time() - start_time
        total_epochs_actual = sum(metrics['epochs_trained'] for metrics in all_stage_metrics.values())

        global_metrics = {
            'total_epochs_planned': cfg.TOTAL_EPOCHS,
            'total_epochs_actual': total_epochs_actual,
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time/60:.1f} min",
            'stage_metrics': all_stage_metrics,
            'final_loss': stage_3_metrics['final_loss'],
            'all_stages_converged': all(m['convergence_met'] for m in all_stage_metrics.values()),
            'global_history': self.global_history,
            'stage_histories': self.stage_histories,
            'stage_start_epochs': self.stage_start_epochs  # AJOUT de la clé manquante
        }

        print(f"\n🎉 === ENTRAÎNEMENT MODULAIRE TERMINÉ ===")
        print(f"⏱️  Temps total: {total_time/60:.1f} minutes")
        print(f"📊 Époques totales: {total_epochs_actual}/{cfg.TOTAL_EPOCHS}")
        print(f"🎯 Toutes étapes convergées: {'✅ OUI' if global_metrics['all_stages_converged'] else '❌ NON'}")
        print(f"📉 Perte finale: {global_metrics['final_loss']:.6f}")

        # Sauvegarde du modèle final et des métriques
        self.save_final_model(global_metrics)

        return global_metrics

    def save_stage_checkpoint(self, stage: int, metrics: Dict[str, Any]):
        """Sauvegarde le checkpoint d'une étape."""
        stage_dir = Path(cfg.OUTPUT_DIR) / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarde du modèle
        model_path = stage_dir / "model_checkpoint.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stage': stage,
            'metrics': metrics,
            'config': cfg.__dict__
        }, model_path)

        # Sauvegarde des métriques en JSON
        metrics_path = stage_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"💾 Checkpoint étape {stage} sauvegardé: {stage_dir}")

    def save_final_model(self, global_metrics: Dict[str, Any]):
        """Sauvegarde le modèle final et toutes les métriques."""
        # Modèle final
        final_model_path = Path(cfg.OUTPUT_DIR) / "final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_metrics': global_metrics,
            'config': cfg.__dict__
        }, final_model_path)

        # Métriques complètes
        full_metrics_path = Path(cfg.OUTPUT_DIR) / "complete_metrics.json"
        with open(full_metrics_path, 'w') as f:
            json.dump(global_metrics, f, indent=2, default=str)

        print(f"💾 Modèle final et métriques sauvegardés: {cfg.OUTPUT_DIR}")

# =============================================================================
# Système de visualisation progressive (NOUVEAU)
# =============================================================================

class ProgressiveVisualizer:
    """
    Système de visualisation avancé pour l'apprentissage modulaire.
    Génère des animations et graphiques comparatifs par étape.
    """

    def __init__(self, interactive: bool = interactive_mode):
        self.interactive = interactive
        self.frame_data = {}  # Données par étape

    def visualize_stage_results(self, model: ImprovedNCA, stage: int,
                              vis_seed: int = args.vis_seed) -> Dict[str, Any]:
        """
        Visualise les résultats d'une étape spécifique.

        Args:
            model: Modèle NCA entraîné
            stage: Numéro d'étape à visualiser
            vis_seed: Graine pour les visualisations

        Returns:
            Dictionnaire avec les données de visualisation
        """
        print(f"\n🎨 Génération des visualisations pour l'étape {stage}...")

        # Génération de la séquence de test avec seed fixe
        torch.manual_seed(vis_seed)
        np.random.seed(vis_seed)

        target_seq, source_mask, obstacle_mask = simulator.generate_stage_sequence(
            stage=stage,
            n_steps=cfg.POSTVIS_STEPS,
            size=cfg.GRID_SIZE,
            seed=vis_seed
        )

        # Prédiction du modèle
        model.eval()
        updater = OptimizedNCAUpdater(model, cfg.DEVICE) if cfg.USE_OPTIMIZATIONS else NCAUpdater(model, cfg.DEVICE)

        # Simulation NCA avec torch.no_grad() pour éviter le gradient
        nca_sequence = []
        grid_pred = torch.zeros_like(target_seq[0])
        grid_pred[source_mask] = cfg.SOURCE_INTENSITY
        nca_sequence.append(grid_pred.clone())

        with torch.no_grad():  # Désactive le calcul de gradient pour les visualisations
            for _ in range(cfg.POSTVIS_STEPS):
                grid_pred = updater.step(grid_pred, source_mask, obstacle_mask)
                nca_sequence.append(grid_pred.clone())

        # Création des visualisations avec .detach() pour sécurité
        vis_data = {
            'stage': stage,
            'target_sequence': [t.detach().cpu().numpy() for t in target_seq],
            'nca_sequence': [t.detach().cpu().numpy() for t in nca_sequence],
            'source_mask': source_mask.detach().cpu().numpy(),
            'obstacle_mask': obstacle_mask.detach().cpu().numpy(),
            'vis_seed': vis_seed
        }

        # Sauvegarde des animations
        self._create_stage_animations(vis_data)
        self._create_stage_convergence_plot(vis_data)

        model.train()
        return vis_data

    def _create_stage_animations(self, vis_data: Dict[str, Any]):
        """Crée les animations GIF pour une étape."""
        stage = vis_data['stage']
        stage_dir = Path(cfg.OUTPUT_DIR) / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Animation comparative
        self._save_comparison_gif(
            vis_data['target_sequence'],
            vis_data['nca_sequence'],
            vis_data['obstacle_mask'],
            stage_dir / f"animation_comparaison_étape_{stage}.gif",
            f"Étape {stage} - Comparaison Cible vs NCA"
        )

        # Animation NCA seule
        self._save_single_gif(
            vis_data['nca_sequence'],
            vis_data['obstacle_mask'],
            stage_dir / f"animation_nca_étape_{stage}.gif",
            f"Étape {stage} - Prédiction NCA"
        )

        print(f"✅ Animations étape {stage} sauvegardées dans {stage_dir}")

    def _create_stage_convergence_plot(self, vis_data: Dict[str, Any]):
        """Crée le graphique de convergence pour une étape."""
        stage = vis_data['stage']
        stage_dir = Path(cfg.OUTPUT_DIR) / f"stage_{stage}"

        target_seq = vis_data['target_sequence']
        nca_seq = vis_data['nca_sequence']

        # Calcul de l'erreur temporelle
        errors = []
        for t in range(min(len(target_seq), len(nca_seq))):
            error = np.mean((target_seq[t] - nca_seq[t]) ** 2)
            errors.append(error)

        # Graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(errors, 'b-', linewidth=2, label='Erreur MSE')
        ax.axhline(y=cfg.CONVERGENCE_THRESHOLDS.get(stage, 0.05),
                  color='r', linestyle='--', label=f'Seuil convergence étape {stage}')

        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_title(f'Convergence Étape {stage} - Seed {vis_data["vis_seed"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        convergence_path = stage_dir / f"convergence_étape_{stage}.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ Graphique de convergence étape {stage} sauvegardé: {convergence_path}")

    def _save_comparison_gif(self, target_seq: List[np.ndarray], nca_seq: List[np.ndarray],
                            obstacle_mask: np.ndarray, filepath: Path, title: str):
        """Sauvegarde un GIF de comparaison côte à côte."""
        import matplotlib.animation as animation

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        def animate(frame):
            ax1.clear()
            ax2.clear()

            # Cible
            im1 = ax1.imshow(target_seq[frame], cmap='hot', vmin=0, vmax=1)
            ax1.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax1.set_title(f'Cible - t={frame}')
            ax1.set_xticks([])
            ax1.set_yticks([])

            # NCA
            im2 = ax2.imshow(nca_seq[frame], cmap='hot', vmin=0, vmax=1)
            ax2.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax2.set_title(f'NCA - t={frame}')
            ax2.set_xticks([])
            ax2.set_yticks([])

            return [im1, im2]

        n_frames = min(len(target_seq), len(nca_seq))
        ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200, blit=False)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()

    def _save_single_gif(self, sequence: List[np.ndarray], obstacle_mask: np.ndarray,
                        filepath: Path, title: str):
        """Sauvegarde un GIF d'une séquence unique."""
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(8, 8))

        def animate(frame):
            ax.clear()
            im = ax.imshow(sequence[frame], cmap='hot', vmin=0, vmax=1)
            ax.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax.set_title(f'{title} - t={frame}')
            ax.set_xticks([])
            ax.set_yticks([])
            return [im]

        ani = animation.FuncAnimation(fig, animate, frames=len(sequence), interval=200, blit=False)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()

    def create_curriculum_summary(self, global_metrics: Dict[str, Any]):
        """Crée un résumé visuel complet du curriculum d'apprentissage."""
        print("\n🎨 Génération du résumé visuel du curriculum...")

        # Graphique de progression globale
        self._plot_curriculum_progression(global_metrics)

        # Comparaison inter-étapes
        self._plot_stage_comparison(global_metrics)

        # Métriques de performance
        self._plot_performance_metrics(global_metrics)

        print("✅ Résumé visuel complet généré")

    def _plot_curriculum_progression(self, metrics: Dict[str, Any]):
        """Graphique de la progression globale du curriculum."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Historique des pertes avec codes couleur par étape
        losses = metrics['global_history']['losses']
        stages = metrics['global_history']['stages']
        epochs = metrics['global_history']['epochs']

        stage_colors = {1: 'green', 2: 'orange', 3: 'red'}

        for stage in [1, 2, 3]:
            stage_indices = [i for i, s in enumerate(stages) if s == stage]
            stage_losses = [losses[i] for i in stage_indices]
            stage_epochs = [epochs[i] for i in stage_indices]

            if stage_losses:
                ax1.plot(stage_epochs, stage_losses,
                        color=stage_colors[stage],
                        label=f'Étape {stage}',
                        linewidth=2)

        # Seuils de convergence
        for stage in [1, 2, 3]:
            threshold = cfg.CONVERGENCE_THRESHOLDS.get(stage, 0.05)
            ax1.axhline(y=threshold, color=stage_colors[stage],
                       linestyle='--', alpha=0.7,
                       label=f'Seuil étape {stage}')

        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Perte MSE')
        ax1.set_title('Progression du Curriculum d\'Apprentissage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Learning rate par étape
        for stage in [1, 2, 3]:
            stage_history = metrics['stage_histories'][stage]
            if stage_history['lr']:
                stage_epochs_local = [metrics['stage_start_epochs'].get(stage, 0) + e
                                    for e in stage_history['epochs']]
                ax2.plot(stage_epochs_local, stage_history['lr'],
                        color=stage_colors[stage],
                        label=f'LR Étape {stage}',
                        linewidth=2)

        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Évolution du Learning Rate par Étape')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(Path(cfg.OUTPUT_DIR) / "curriculum_progression.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_stage_comparison(self, metrics: Dict[str, Any]):
        """Graphique de comparaison entre étapes."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        stages = [1, 2, 3]
        stage_names = ["Sans obstacles", "Un obstacle", "Obstacles multiples"]
        stage_colors = ['green', 'orange', 'red']

        # Pertes finales par étape
        final_losses = [metrics['stage_metrics'][s]['final_loss'] for s in stages]
        convergence_status = [metrics['stage_metrics'][s]['convergence_met'] for s in stages]

        bars = ax1.bar(stage_names, final_losses, color=stage_colors, alpha=0.7)
        for i, (bar, converged) in enumerate(zip(bars, convergence_status)):
            if converged:
                bar.set_edgecolor('darkgreen')
                bar.set_linewidth(3)

        ax1.set_ylabel('Perte finale')
        ax1.set_title('Perte Finale par Étape')
        ax1.set_yscale('log')

        # Époques utilisées par étape
        epochs_used = [metrics['stage_metrics'][s]['epochs_trained'] for s in stages]
        epochs_planned = [cfg.STAGE_1_EPOCHS, cfg.STAGE_2_EPOCHS, cfg.STAGE_3_EPOCHS]

        x = np.arange(len(stages))
        width = 0.35

        ax2.bar(x - width/2, epochs_planned, width, label='Prévues', alpha=0.7, color='lightblue')
        ax2.bar(x + width/2, epochs_used, width, label='Utilisées', alpha=0.7, color='darkblue')

        ax2.set_xlabel('Étape')
        ax2.set_ylabel('Nombre d\'époques')
        ax2.set_title('Époques Prévues vs Utilisées')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stage_names, rotation=15)
        ax2.legend()

        # Temps de convergence
        convergence_times = []
        for stage in stages:
            stage_losses = metrics['stage_metrics'][stage]['loss_history']
            threshold = cfg.CONVERGENCE_THRESHOLDS.get(stage, 0.05)

            convergence_epoch = None
            for i, loss in enumerate(stage_losses):
                if loss < threshold:
                    convergence_epoch = i
                    break

            convergence_times.append(convergence_epoch if convergence_epoch else len(stage_losses))

        ax3.plot(stages, convergence_times, 'o-', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('Étape')
        ax3.set_ylabel('Époque de convergence')
        ax3.set_title('Vitesse de Convergence par Étape')
        ax3.grid(True, alpha=0.3)

        # Efficacité (convergence / époques utilisées)
        efficiency = [(1.0 if convergence_status[i] else 0.5) / max(epochs_used[i], 1)
                     for i in range(len(stages))]

        ax4.bar(stage_names, efficiency, color=stage_colors, alpha=0.7)
        ax4.set_ylabel('Efficacité (convergence/époque)')
        ax4.set_title('Efficacité d\'Apprentissage par Étape')

        plt.tight_layout()
        plt.savefig(Path(cfg.OUTPUT_DIR) / "stage_comparison.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_performance_metrics(self, metrics: Dict[str, Any]):
        """Graphique des métriques de performance globales."""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Résumé textuel des performances
        total_time = metrics['total_time_seconds']
        total_epochs = metrics['total_epochs_actual']
        all_converged = metrics['all_stages_converged']
        final_loss = metrics['final_loss']

        summary_text = f"""
🎯 RÉSUMÉ ENTRAÎNEMENT MODULAIRE NCA v7__

📊 STATISTIQUES GLOBALES:
   • Seed: {cfg.SEED}
   • Temps total: {total_time/60:.1f} minutes ({total_time:.1f}s)
   • Époques totales: {total_epochs}/{cfg.TOTAL_EPOCHS}
   • Toutes étapes convergées: {'✅ OUI' if all_converged else '❌ NON'}
   • Perte finale: {final_loss:.6f}

🏆 PERFORMANCE PAR ÉTAPE:"""

        for stage in [1, 2, 3]:
            stage_data = metrics['stage_metrics'][stage]
            stage_name = {1: "Sans obstacles", 2: "Un obstacle", 3: "Obstacles multiples"}[stage]

            summary_text += f"""
   • Étape {stage} ({stage_name}):
     - Époques: {stage_data['epochs_trained']} (convergée: {'✅' if stage_data['convergence_met'] else '❌'})
     - Perte finale: {stage_data['final_loss']:.6f}
     - Arrêt précoce: {'✅' if stage_data['early_stopped'] else '❌'}"""

        summary_text += f"""

⚙️  CONFIGURATION:
   • Curriculum learning: {'✅' if cfg.ENABLE_CURRICULUM else '❌'}
   • Seuils adaptatifs: {'✅' if cfg.ADAPTIVE_THRESHOLDS else '❌'}
   • Cache optimisé: {'✅' if cfg.USE_SEQUENCE_CACHE else '❌'}
   • Updater vectorisé: {'✅' if cfg.USE_VECTORIZED_PATCHES else '❌'}

📈 ARCHITECTURE:
   • Taille grille: {cfg.GRID_SIZE}x{cfg.GRID_SIZE}
   • Couches cachées: {cfg.HIDDEN_SIZE} neurones, {cfg.N_LAYERS} couches
   • Pas temporels NCA: {cfg.NCA_STEPS}
   • Taille batch: {cfg.BATCH_SIZE}
        """

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Résumé Performance Entraînement Modulaire NCA v7__',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(Path(cfg.OUTPUT_DIR) / "performance_summary.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

# =============================================================================
# Fonction principale d'exécution (NOUVEAU)
# =============================================================================

def main():
    """
    Fonction principale pour l'entraînement modulaire progressif.
    Orchestre tout le processus d'apprentissage par curriculum.
    """
    print(f"\n" + "="*80)
    print(f"🚀 NEURAL CELLULAR AUTOMATON - APPRENTISSAGE MODULAIRE v7__")
    print(f"="*80)

    try:
        # Initialisation du modèle
        print("\n🔧 Initialisation du modèle...")
        model = ImprovedNCA(
            input_size=11,  # 9 (patch 3x3) + 1 (source) + 1 (obstacle)
            hidden_size=cfg.HIDDEN_SIZE,
            n_layers=cfg.N_LAYERS
        ).to(cfg.DEVICE)

        print(f"📊 Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")

        # Initialisation de l'entraîneur modulaire
        print("🎯 Initialisation de l'entraîneur modulaire...")
        trainer = ModularTrainer(model, cfg.DEVICE)

        # Lancement de l'entraînement complet
        print("🚀 Lancement de l'entraînement modulaire...")
        global_metrics = trainer.train_full_curriculum()

        # Génération des visualisations progressives
        print("\n🎨 Génération des visualisations...")
        visualizer = ProgressiveVisualizer(interactive_mode)

        # Visualisation par étape avec le modèle final
        for stage in [1, 2, 3]:
            stage_vis = visualizer.visualize_stage_results(model, stage, args.vis_seed)

        # Résumé visuel complet du curriculum
        visualizer.create_curriculum_summary(global_metrics)

        # Rapport final
        print(f"\n" + "="*80)
        print(f"🎉 ENTRAÎNEMENT MODULAIRE TERMINÉ AVEC SUCCÈS!")
        print(f"="*80)
        print(f"📁 Résultats sauvegardés dans: {cfg.OUTPUT_DIR}")
        print(f"⏱️  Temps total: {global_metrics['total_time_formatted']}")
        print(f"📊 Époques: {global_metrics['total_epochs_actual']}/{global_metrics['total_epochs_planned']}")
        print(f"🎯 Convergence: {'✅ TOUTES' if global_metrics['all_stages_converged'] else '❌ PARTIELLE'}")
        print(f"📉 Perte finale: {global_metrics['final_loss']:.6f}")

        # Détail par étape
        print(f"\n📋 DÉTAIL PAR ÉTAPE:")
        for stage in [1, 2, 3]:
            stage_data = global_metrics['stage_metrics'][stage]
            stage_name = {1: "Sans obstacles", 2: "Un obstacle", 3: "Obstacles multiples"}[stage]
            status = "✅ CONVERGÉE" if stage_data['convergence_met'] else "❌ NON CONVERGÉE"
            print(f"   Étape {stage} ({stage_name}): {status} - {stage_data['final_loss']:.6f}")

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
