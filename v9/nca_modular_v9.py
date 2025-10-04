"""
NCA Modulaire v9 - Architecture découplée avec stages modulaires.
Nouvelle architecture permettant l'ajout facile de nouveaux stages.
"""

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

# Import de l'architecture modulaire
from stages import ModularStageManager, BaseStage

# Import du module de visualisation v9
from visualize_modular_progressive_obstacles_variable_intensity import create_complete_visualization_suite

# =============================================================================
# Configuration globale simplifiée
# =============================================================================

class GlobalConfig:
    """
    Configuration globale simplifiée - ne contient plus les détails des stages.
    Les stages gèrent maintenant leur propre configuration.
    """
    def __init__(self, seed: int = 123):
        # Paramètres matériels de base
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SEED = seed

        # Paramètres de grille
        self.GRID_SIZE = 16
        self.SOURCE_INTENSITY = 1.0

        # Paramètres d'entraînement de base
        self.TOTAL_EPOCHS = 500
        self.NCA_STEPS = 20
        self.LEARNING_RATE = 1e-3
        self.BATCH_SIZE = 4

        # Paramètres du modèle
        self.HIDDEN_SIZE = 128
        self.N_LAYERS = 3

        # Paramètres d'obstacles globaux
        self.MIN_OBSTACLE_SIZE = 2
        self.MAX_OBSTACLE_SIZE = 4

        # Paramètres de visualisation
        self.PREVIS_STEPS = 30
        self.POSTVIS_STEPS = 50
        self.SAVE_ANIMATIONS = True
        self.SAVE_STAGE_CHECKPOINTS = True
        self.OUTPUT_DIR = "nca_outputs_modular_progressive_obstacles_variable_intensity"

        # Optimisations
        self.USE_OPTIMIZATIONS = True
        self.USE_VECTORIZED_PATCHES = True

        # Configuration optionnelle de séquence personnalisée
        # self.CUSTOM_STAGE_SEQUENCE = [1, 2, 3, 4]  # Si on veut personnaliser


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description='Neural Cellular Automaton - Architecture Modulaire v9',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments de base
    parser.add_argument('--seed', type=int, default=123,
                       help='Graine aléatoire pour la reproductibilité')
    parser.add_argument('--vis-seed', type=int, default=3333,
                       help='Graine pour les visualisations')
    parser.add_argument('--total-epochs', type=int, default=500,
                       help='Nombre total d\'époques d\'entraînement')
    parser.add_argument('--grid-size', type=int, default=16,
                       help='Taille de la grille')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Taille des batches')
    parser.add_argument('--save-checkpoints', action='store_true', default=True,
                       help='Sauvegarder les checkpoints par stage')

    return parser.parse_args()


# =============================================================================
# Simulateur de diffusion adapté à l'architecture modulaire
# =============================================================================

class ModularDiffusionSimulator:
    """
    Simulateur de diffusion adapté pour l'architecture modulaire.
    Interface avec les stages pour générer les environnements.
    """

    def __init__(self, device: str):
        self.kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
        self.device = device

    def step(self, grid: torch.Tensor, source_mask: torch.Tensor,
             obstacle_mask: torch.Tensor, source_intensity: Optional[float] = None) -> torch.Tensor:
        """Un pas de diffusion de chaleur avec support intensité variable."""
        x = grid.unsqueeze(0).unsqueeze(0)
        new_grid = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)

        # Contraintes
        new_grid[obstacle_mask] = 0.0

        # Support intensité variable pour Stage 4
        if source_intensity is not None:
            new_grid[source_mask] = source_intensity
        else:
            new_grid[source_mask] = grid[source_mask]

        return new_grid

    def generate_sequence_with_stage(self, stage: BaseStage, n_steps: int, size: int,
                                   source_intensity: Optional[float] = None,
                                   seed: Optional[int] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, float]:
        """
        Génère une séquence en utilisant un stage spécifique.
        
        Args:
            stage: Instance du stage à utiliser
            n_steps: Nombre d'étapes de simulation
            size: Taille de la grille
            source_intensity: Intensité spécifique (pour Stage 4)
            seed: Graine pour la reproductibilité
            
        Returns:
            (séquence, masque_source, masque_obstacles, intensité_utilisée)
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

        # Génération d'obstacles via le stage
        obstacle_mask = stage.generate_environment(size, (i0, j0), seed)

        # Gestion intensité variable pour Stage 4
        if hasattr(stage, 'sample_source_intensity') and source_intensity is None:
            # Stage 4 avec échantillonnage automatique
            used_intensity = stage.sample_source_intensity(0.5)  # Milieu de progression
        elif source_intensity is not None:
            # Intensité spécifiée
            used_intensity = source_intensity
        else:
            # Intensité standard
            used_intensity = 1.0

        # Initialisation
        grid = torch.zeros((size, size), device=self.device)
        grid[i0, j0] = used_intensity

        source_mask = torch.zeros_like(grid, dtype=torch.bool)
        source_mask[i0, j0] = True

        # S'assurer que la source n'est pas dans un obstacle
        if obstacle_mask[i0, j0]:
            obstacle_mask[i0, j0] = False

        # Simulation temporelle
        sequence = [grid.clone()]
        for _ in range(n_steps):
            grid = self.step(grid, source_mask, obstacle_mask,
                           used_intensity if hasattr(stage, 'sample_source_intensity') else None)
            sequence.append(grid.clone())

        return sequence, source_mask, obstacle_mask, used_intensity


# =============================================================================
# Modèle NCA (inchangé)
# =============================================================================

class ImprovedNCA(nn.Module):
    """Neural Cellular Automaton optimisé."""

    def __init__(self, input_size: int = 11, hidden_size: int = 128, n_layers: int = 3):
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

        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)
        self.delta_scale = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.network(x)
        return delta * self.delta_scale


# =============================================================================
# Updater NCA modulaire
# =============================================================================

class ModularNCAUpdater:
    """Updater NCA adapté à l'architecture modulaire."""

    def __init__(self, model: ImprovedNCA, device: str):
        self.model = model
        self.device = device

    def step(self, grid: torch.Tensor, source_mask: torch.Tensor,
             obstacle_mask: torch.Tensor, source_intensity: Optional[float] = None) -> torch.Tensor:
        """Application du NCA avec support intensité variable."""
        H, W = grid.shape

        # Extraction vectorisée des patches 3x3
        grid_padded = F.pad(grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        patches = F.unfold(grid_padded, kernel_size=3, stride=1)
        patches = patches.squeeze(0).transpose(0, 1)

        # Features additionnelles
        source_flat = source_mask.flatten().float().unsqueeze(1)
        obstacle_flat = obstacle_mask.flatten().float().unsqueeze(1)
        full_patches = torch.cat([patches, source_flat, obstacle_flat], dim=1)

        # Application sur positions valides
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
        if source_intensity is not None:
            new_grid[source_mask] = source_intensity
        else:
            new_grid[source_mask] = grid[source_mask]

        return new_grid


# =============================================================================
# Entraîneur modulaire simplifié
# =============================================================================

class ModularTrainer:
    """
    Entraîneur simplifié qui délègue la logique aux stages.
    Plus besoin de connaître les détails de chaque stage.
    """

    def __init__(self, model: ImprovedNCA, global_config: GlobalConfig):
        self.model = model
        self.config = global_config
        self.device = global_config.DEVICE
        
        self.updater = ModularNCAUpdater(model, self.device)
        self.simulator = ModularDiffusionSimulator(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=global_config.LEARNING_RATE, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss()

        # Gestionnaire de stages modulaire
        self.stage_manager = ModularStageManager(global_config, self.device)

    def train_stage_callback(self, stage: BaseStage, max_epochs: int, early_stopping: bool = True) -> Dict[str, Any]:
        """
        Fonction callback pour l'entraînement d'un stage.
        Appelée par le gestionnaire de stages.
        """
        print(f"🎯 Entraînement {stage.config.name} - {max_epochs} époques max")
        
        stage_losses = []
        early_stop = False
        
        for epoch_in_stage in range(max_epochs):
            epoch_losses = []
            
            # Ajustement du learning rate via le stage
            lr = stage.get_learning_rate_schedule(epoch_in_stage, max_epochs, self.config.LEARNING_RATE)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Entraînement par batch
            for _ in range(self.config.BATCH_SIZE):
                # Génération de données via le stage
                if hasattr(stage, 'sample_source_intensity'):
                    # Stage 4 avec intensités variables
                    epoch_progress = epoch_in_stage / max(max_epochs - 1, 1)
                    source_intensity = stage.sample_source_intensity(epoch_progress)
                else:
                    source_intensity = None
                
                target_seq, source_mask, obstacle_mask, used_intensity = \
                    self.simulator.generate_sequence_with_stage(
                        stage, self.config.NCA_STEPS, self.config.GRID_SIZE,
                        source_intensity
                    )
                
                loss = self._train_step(target_seq, source_mask, obstacle_mask, stage, used_intensity)
                epoch_losses.append(loss)
            
            # Statistiques de l'époque
            avg_epoch_loss = np.mean(epoch_losses)
            stage_losses.append(avg_epoch_loss)
            
            # Hook post-époque du stage
            stage.post_epoch_hook(epoch_in_stage, avg_epoch_loss, {'lr': lr})
            
            # Affichage périodique
            if epoch_in_stage % 10 == 0 or epoch_in_stage == max_epochs - 1:
                print(f"  Époque {epoch_in_stage:3d}/{max_epochs-1} | "
                      f"Loss: {avg_epoch_loss:.6f} | LR: {lr:.2e}")
                
                # Affichage des métriques du stage si disponibles
                if hasattr(stage, 'get_intensity_statistics'):
                    stats = stage.get_intensity_statistics()
                    if stats['count'] > 0:
                        print(f"    Intensités: moy={stats['mean']:.3f}, "
                              f"std={stats['std']:.3f}, plage=[{stats['min']:.3f}, {stats['max']:.3f}]")
            
            # Vérification de convergence via le stage
            if early_stopping and epoch_in_stage >= 10:
                if stage.validate_convergence(stage_losses, epoch_in_stage):
                    print(f"🎯 Convergence atteinte à l'époque {epoch_in_stage}")
                    early_stop = True
                    break
        
        return {
            'epochs_trained': epoch_in_stage + 1,
            'final_loss': stage_losses[-1] if stage_losses else float('inf'),
            'converged': early_stop,
            'loss_history': stage_losses
        }

    def _train_step(self, target_sequence: List[torch.Tensor], source_mask: torch.Tensor,
                   obstacle_mask: torch.Tensor, stage: BaseStage, source_intensity: float) -> float:
        """Un pas d'entraînement."""
        self.optimizer.zero_grad()

        # Initialisation
        grid_pred = torch.zeros_like(target_sequence[0])
        grid_pred[source_mask] = source_intensity

        total_loss = torch.tensor(0.0, device=self.device)

        # Déroulement temporel
        for t_step in range(self.config.NCA_STEPS):
            target = target_sequence[t_step + 1]
            grid_pred = self.updater.step(grid_pred, source_mask, obstacle_mask,
                                        source_intensity if hasattr(stage, 'sample_source_intensity') else None)

            # Perte pondérée selon le stage
            step_loss = self.loss_fn(grid_pred, target)
            loss_weights = stage.get_loss_weights()
            step_loss = step_loss * loss_weights.get('mse', 1.0)

            total_loss = total_loss + step_loss

        avg_loss = total_loss / self.config.NCA_STEPS
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return avg_loss.item()

    def train_full_curriculum(self) -> Dict[str, Any]:
        """Exécute le curriculum complet via le gestionnaire de stages."""
        return self.stage_manager.execute_full_curriculum(self.train_stage_callback)

    def save_stage_checkpoint(self, stage_id: int):
        """Sauvegarde le checkpoint d'un stage."""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        output_dir = Path(self.config.OUTPUT_DIR)
        self.stage_manager.save_stage_checkpoint(stage_id, model_state, output_dir)

    def generate_stage_visualizations(self, vis_seed: int):
        """Génère les visualisations pour chaque stage en utilisant les visualiseurs spécialisés."""
        print(f"🎨 Génération des visualisations par stage...")
        
        # Import des visualiseurs spécialisés
        try:
            from stages.visualizers import get_visualizer
            has_specialized_visualizers = True
            print(f"  ✓ Visualiseurs spécialisés disponibles")
        except ImportError:
            has_specialized_visualizers = False
            print(f"  ⚠️ Visualiseurs spécialisés non disponibles, utilisation du visualiseur générique")
        
        # Configuration matplotlib
        matplotlib.use('Agg')  # Mode non-interactif pour sauvegarde
        
        # Données pour visualisations multi-intensités du Stage 4
        stage4_intensity_data = {}
        
        for stage_id in self.stage_manager.stage_sequence:
            if stage_id in self.stage_manager.active_stages:
                stage = self.stage_manager.active_stages[stage_id]
                print(f"  🎨 Visualisations pour Stage {stage_id} ({stage.config.name})...")
                
                # Création du répertoire du stage
                stage_dir = Path(self.config.OUTPUT_DIR) / f"stage_{stage_id}"
                stage_dir.mkdir(parents=True, exist_ok=True)
                
                # Récupération du visualiseur spécialisé si disponible
                specialized_visualizer = get_visualizer(stage_id) if has_specialized_visualizers else None
                
                # Génération de la visualisation principale
                torch.manual_seed(vis_seed)
                np.random.seed(vis_seed)
                
                # Gestion spéciale pour Stage 4 (intensités variables)
                if hasattr(stage, 'sample_source_intensity'):
                    source_intensity = stage.sample_source_intensity(0.5)  # Milieu de progression
                else:
                    source_intensity = None
                
                target_seq, source_mask, obstacle_mask, used_intensity = \
                    self.simulator.generate_sequence_with_stage(
                        stage, self.config.POSTVIS_STEPS, self.config.GRID_SIZE,
                        source_intensity, vis_seed
                    )
                
                # Prédiction du modèle pour la visualisation principale
                self.model.eval()
                nca_sequence = []
                grid_pred = torch.zeros_like(target_seq[0])
                grid_pred[source_mask] = used_intensity
                nca_sequence.append(grid_pred.clone())
                
                with torch.no_grad():
                    for _ in range(self.config.POSTVIS_STEPS):
                        grid_pred = self.updater.step(
                            grid_pred, source_mask, obstacle_mask,
                            used_intensity if hasattr(stage, 'sample_source_intensity') else None
                        )
                        nca_sequence.append(grid_pred.clone())
                
                # Utilisation du visualiseur spécialisé si disponible
                if specialized_visualizer:
                    # Récupération d'informations supplémentaires pour le Stage 4
                    intensity_history = None
                    if stage_id == 4 and hasattr(stage, 'intensity_manager'):
                        try:
                            intensity_history = stage.intensity_manager.intensity_history
                        except:
                            pass
                    
                    # Création des visualisations spécialisées
                    print(f"    🎨 Utilisation du visualiseur spécialisé pour Stage {stage_id}")
                    specialized_visualizer.create_visualizations(
                        stage_dir, target_seq, nca_sequence,
                        obstacle_mask, source_mask, used_intensity,
                        vis_seed, intensity_history if stage_id == 4 else None
                    )
                    
                    # Stockage des données pour visualisation multi-intensités du Stage 4
                    if stage_id == 4:
                        # Stockage des données principales
                        stage4_intensity_data[used_intensity] = {
                            'target_sequence': [t.detach().cpu().numpy() for t in target_seq],
                            'nca_sequence': [t.detach().cpu().numpy() for t in nca_sequence],
                            'source_mask': source_mask.detach().cpu().numpy(),
                            'obstacle_mask': obstacle_mask.detach().cpu().numpy(),
                        }
                else:
                    # Utilisation du visualiseur générique
                    print(f"    🎨 Utilisation du visualiseur générique pour Stage {stage_id}")
                    self._create_stage_animations(
                        stage_id, stage_dir, target_seq, nca_sequence,
                        obstacle_mask, used_intensity, suffix=""
                    )
                    self._create_stage_convergence_plot(
                        stage_id, stage_dir, target_seq, nca_sequence, vis_seed, suffix=""
                    )
                
                # NOUVEAU : Visualisations supplémentaires pour Stage 4
                if stage_id == 4 and hasattr(stage, 'sample_source_intensity'):
                    print(f"    🎨 Visualisations supplémentaires pour Stage 4 avec intensités fixes...")
                    
                    # Intensités supplémentaires à tester
                    additional_intensities = [0.25, 0.5]
                    
                    for intensity in additional_intensities:
                        print(f"      🔸 Génération avec intensité {intensity}...")
                        
                        # Génération avec intensité fixe
                        torch.manual_seed(vis_seed + int(intensity * 100))  # Seed légèrement différent
                        np.random.seed(vis_seed + int(intensity * 100))
                        
                        target_seq_fixed, source_mask_fixed, obstacle_mask_fixed, _ = \
                            self.simulator.generate_sequence_with_stage(
                                stage, self.config.POSTVIS_STEPS, self.config.GRID_SIZE,
                                intensity, vis_seed + int(intensity * 100)
                            )
                        
                        # Prédiction du modèle avec intensité fixe
                        nca_sequence_fixed = []
                        grid_pred_fixed = torch.zeros_like(target_seq_fixed[0])
                        grid_pred_fixed[source_mask_fixed] = intensity
                        nca_sequence_fixed.append(grid_pred_fixed.clone())
                        
                        with torch.no_grad():
                            for _ in range(self.config.POSTVIS_STEPS):
                                grid_pred_fixed = self.updater.step(
                                    grid_pred_fixed, source_mask_fixed, obstacle_mask_fixed, intensity
                                )
                                nca_sequence_fixed.append(grid_pred_fixed.clone())
                        
                        # Stockage des données pour visualisation comparative
                        if specialized_visualizer and stage_id == 4:
                            stage4_intensity_data[intensity] = {
                                'target_sequence': [t.detach().cpu().numpy() for t in target_seq_fixed],
                                'nca_sequence': [t.detach().cpu().numpy() for t in nca_sequence_fixed],
                                'source_mask': source_mask_fixed.detach().cpu().numpy(),
                                'obstacle_mask': obstacle_mask_fixed.detach().cpu().numpy(),
                            }
                        
                        # Création des visualisations
                        if specialized_visualizer:
                            # Utilisation du visualiseur spécialisé
                            suffix = f"_intensity_{intensity:.2f}".replace(".", "")
                            specialized_visualizer._create_animations_with_intensity(
                                stage_dir, target_seq_fixed, nca_sequence_fixed,
                                obstacle_mask_fixed, intensity, suffix
                            )
                            specialized_visualizer._create_convergence_plot_with_intensity(
                                stage_dir, target_seq_fixed, nca_sequence_fixed,
                                vis_seed, intensity, threshold=stage.config.convergence_threshold, suffix=suffix
                            )
                            specialized_visualizer._create_intensity_influence_plot(
                                stage_dir, nca_sequence_fixed, intensity
                            )
                        else:
                            # Utilisation du visualiseur générique
                            intensity_suffix = f"_intensity_{intensity:.2f}".replace(".", "")
                            self._create_stage_animations(
                                stage_id, stage_dir, target_seq_fixed, nca_sequence_fixed,
                                obstacle_mask_fixed, intensity, suffix=intensity_suffix
                            )
                            self._create_stage_convergence_plot(
                                stage_id, stage_dir, target_seq_fixed, nca_sequence_fixed,
                                vis_seed, suffix=intensity_suffix
                            )
                
                self.model.train()
                
                # Création des visualisations comparatives pour Stage 4
                if stage_id == 4 and specialized_visualizer and len(stage4_intensity_data) > 1:
                    intensities = sorted(stage4_intensity_data.keys())
                    vis_data_list = [stage4_intensity_data[i] for i in intensities]
                    
                    print(f"    🎨 Création des visualisations comparatives entre intensités...")
                    specialized_visualizer._create_intensity_comparison_plot(
                        stage_dir, intensities, vis_data_list, vis_seed
                    )
                    specialized_visualizer._create_multi_intensity_animation(
                        stage_dir, intensities, vis_data_list
                    )
                
                print(f"    ✅ Visualisations Stage {stage_id} sauvegardées dans {stage_dir}")
    
    def _create_stage_animations(self, stage_id: int, stage_dir: Path,
                                target_seq: List[torch.Tensor], nca_seq: List[torch.Tensor],
                                obstacle_mask: torch.Tensor, source_intensity: float, suffix: str = ""):
        """Crée les animations GIF pour un stage."""
        import matplotlib.animation as animation
        
        # Conversion en numpy
        target_np = [t.detach().cpu().numpy() for t in target_seq]
        nca_np = [t.detach().cpu().numpy() for t in nca_seq]
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        
        # Animation comparative
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        def animate_comparison(frame):
            ax1.clear()
            ax2.clear()
            
            # Cible
            im1 = ax1.imshow(target_np[frame], cmap='hot', vmin=0, vmax=1)
            ax1.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax1.set_title(f'Cible - t={frame} (I={source_intensity:.3f})')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # NCA
            im2 = ax2.imshow(nca_np[frame], cmap='hot', vmin=0, vmax=1)
            ax2.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax2.set_title(f'NCA - t={frame} (I={source_intensity:.3f})')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            return [im1, im2]
        
        n_frames = min(len(target_np), len(nca_np))
        ani = animation.FuncAnimation(
            fig, animate_comparison, frames=n_frames, interval=200, blit=False
        )
        
        comparison_path = stage_dir / f"animation_comparaison_stage_{stage_id}{suffix}.gif"
        ani.save(comparison_path, writer='pillow', fps=5)
        plt.close()
        
        # Animation NCA seule
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate_nca(frame):
            ax.clear()
            im = ax.imshow(nca_np[frame], cmap='hot', vmin=0, vmax=1)
            ax.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax.set_title(f'Stage {stage_id} - NCA - t={frame} (I={source_intensity:.3f})')
            ax.set_xticks([])
            ax.set_yticks([])
            return [im]
        
        ani_nca = animation.FuncAnimation(
            fig, animate_nca, frames=len(nca_np), interval=200, blit=False
        )
        
        nca_path = stage_dir / f"animation_nca_stage_{stage_id}{suffix}.gif"
        ani_nca.save(nca_path, writer='pillow', fps=5)
        plt.close()
    
    def _create_stage_convergence_plot(self, stage_id: int, stage_dir: Path,
                                     target_seq: List[torch.Tensor], nca_seq: List[torch.Tensor],
                                     vis_seed: int, suffix: str = ""):
        """Crée le graphique de convergence pour un stage."""
        # Calcul de l'erreur temporelle
        errors = []
        for t in range(min(len(target_seq), len(nca_seq))):
            target_np = target_seq[t].detach().cpu().numpy()
            nca_np = nca_seq[t].detach().cpu().numpy()
            error = np.mean((target_np - nca_np) ** 2)
            errors.append(error)
        
        # Graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(errors, 'b-', linewidth=2, label='Erreur MSE')
        
        # Ligne de seuil si disponible
        if stage_id in self.stage_manager.active_stages:
            stage = self.stage_manager.active_stages[stage_id]
            threshold = stage.config.convergence_threshold
            ax.axhline(y=threshold, color='r', linestyle='--',
                      label=f'Seuil convergence Stage {stage_id}')
        
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        
        # Titre adapté selon le suffixe
        if suffix:
            # Extraction de l'intensité depuis le suffixe
            intensity_str = suffix.replace("_intensity_", "")
            if intensity_str == "025":
                intensity_value = 0.25
            elif intensity_str == "050":
                intensity_value = 0.5
            else:
                intensity_value = float(f"0.{intensity_str}")
            ax.set_title(f'Convergence Stage {stage_id} - Intensité {intensity_value} - Seed {vis_seed}')
        else:
            ax.set_title(f'Convergence Stage {stage_id} - Seed {vis_seed}')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = stage_dir / f"convergence_stage_{stage_id}{suffix}.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()

    def generate_complete_visualization_suite(self, results: Dict[str, Any]):
        """Génère la suite complète de visualisations après l'entraînement."""
        print(f"🎨 Génération de la suite complète de visualisations...")
        
        try:
            # Préparation des métriques pour la suite de visualisations
            adapted_results = self._adapt_results_for_visualization(results)
            
            # Appel de la suite de visualisation complète
            create_complete_visualization_suite(
                self.model, adapted_results, self.simulator, self.config
            )
            
            print(f"✅ Suite complète de visualisations générée")
            
        except Exception as e:
            print(f"⚠️  Erreur lors de la génération de la suite complète: {e}")
            print(f"   Les visualisations par stage sont disponibles")
    
    def _adapt_results_for_visualization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Adapte les résultats pour la compatibilité avec le système de visualisation."""
        # Conversion du format modulaire vers le format attendu par create_complete_visualization_suite
        adapted = {
            'total_epochs_planned': results['total_epochs_planned'],
            'total_epochs_actual': results['total_epochs_actual'],
            'total_time_seconds': results['total_time_seconds'],
            'total_time_formatted': results['total_time_formatted'],
            'final_loss': results['final_loss'],
            'all_stages_converged': results['all_stages_converged'],
            'stage_metrics': {},
            'global_history': {'losses': [], 'stages': [], 'epochs': []},
            'stage_histories': {}
        }
        
        # Conversion des résultats par stage
        for stage_id, stage_result in results['stage_results'].items():
            adapted['stage_metrics'][stage_id] = {
                'stage': stage_id,
                'epochs_trained': stage_result.get('epochs_trained', 0),
                'final_loss': stage_result.get('final_loss', float('inf')),
                'convergence_met': stage_result.get('converged', False),
                'early_stopped': stage_result.get('converged', False),
                'loss_history': stage_result.get('loss_history', [])
            }
            
            # Reconstruction de l'historique global
            stage_losses = stage_result.get('loss_history', [])
            for i, loss in enumerate(stage_losses):
                adapted['global_history']['losses'].append(loss)
                adapted['global_history']['stages'].append(stage_id)
                adapted['global_history']['epochs'].append(len(adapted['global_history']['epochs']))
            
            # Historique par stage
            adapted['stage_histories'][stage_id] = {
                'losses': stage_losses,
                'epochs': list(range(len(stage_losses))),
                'lr': [0.001] * len(stage_losses)  # Placeholder pour LR
            }
        
        return adapted


# =============================================================================
# Fonction principale
# =============================================================================

def main():
    """Fonction principale avec architecture modulaire."""
    print(f"\n" + "="*80)
    print(f"🚀 NEURAL CELLULAR AUTOMATON - ARCHITECTURE MODULAIRE v9")
    print(f"="*80)

    try:
        # Parsing des arguments
        args = parse_arguments()
        
        # Configuration globale
        global_config = GlobalConfig(seed=args.seed)
        global_config.TOTAL_EPOCHS = args.total_epochs
        global_config.GRID_SIZE = args.grid_size
        global_config.BATCH_SIZE = args.batch_size
        global_config.SAVE_STAGE_CHECKPOINTS = args.save_checkpoints
        
        # Création du répertoire de sortie
        global_config.OUTPUT_DIR = f"{global_config.OUTPUT_DIR}_seed_{global_config.SEED}"
        if global_config.SAVE_ANIMATIONS:
            os.makedirs(global_config.OUTPUT_DIR, exist_ok=True)

        # Initialisation
        torch.manual_seed(global_config.SEED)
        np.random.seed(global_config.SEED)

        print(f"🎯 Configuration:")
        print(f"   Device: {global_config.DEVICE}")
        print(f"   Seed: {global_config.SEED}")
        print(f"   Époques totales: {global_config.TOTAL_EPOCHS}")
        print(f"   Répertoire: {global_config.OUTPUT_DIR}")

        # Initialisation du modèle
        model = ImprovedNCA(
            input_size=11,
            hidden_size=global_config.HIDDEN_SIZE,
            n_layers=global_config.N_LAYERS
        ).to(global_config.DEVICE)

        print(f"📊 Modèle: {sum(p.numel() for p in model.parameters()):,} paramètres")

        # Entraîneur modulaire
        trainer = ModularTrainer(model, global_config)
        
        # Affichage du curriculum configuré
        curriculum_summary = trainer.stage_manager.get_curriculum_summary()
        print(f"🔄 Curriculum: {curriculum_summary['stage_sequence']}")
        print(f"📊 Époques par stage: {curriculum_summary['epochs_per_stage']}")

        # Exécution du curriculum complet
        results = trainer.train_full_curriculum()

        # Génération des visualisations par stage
        print(f"\n🎨 Génération des visualisations par stage...")
        trainer.generate_stage_visualizations(args.vis_seed)

        # Génération de la suite complète de visualisations
        print(f"\n🎨 Génération de la suite complète de visualisations...")
        trainer.generate_complete_visualization_suite(results)

        # Sauvegarde finale
        final_model_path = Path(global_config.OUTPUT_DIR) / "final_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'results': results,
            'config': global_config.__dict__
        }, final_model_path)

        print(f"\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print(f"📁 Résultats: {global_config.OUTPUT_DIR}")
        print(f"⏱️  Temps: {results['total_time_formatted']}")
        print(f"🎯 Convergence: {'✅ TOUTES' if results['all_stages_converged'] else '❌ PARTIELLE'}")
        
        # Résumé des visualisations générées
        print(f"\n🎨 Visualisations générées:")
        print(f"   • Répertoires par stage: stage_1/, stage_2/, stage_3/, stage_4/")
        print(f"   • Animations comparatives par stage")
        print(f"   • Graphiques de convergence par stage")
        print(f"   • Suite complète de visualisations étendues")
        print(f"   • Métriques et checkpoints JSON")
        print(f"   • NOUVEAU : Visualisations Stage 4 avec intensités 0.25 et 0.5")

        return results

    except KeyboardInterrupt:
        print(f"\n⚠️  Interrompu par l'utilisateur")
        return None
    except Exception as e:
        print(f"\n❌ ERREUR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n🎯 Programme terminé avec succès!")
    else:
        print(f"\n❌ Programme terminé avec erreurs")
        exit(1)
