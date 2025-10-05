"""
Visualisations spécifiques pour le Stage 1 - Sans obstacles.
Ce module contient les visualisations personnalisées pour le stage de base.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple


class Stage1Visualizer:
    """Visualisations spécialisées pour le Stage 1 - sans obstacles."""
    
    @staticmethod
    def create_visualizations(stage_dir: Path, target_seq: List[torch.Tensor],
                             nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                             source_mask: torch.Tensor, source_intensity: float,
                             vis_seed: int, intensity_history: None = None):
        """
        Crée les visualisations complètes pour le Stage 1.
        
        Args:
            stage_dir: Répertoire où sauvegarder les visualisations
            target_seq: Séquence cible
            nca_seq: Séquence prédite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source
            source_intensity: Intensité de la source
            vis_seed: Graine de visualisation
            intensity_history: Non utilisé pour Stage 1, inclus pour compatibilité
        """
        # Crée les animations standard
        Stage1Visualizer._create_animations(
            stage_dir, target_seq, nca_seq, obstacle_mask, source_intensity
        )
        
        # Crée les graphiques de convergence
        Stage1Visualizer._create_convergence_plot(
            stage_dir, target_seq, nca_seq, vis_seed
        )
        
        # Visualisations spécifiques au Stage 1
        Stage1Visualizer._create_diffusion_pattern_plot(
            stage_dir, nca_seq, source_mask
        )
    
    @staticmethod
    def _create_animations(stage_dir: Path, target_seq: List[torch.Tensor],
                          nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                          source_intensity: float):
        """Crée les animations GIF pour le Stage 1."""
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
            ax1.set_title(f'Cible - t={frame}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # NCA
            im2 = ax2.imshow(nca_np[frame], cmap='hot', vmin=0, vmax=1)
            ax2.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax2.set_title(f'NCA - t={frame}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            return [im1, im2]
        
        n_frames = min(len(target_np), len(nca_np))
        ani = animation.FuncAnimation(
            fig, animate_comparison, frames=n_frames, interval=200, blit=False
        )
        
        comparison_path = stage_dir / "animation_comparaison_stage_1.gif"
        ani.save(comparison_path, writer='pillow', fps=5)
        plt.close()
        
        # Animation NCA seule
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate_nca(frame):
            ax.clear()
            im = ax.imshow(nca_np[frame], cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Stage 1 - NCA - t={frame}')
            ax.set_xticks([])
            ax.set_yticks([])
            return [im]
        
        ani_nca = animation.FuncAnimation(
            fig, animate_nca, frames=len(nca_np), interval=200, blit=False
        )
        
        nca_path = stage_dir / "animation_nca_stage_1.gif"
        ani_nca.save(nca_path, writer='pillow', fps=5)
        plt.close()
    
    @staticmethod
    def _create_convergence_plot(stage_dir: Path, target_seq: List[torch.Tensor],
                                nca_seq: List[torch.Tensor], vis_seed: int,
                                threshold: float = 0.0002):
        """Crée le graphique de convergence pour le Stage 1."""
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
        ax.axhline(y=threshold, color='r', linestyle='--',
                  label=f'Seuil convergence Stage 1')
        
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_title(f'Convergence Stage 1 - Seed {vis_seed}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = stage_dir / "convergence_stage_1.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_diffusion_pattern_plot(stage_dir: Path, nca_seq: List[torch.Tensor],
                                      source_mask: torch.Tensor):
        """
        Visualisation spécifique au Stage 1: motif de diffusion.
        Analyse comment la diffusion se propage depuis la source.
        """
        # On prend quelques frames clés pour l'analyse
        frames = [0, len(nca_seq)//4, len(nca_seq)//2, len(nca_seq)-1]
        fig, axes = plt.subplots(1, len(frames), figsize=(16, 4))
        
        for i, frame_idx in enumerate(frames):
            if frame_idx >= len(nca_seq):
                continue
                
            nca_frame = nca_seq[frame_idx].detach().cpu().numpy()
            source_pos = np.where(source_mask.detach().cpu().numpy())
            
            # Affichage du frame
            im = axes[i].imshow(nca_frame, cmap='hot', vmin=0, vmax=1)
            axes[i].plot(source_pos[1], source_pos[0], 'wo', markersize=4)
            axes[i].set_title(f't={frame_idx}')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        plt.suptitle("Progression de la Diffusion - Stage 1")
        plt.tight_layout()
        
        # Sauvegarde
        diffusion_path = stage_dir / "diffusion_pattern_stage_1.png"
        plt.savefig(diffusion_path, dpi=150, bbox_inches='tight')
        plt.close()
