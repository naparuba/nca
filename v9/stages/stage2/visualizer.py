"""
Visualisations spécifiques pour le Stage 2 - Un obstacle.
Ce module contient les visualisations personnalisées pour le stage avec un obstacle unique.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple


class Stage2Visualizer:
    """Visualisations spécialisées pour le Stage 2 - un obstacle unique."""
    
    @staticmethod
    def create_visualizations(stage_dir: Path, target_seq: List[torch.Tensor],
                             nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                             source_mask: torch.Tensor, source_intensity: float,
                             vis_seed: int, intensity_history: None = None):
        """
        Crée les visualisations complètes pour le Stage 2.
        
        Args:
            stage_dir: Répertoire où sauvegarder les visualisations
            target_seq: Séquence cible
            nca_seq: Séquence prédite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source
            source_intensity: Intensité de la source
            vis_seed: Graine de visualisation
            intensity_history: Non utilisé pour Stage 2, inclus pour compatibilité
        """
        # Crée les animations standard
        Stage2Visualizer._create_animations(
            stage_dir, target_seq, nca_seq, obstacle_mask, source_intensity
        )
        
        # Crée les graphiques de convergence
        Stage2Visualizer._create_convergence_plot(
            stage_dir, target_seq, nca_seq, vis_seed
        )
        
        # Visualisations spécifiques au Stage 2
        Stage2Visualizer._create_obstacle_contournement_plot(
            stage_dir, nca_seq, obstacle_mask, source_mask
        )
        
        # Analyse du flux de diffusion
        Stage2Visualizer._create_flow_analysis(
            stage_dir, target_seq, nca_seq, obstacle_mask, source_mask
        )
    
    @staticmethod
    def _create_animations(stage_dir: Path, target_seq: List[torch.Tensor],
                          nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                          source_intensity: float):
        """Crée les animations GIF pour le Stage 2."""
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
        
        comparison_path = stage_dir / "animation_comparaison_stage_2.gif"
        ani.save(comparison_path, writer='pillow', fps=5)
        plt.close()
        
        # Animation NCA seule
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate_nca(frame):
            ax.clear()
            im = ax.imshow(nca_np[frame], cmap='hot', vmin=0, vmax=1)
            ax.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax.set_title(f'Stage 2 - NCA - t={frame}')
            ax.set_xticks([])
            ax.set_yticks([])
            return [im]
        
        ani_nca = animation.FuncAnimation(
            fig, animate_nca, frames=len(nca_np), interval=200, blit=False
        )
        
        nca_path = stage_dir / "animation_nca_stage_2.gif"
        ani_nca.save(nca_path, writer='pillow', fps=5)
        plt.close()
    
    @staticmethod
    def _create_convergence_plot(stage_dir: Path, target_seq: List[torch.Tensor],
                                nca_seq: List[torch.Tensor], vis_seed: int,
                                threshold: float = 0.0002):
        """Crée le graphique de convergence pour le Stage 2."""
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
                  label=f'Seuil convergence Stage 2')
        
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_title(f'Convergence Stage 2 - Seed {vis_seed}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = stage_dir / "convergence_stage_2.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_obstacle_contournement_plot(stage_dir: Path, nca_seq: List[torch.Tensor],
                                           obstacle_mask: torch.Tensor, source_mask: torch.Tensor):
        """
        Visualisation spécifique au Stage 2: contournement d'obstacle.
        Analyse comment la diffusion contourne l'obstacle.
        """
        # On prend quelques frames clés pour l'analyse
        frames = [0, len(nca_seq)//4, len(nca_seq)//2, len(nca_seq)-1]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()
        
        # Récupération des masques
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        source_pos = np.argwhere(source_mask.detach().cpu().numpy())
        
        # Calcul du centre de l'obstacle
        obstacle_cells = np.argwhere(obstacle_np)
        if len(obstacle_cells) > 0:
            obstacle_center = np.mean(obstacle_cells, axis=0)
        else:
            obstacle_center = np.array([nca_seq[0].shape[0]/2, nca_seq[0].shape[1]/2])
        
        for i, frame_idx in enumerate(frames):
            if frame_idx >= len(nca_seq):
                continue
                
            nca_frame = nca_seq[frame_idx].detach().cpu().numpy()
            
            # Affichage du frame
            im = axes[i].imshow(nca_frame, cmap='hot', vmin=0, vmax=1)
            axes[i].contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            
            # Annotation de l'obstacle et source
            if len(source_pos) > 0:
                axes[i].plot(source_pos[0,1], source_pos[0,0], 'wo', markersize=6)
            axes[i].plot(obstacle_center[1], obstacle_center[0], 'co', markersize=4)
            
            # Calcul des gradients pour visualiser le flux
            if frame_idx > 0:
                dy, dx = np.gradient(nca_frame)
                # Sous-échantillonnage pour la clarté
                step = 2
                y, x = np.mgrid[0:nca_frame.shape[0]:step, 0:nca_frame.shape[1]:step]
                skip = (slice(None, None, step), slice(None, None, step))
                axes[i].quiver(x, y, dx[skip], dy[skip], alpha=0.4, color='yellow',
                             angles='xy', scale_units='xy', scale=0.2, width=0.001)
            
            axes[i].set_title(f't={frame_idx}')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        plt.suptitle("Contournement de l'Obstacle - Stage 2")
        plt.tight_layout()
        
        # Sauvegarde
        contour_path = stage_dir / "contournement_obstacle_stage_2.png"
        plt.savefig(contour_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_flow_analysis(stage_dir: Path, target_seq: List[torch.Tensor],
                             nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                             source_mask: torch.Tensor):
        """
        Analyse du flux de diffusion autour de l'obstacle.
        Compare les flux entre la cible et la prédiction NCA.
        """
        # Utilisation de la dernière frame pour l'analyse
        target_final = target_seq[-1].detach().cpu().numpy()
        nca_final = nca_seq[-1].detach().cpu().numpy()
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        source_np = source_mask.detach().cpu().numpy()
        
        # Graphique d'analyse des chemins de diffusion
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gradients pour visualiser le flux
        dy_target, dx_target = np.gradient(target_final)
        dy_nca, dx_nca = np.gradient(nca_final)
        
        # Sous-échantillonnage pour la clarté
        step = 2
        y, x = np.mgrid[0:target_final.shape[0]:step, 0:target_final.shape[1]:step]
        skip = (slice(None, None, step), slice(None, None, step))
        
        # Cible
        im1 = ax1.imshow(target_final, cmap='hot', vmin=0, vmax=1)
        ax1.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
        ax1.quiver(x, y, dx_target[skip], dy_target[skip], alpha=0.6, color='yellow',
                 angles='xy', scale_units='xy', scale=0.3, width=0.002)
        ax1.set_title('Flux de Diffusion - Cible')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # NCA
        im2 = ax2.imshow(nca_final, cmap='hot', vmin=0, vmax=1)
        ax2.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
        ax2.quiver(x, y, dx_nca[skip], dy_nca[skip], alpha=0.6, color='yellow',
                 angles='xy', scale_units='xy', scale=0.3, width=0.002)
        ax2.set_title('Flux de Diffusion - NCA')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Coloration de la source
        for ax in (ax1, ax2):
            if np.any(source_np):
                source_y, source_x = np.where(source_np)
                for sy, sx in zip(source_y, source_x):
                    ax.plot(sx, sy, 'wo', markersize=6)
        
        plt.suptitle("Analyse des Flux de Diffusion - Stage 2")
        plt.tight_layout()
        
        # Sauvegarde
        flow_path = stage_dir / "analyse_flux_stage_2.png"
        plt.savefig(flow_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Analyse des profils de contournement
        Stage2Visualizer._create_contournement_profile(stage_dir, target_final, nca_final, obstacle_np, source_np)
    
    @staticmethod
    def _create_contournement_profile(stage_dir: Path, target_final: np.ndarray,
                                     nca_final: np.ndarray, obstacle_np: np.ndarray,
                                     source_np: np.ndarray):
        """
        Analyse des profils de contournement autour de l'obstacle.
        Compare les profils de diffusion autour de l'obstacle entre la cible et la prédiction NCA.
        """
        # Si aucun obstacle, on ne peut pas faire l'analyse
        if not np.any(obstacle_np):
            return
            
        # Calcul du centre de l'obstacle
        obstacle_cells = np.argwhere(obstacle_np)
        if len(obstacle_cells) > 0:
            obstacle_center = np.mean(obstacle_cells, axis=0)
        else:
            return
        
        # Calcul des coordonnées pour tracer des lignes depuis le centre de l'obstacle
        center_y, center_x = obstacle_center.astype(int)
        height, width = obstacle_np.shape
        
        # Calcul des profils dans plusieurs directions
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, angle in enumerate(angles):
            # Convertir l'angle en radians
            rad = np.deg2rad(angle)
            
            # Calculer la direction
            dx = np.cos(rad)
            dy = np.sin(rad)
            
            # Tracer une ligne depuis le centre de l'obstacle
            line_length = min(height, width) // 2
            points_y = []
            points_x = []
            values_target = []
            values_nca = []
            
            for dist in range(1, line_length):
                y = int(center_y + dist * dy)
                x = int(center_x + dist * dx)
                
                # Vérifier que le point est dans la grille
                if 0 <= y < height and 0 <= x < width:
                    if not obstacle_np[y, x]:  # Ignorer les points dans l'obstacle
                        points_y.append(y)
                        points_x.append(x)
                        values_target.append(target_final[y, x])
                        values_nca.append(nca_final[y, x])
            
            # Tracer le profil
            if points_y:
                distances = np.sqrt((np.array(points_y) - center_y)**2 +
                                  (np.array(points_x) - center_x)**2)
                
                axes[i].plot(distances, values_target, 'b-', linewidth=2, label='Cible')
                axes[i].plot(distances, values_nca, 'r--', linewidth=2, label='NCA')
                axes[i].set_title(f'Angle {angle}°')
                axes[i].set_xlabel('Distance depuis l\'obstacle')
                axes[i].set_ylabel('Intensité')
                axes[i].grid(True, alpha=0.3)
                
                if i == 0:
                    axes[i].legend()
        
        plt.suptitle("Profils de Contournement de l'Obstacle - Stage 2")
        plt.tight_layout()
        
        # Sauvegarde
        profile_path = stage_dir / "profils_contournement_stage_2.png"
        plt.savefig(profile_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Création d'une visualisation 2D du contournement
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Visualisation avec lignes de niveau
        im1 = ax1.imshow(target_final, cmap='hot', vmin=0, vmax=1)
        contour1 = ax1.contour(target_final, levels=np.linspace(0.1, 0.9, 5),
                             colors='white', alpha=0.7, linewidths=1)
        ax1.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
        ax1.set_title('Contournement - Cible')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        im2 = ax2.imshow(nca_final, cmap='hot', vmin=0, vmax=1)
        contour2 = ax2.contour(nca_final, levels=np.linspace(0.1, 0.9, 5),
                             colors='white', alpha=0.7, linewidths=1)
        ax2.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
        ax2.set_title('Contournement - NCA')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Affichage des sources
        if np.any(source_np):
            source_y, source_x = np.where(source_np)
            ax1.plot(source_x, source_y, 'wo', markersize=5)
            ax2.plot(source_x, source_y, 'wo', markersize=5)
        
        plt.suptitle("Lignes de Niveau du Contournement - Stage 2")
        plt.tight_layout()
        
        # Sauvegarde
        contour_path = stage_dir / "lignes_niveau_contournement_stage_2.png"
        plt.savefig(contour_path, dpi=150, bbox_inches='tight')
        plt.close()
