"""
Visualisations spécifiques pour le Stage 3 - Obstacles multiples.
Ce module contient les visualisations personnalisées pour le stage avec plusieurs obstacles.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


class Stage3Visualizer:
    """Visualisations spécialisées pour le Stage 3 - obstacles multiples."""
    
    @staticmethod
    def create_visualizations(stage_dir: Path, target_seq: List[torch.Tensor],
                             nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                             source_mask: torch.Tensor, source_intensity: float,
                             vis_seed: int, intensity_history: None = None):
        """
        Crée les visualisations complètes pour le Stage 3.
        
        Args:
            stage_dir: Répertoire où sauvegarder les visualisations
            target_seq: Séquence cible
            nca_seq: Séquence prédite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source
            source_intensity: Intensité de la source
            vis_seed: Graine de visualisation
            intensity_history: Non utilisé pour Stage 3, inclus pour compatibilité
        """
        # Crée les animations standard
        Stage3Visualizer._create_animations(
            stage_dir, target_seq, nca_seq, obstacle_mask, source_intensity
        )
        
        # Crée les graphiques de convergence
        Stage3Visualizer._create_convergence_plot(
            stage_dir, target_seq, nca_seq, vis_seed
        )
        
        # Visualisations spécifiques au Stage 3
        Stage3Visualizer._create_obstacle_analysis(
            stage_dir, nca_seq, obstacle_mask, source_mask
        )
        
        # Analyse de la propagation dans un environnement complexe
        Stage3Visualizer._create_complexity_analysis(
            stage_dir, target_seq, nca_seq, obstacle_mask, source_mask
        )
        
        # Carte de chaleur des différences
        Stage3Visualizer._create_difference_heatmap(
            stage_dir, target_seq[-1], nca_seq[-1], obstacle_mask
        )
    
    @staticmethod
    def _create_animations(stage_dir: Path, target_seq: List[torch.Tensor],
                          nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                          source_intensity: float):
        """Crée les animations GIF pour le Stage 3."""
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
        
        comparison_path = stage_dir / "animation_comparaison_stage_3.gif"
        ani.save(comparison_path, writer='pillow', fps=5)
        plt.close()
        
        # Animation NCA seule
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate_nca(frame):
            ax.clear()
            im = ax.imshow(nca_np[frame], cmap='hot', vmin=0, vmax=1)
            ax.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax.set_title(f'Stage 3 - NCA - t={frame}')
            ax.set_xticks([])
            ax.set_yticks([])
            return [im]
        
        ani_nca = animation.FuncAnimation(
            fig, animate_nca, frames=len(nca_np), interval=200, blit=False
        )
        
        nca_path = stage_dir / "animation_nca_stage_3.gif"
        ani_nca.save(nca_path, writer='pillow', fps=5)
        plt.close()
    
    @staticmethod
    def _create_convergence_plot(stage_dir: Path, target_seq: List[torch.Tensor],
                                nca_seq: List[torch.Tensor], vis_seed: int,
                                threshold: float = 0.0002):
        """Crée le graphique de convergence pour le Stage 3."""
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
                  label=f'Seuil convergence Stage 3')
        
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_title(f'Convergence Stage 3 - Seed {vis_seed}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = stage_dir / "convergence_stage_3.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_obstacle_analysis(stage_dir: Path, nca_seq: List[torch.Tensor],
                                 obstacle_mask: torch.Tensor, source_mask: torch.Tensor):
        """
        Visualisation spécifique au Stage 3: analyse des obstacles multiples.
        Analyse des zones de couverture et des obstacles.
        """
        # Utilisation de la dernière frame
        final_frame = nca_seq[-1].detach().cpu().numpy()
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        source_np = source_mask.detach().cpu().numpy()
        
        # Analyse de la structure des obstacles
        obstacle_labeled = Stage3Visualizer._label_obstacles(obstacle_np)
        n_obstacles = obstacle_labeled.max()
        
        # Segmentation en zones de diffusion
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Affichage de la diffusion finale
        im = ax.imshow(final_frame, cmap='hot', vmin=0, vmax=1, alpha=0.8)
        
        # Affichage et analyse des obstacles
        unique_obstacles = np.unique(obstacle_labeled)
        unique_obstacles = unique_obstacles[unique_obstacles > 0]  # Ignorer le fond
        
        # Palette de couleurs pour les obstacles
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_obstacles) + 1))
        
        for i, obstacle_id in enumerate(unique_obstacles):
            mask = obstacle_labeled == obstacle_id
            y, x = np.where(mask)
            if len(x) == 0 or len(y) == 0:
                continue
                
            # Calcul de la boîte englobante
            min_x, max_x = np.min(x), np.max(x)
            min_y, max_y = np.min(y), np.max(y)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            
            # Dessiner un rectangle autour de l'obstacle
            rect = Rectangle((min_x-0.5, min_y-0.5), width, height,
                           edgecolor=colors[i], facecolor='none',
                           linewidth=2, linestyle='--', alpha=0.7)
            ax.add_patch(rect)
            
            # Annoter l'obstacle
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2
            ax.text(center_x, center_y, f"{i+1}", color='white',
                   ha='center', va='center', fontweight='bold')
        
        # Calculer les zones de diffusion
        if np.any(source_np):
            source_y, source_x = np.where(source_np)
            ax.plot(source_x, source_y, 'wo', markersize=8, markeredgecolor='black')
            
            # Calcul des zones d'influence
            influence_zones = Stage3Visualizer._calculate_influence_zones(
                final_frame, obstacle_np
            )
            
            # Affichage des contours des zones d'influence
            contour = ax.contour(influence_zones, levels=[0.2, 0.5, 0.8],
                               colors=['yellow', 'orange', 'red'],
                               alpha=0.5, linewidths=1.5)
            ax.clabel(contour, inline=True, fontsize=8)
            
        ax.set_title(f'Analyse des Obstacles ({n_obstacles} obstacles)')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.colorbar(im, ax=ax, label='Intensité de diffusion')
        plt.tight_layout()
        
        # Sauvegarde
        obstacle_path = stage_dir / "analyse_obstacles_stage_3.png"
        plt.savefig(obstacle_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_complexity_analysis(stage_dir: Path, target_seq: List[torch.Tensor],
                                   nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                                   source_mask: torch.Tensor):
        """
        Analyse de la propagation dans un environnement complexe.
        Visualisation des chemins de diffusion avec des obstacles multiples.
        """
        # Selection de quelques frames clés
        n_frames = len(nca_seq)
        selected_frames = [n_frames//5, n_frames//3, n_frames//2, n_frames-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        source_np = source_mask.detach().cpu().numpy()
        
        # Création d'une colormap personnalisée pour visualiser les fronts de diffusion
        front_cmap = LinearSegmentedColormap.from_list('front_cmap',
                                                     ['blue', 'cyan', 'yellow', 'red'])
        
        for i, frame_idx in enumerate(selected_frames):
            if frame_idx >= n_frames:
                continue
            
            nca_frame = nca_seq[frame_idx].detach().cpu().numpy()
            
            # Calcul du gradient pour visualiser la direction du flux
            dy, dx = np.gradient(nca_frame)
            magnitude = np.sqrt(dx**2 + dy**2)
            
            # Visualisation avec masque d'obstacle
            im = axes[i].imshow(nca_frame, cmap='hot', vmin=0, vmax=1)
            
            # Contours pour visualiser les fronts de diffusion
            levels = np.linspace(0.1, 0.9, 5)
            contour = axes[i].contour(nca_frame, levels=levels, cmap=front_cmap,
                                    alpha=0.7, linewidths=1)
            
            # Obstacles et source
            axes[i].contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            
            if np.any(source_np):
                source_y, source_x = np.where(source_np)
                axes[i].plot(source_x, source_y, 'wo', markersize=6)
            
            # Visualisation du flux avec quiver
            step = 2
            y, x = np.mgrid[0:nca_frame.shape[0]:step, 0:nca_frame.shape[1]:step]
            skip = (slice(None, None, step), slice(None, None, step))
            
            # Ne dessiner que les flèches significatives
            mask = magnitude[skip] > 0.01
            axes[i].quiver(x[mask], y[mask], dx[skip][mask], dy[skip][mask],
                        alpha=0.6, color='yellow', angles='xy',
                        scale_units='xy', scale=0.2, width=0.002)
            
            axes[i].set_title(f't={frame_idx}')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        plt.suptitle("Analyse de Diffusion en Environnement Complexe - Stage 3")
        plt.tight_layout()
        
        # Sauvegarde
        complexity_path = stage_dir / "analyse_diffusion_complexe_stage_3.png"
        plt.savefig(complexity_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_difference_heatmap(stage_dir: Path, target_final: torch.Tensor,
                                 nca_final: torch.Tensor, obstacle_mask: torch.Tensor):
        """
        Crée une carte de chaleur des différences entre la cible et la prédiction NCA.
        Permet d'identifier les zones de difficulté.
        """
        # Conversion en numpy
        target_np = target_final.detach().cpu().numpy()
        nca_np = nca_final.detach().cpu().numpy()
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        
        # Calcul des différences
        diff = np.abs(target_np - nca_np)
        
        # Visualisation
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Target
        im1 = ax1.imshow(target_np, cmap='hot', vmin=0, vmax=1)
        ax1.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=1.5)
        ax1.set_title('Diffusion Cible')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # NCA
        im2 = ax2.imshow(nca_np, cmap='hot', vmin=0, vmax=1)
        ax2.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=1.5)
        ax2.set_title('Diffusion NCA')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Différence
        diff_cmap = plt.cm.RdBu_r
        im3 = ax3.imshow(diff, cmap=diff_cmap, vmin=0, vmax=0.2)
        ax3.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=1.5)
        ax3.set_title('Différence Absolue')
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        # Colorbar
        plt.colorbar(im3, ax=ax3, label='Différence')
        
        plt.suptitle("Analyse des Différences - Stage 3")
        plt.tight_layout()
        
        # Sauvegarde
        diff_path = stage_dir / "differences_cible_nca_stage_3.png"
        plt.savefig(diff_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _label_obstacles(obstacle_mask: np.ndarray) -> np.ndarray:
        """
        Étiquette les obstacles distincts dans le masque.
        
        Args:
            obstacle_mask: Masque binaire des obstacles
            
        Returns:
            Masque étiqueté où chaque obstacle a un identifiant unique
        """
        from scipy import ndimage
        
        # Étiquetage des composantes connexes
        labeled_mask, num_features = ndimage.label(obstacle_mask)
        return labeled_mask
    
    @staticmethod
    def _calculate_influence_zones(diffusion: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
        """
        Calcule les zones d'influence de diffusion en présence d'obstacles.
        
        Args:
            diffusion: Carte de diffusion finale
            obstacle_mask: Masque des obstacles
            
        Returns:
            Zones d'influence normalisées
        """
        # Création d'une carte de distance basée sur la diffusion
        influence = diffusion.copy()
        
        # Les obstacles n'ont pas d'influence
        influence[obstacle_mask] = 0
        
        # Normalisation pour bien visualiser les zones
        if influence.max() > 0:
            influence = influence / influence.max()
        
        return influence
