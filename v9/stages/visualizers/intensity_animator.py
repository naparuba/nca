"""
Animateur spécialisé pour les visualisations avec intensités variables.
Migré depuis visualize_modular_progressive_obstacles_variable_intensity.py
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List
from pathlib import Path


class IntensityAwareAnimator:
    """
    Générateur d'animations avec affichage d'intensité dans les titres.
    Spécialisé pour la version 8__ avec intensités variables.
    """
    
    def create_intensity_labeled_gif(self, sequence: List[np.ndarray],
                                   obstacle_mask: np.ndarray,
                                   source_intensity: float,
                                   filepath: Path,
                                   base_title: str):
        """
        Crée un GIF avec l'intensité affichée dans le titre.
        
        Args:
            sequence: Séquence d'images à animer
            obstacle_mask: Masque des obstacles à afficher
            source_intensity: Intensité de la source lumineuse
            filepath: Chemin de sortie du fichier GIF
            base_title: Titre de base pour l'animation
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate(frame):
            ax.clear()
            im = ax.imshow(sequence[frame], cmap='hot', vmin=0, vmax=1)
            ax.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            
            # Titre avec intensité
            title = f'{base_title} (I={source_intensity:.3f}) - t={frame}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            return [im]
        
        ani = animation.FuncAnimation(fig, animate, frames=len(sequence),
                                    interval=200, blit=False, repeat=True)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()
    
    def create_comparison_with_intensity(self, target_seq: List[np.ndarray],
                                       nca_seq: List[np.ndarray],
                                       obstacle_mask: np.ndarray,
                                       source_intensity: float,
                                       filepath: Path):
        """
        Crée une animation de comparaison avec affichage d'intensité.
        
        Args:
            target_seq: Séquence cible
            nca_seq: Séquence générée par le NCA
            obstacle_mask: Masque des obstacles
            source_intensity: Intensité de la source
            filepath: Chemin de sortie du fichier GIF
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Cible
            im1 = ax1.imshow(target_seq[frame], cmap='hot', vmin=0, vmax=1)
            ax1.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax1.set_title(f'Cible (I={source_intensity:.3f}) - t={frame}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # NCA
            im2 = ax2.imshow(nca_seq[frame], cmap='hot', vmin=0, vmax=1)
            ax2.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax2.set_title(f'NCA (I={source_intensity:.3f}) - t={frame}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            return [im1, im2]
        
        n_frames = min(len(target_seq), len(nca_seq))
        ani = animation.FuncAnimation(fig, animate, frames=n_frames,
                                    interval=200, blit=False, repeat=True)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()
