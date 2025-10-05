"""
Visualisations spécifiques pour le Stage 4 - Intensités variables.
Ce module contient les visualisations avancées pour le stage avec intensités variables.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class Stage4Visualizer:
    """Visualisations spécialisées pour le Stage 4 - intensités variables."""
    
    @staticmethod
    def create_visualizations(stage_dir: Path, target_seq: List[torch.Tensor],
                             nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                             source_mask: torch.Tensor, source_intensity: float,
                             vis_seed: int, intensity_history: Optional[List[float]] = None):
        """
        Crée les visualisations complètes pour le Stage 4 avec intensités variables.
        
        Args:
            stage_dir: Répertoire où sauvegarder les visualisations
            target_seq: Séquence cible
            nca_seq: Séquence prédite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source
            source_intensity: Intensité de la source
            vis_seed: Graine de visualisation
            intensity_history: Historique des intensités pendant l'entraînement
        """
        # Crée les animations standard avec indication de l'intensité
        Stage4Visualizer._create_animations_with_intensity(
            stage_dir, target_seq, nca_seq, obstacle_mask, source_intensity
        )
        
        # Crée les graphiques de convergence avec info d'intensité
        Stage4Visualizer._create_convergence_plot_with_intensity(
            stage_dir, target_seq, nca_seq, vis_seed, source_intensity
        )
        
        # Visualisations spécifiques au Stage 4
        if intensity_history:
            Stage4Visualizer._create_intensity_distribution_plot(
                stage_dir, intensity_history
            )
        
        # Graphique de l'influence de l'intensité sur la diffusion
        Stage4Visualizer._create_intensity_influence_plot(
            stage_dir, nca_seq, source_intensity
        )
    
    @staticmethod
    def create_multi_intensity_comparison(stage_dir: Path, intensities: List[float],
                                        vis_data_list: List[Dict[str, Any]],
                                        vis_seed: int):
        """
        Crée une visualisation comparative entre différentes intensités.
        
        Args:
            stage_dir: Répertoire où sauvegarder les visualisations
            intensities: Liste des intensités à comparer
            vis_data_list: Liste des données de visualisation pour chaque intensité
            vis_seed: Graine de visualisation
        """
        # Crée un graphique comparatif des profils de diffusion
        Stage4Visualizer._create_intensity_comparison_plot(
            stage_dir, intensities, vis_data_list, vis_seed
        )
        
        # Crée une animation comparative multi-intensités
        Stage4Visualizer._create_multi_intensity_animation(
            stage_dir, intensities, vis_data_list
        )
    
    @staticmethod
    def _create_animations_with_intensity(stage_dir: Path, target_seq: List[torch.Tensor],
                                        nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                                        source_intensity: float, suffix: str = ""):
        """Crée les animations GIF pour le Stage 4 avec indication d'intensité."""
        # Conversion en numpy
        target_np = [t.detach().cpu().numpy() for t in target_seq]
        nca_np = [t.detach().cpu().numpy() for t in nca_seq]
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        
        # Suffixe pour le nom de fichier
        file_suffix = f"_intensity_{source_intensity:.2f}".replace(".", "") if suffix == "" else suffix
        
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
        
        comparison_path = stage_dir / f"animation_comparaison_stage_4{file_suffix}.gif"
        ani.save(comparison_path, writer='pillow', fps=5)
        plt.close()
        
        # Animation NCA seule
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate_nca(frame):
            ax.clear()
            im = ax.imshow(nca_np[frame], cmap='hot', vmin=0, vmax=1)
            ax.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax.set_title(f'Stage 4 - NCA - t={frame} (I={source_intensity:.3f})')
            ax.set_xticks([])
            ax.set_yticks([])
            return [im]
        
        ani_nca = animation.FuncAnimation(
            fig, animate_nca, frames=len(nca_np), interval=200, blit=False
        )
        
        nca_path = stage_dir / f"animation_nca_stage_4{file_suffix}.gif"
        ani_nca.save(nca_path, writer='pillow', fps=5)
        plt.close()
    
    @staticmethod
    def _create_convergence_plot_with_intensity(stage_dir: Path, target_seq: List[torch.Tensor],
                                              nca_seq: List[torch.Tensor], vis_seed: int,
                                              source_intensity: float, threshold: float = 0.0002,
                                              suffix: str = ""):
        """Crée le graphique de convergence pour le Stage 4 avec information d'intensité."""
        # Calcul de l'erreur temporelle
        errors = []
        for t in range(min(len(target_seq), len(nca_seq))):
            target_np = target_seq[t].detach().cpu().numpy()
            nca_np = nca_seq[t].detach().cpu().numpy()
            error = np.mean((target_np - nca_np) ** 2)
            errors.append(error)
        
        # Suffixe pour le nom de fichier
        file_suffix = f"_intensity_{source_intensity:.2f}".replace(".", "") if suffix == "" else suffix
        
        # Graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(errors, 'b-', linewidth=2, label='Erreur MSE')
        ax.axhline(y=threshold, color='r', linestyle='--',
                  label=f'Seuil convergence Stage 4')
        
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_title(f'Convergence Stage 4 - Intensité {source_intensity:.3f} - Seed {vis_seed}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = stage_dir / f"convergence_stage_4{file_suffix}.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_intensity_distribution_plot(stage_dir: Path, intensity_history: List[float]):
        """
        Visualisation spécifique au Stage 4: distribution des intensités.
        Analyse la répartition des intensités utilisées pendant l'entraînement.
        """
        if not intensity_history:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogramme des intensités
        ax1.hist(intensity_history, bins=20, color='purple', alpha=0.7)
        ax1.set_xlabel('Intensité')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution des Intensités')
        ax1.grid(True, alpha=0.3)
        
        # Évolution temporelle des intensités
        ax2.plot(intensity_history, 'o-', markersize=2, alpha=0.5, color='purple')
        ax2.set_xlabel('Échantillon')
        ax2.set_ylabel('Intensité')
        ax2.set_title('Évolution des Intensités')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle("Analyse des Intensités Variables - Stage 4")
        plt.tight_layout()
        
        # Sauvegarde
        intensity_path = stage_dir / "intensity_distribution_stage_4.png"
        plt.savefig(intensity_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_intensity_influence_plot(stage_dir: Path, nca_seq: List[torch.Tensor],
                                       source_intensity: float):
        """
        Visualisation spécifique au Stage 4: influence de l'intensité.
        Analyse comment l'intensité affecte la propagation de la chaleur.
        """
        # On prend le dernier frame pour l'analyse
        final_frame = nca_seq[-1].detach().cpu().numpy()
        
        # Graphique du profil radial moyen
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculer le centre approximatif (supposons que c'est le point le plus chaud)
        max_pos = np.unravel_index(final_frame.argmax(), final_frame.shape)
        center_y, center_x = max_pos
        
        # Calculer les distances et valeurs radiales
        y_indices, x_indices = np.indices(final_frame.shape)
        r = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
        r_flat = r.flatten()
        v_flat = final_frame.flatten()
        
        # Calculer le profil radial moyen
        r_bins = np.linspace(0, np.max(r_flat), 20)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        mean_values = []
        
        for i in range(len(r_bins) - 1):
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
            if np.any(mask):
                mean_values.append(np.mean(v_flat[mask]))
            else:
                mean_values.append(0)
        
        # Tracer le profil radial moyen
        ax.plot(r_centers, mean_values, 'o-', linewidth=2)
        ax.set_xlabel('Distance du centre')
        ax.set_ylabel('Intensité moyenne')
        ax.set_title(f'Profil Radial Moyen - Intensité {source_intensity:.3f}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarde - Correction de la construction du chemin
        intensity_str = f"{source_intensity:.2f}".replace(".", "")
        profile_path = stage_dir / f"intensity_profile_stage_4_{intensity_str}.png"
        plt.savefig(profile_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_intensity_comparison_plot(stage_dir: Path, intensities: List[float],
                                        vis_data_list: List[Dict[str, Any]], vis_seed: int):
        """
        Crée un graphique comparant les profils de diffusion entre différentes intensités.
        
        Args:
            stage_dir: Répertoire où sauvegarder les visualisations
            intensities: Liste des intensités à comparer
            vis_data_list: Liste des données de visualisation pour chaque intensité
            vis_seed: Graine de visualisation
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(intensities)))
        
        for i, (intensity, vis_data) in enumerate(zip(intensities, vis_data_list)):
            # Utiliser la dernière frame de la séquence NCA
            nca_final = vis_data['nca_sequence'][-1]
            
            # Calculer le centre approximatif (supposons que c'est le point le plus chaud)
            max_pos = np.unravel_index(nca_final.argmax(), nca_final.shape)
            center_y, center_x = max_pos
            
            # Calculer les distances et valeurs radiales
            y_indices, x_indices = np.indices(nca_final.shape)
            r = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
            r_flat = r.flatten()
            v_flat = nca_final.flatten()
            
            # Calculer le profil radial moyen
            r_bins = np.linspace(0, np.max(r_flat), 20)
            r_centers = (r_bins[:-1] + r_bins[1:]) / 2
            mean_values = []
            
            for j in range(len(r_bins) - 1):
                mask = (r_flat >= r_bins[j]) & (r_flat < r_bins[j+1])
                if np.any(mask):
                    mean_values.append(np.mean(v_flat[mask]))
                else:
                    mean_values.append(0)
            
            # Tracer le profil radial moyen
            ax.plot(r_centers, mean_values, 'o-', linewidth=2, color=colors[i],
                   label=f'Intensité {intensity:.2f}')
        
        ax.set_xlabel('Distance du centre')
        ax.set_ylabel('Intensité moyenne')
        ax.set_title('Comparaison des Profils Radiaux par Intensité')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarde
        comparison_path = stage_dir / "intensity_comparison_profiles_stage_4.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _create_multi_intensity_animation(stage_dir: Path, intensities: List[float],
                                        vis_data_list: List[Dict[str, Any]]):
        """
        Crée une animation comparative des différentes intensités.
        
        Args:
            stage_dir: Répertoire où sauvegarder les visualisations
            intensities: Liste des intensités à comparer
            vis_data_list: Liste des données de visualisation pour chaque intensité
        """
        # Sélection de quelques frames clés
        num_frames = min([len(vis_data['nca_sequence']) for vis_data in vis_data_list])
        frames_to_show = [0, num_frames//3, 2*num_frames//3, num_frames-1]
        
        fig, axes = plt.subplots(len(intensities), len(frames_to_show), figsize=(16, 3*len(intensities)))
        
        for i, (intensity, vis_data) in enumerate(zip(intensities, vis_data_list)):
            nca_seq = vis_data['nca_sequence']
            obstacle_mask = vis_data['obstacle_mask']
            
            for j, frame_idx in enumerate(frames_to_show):
                ax = axes[i, j] if len(intensities) > 1 else axes[j]
                
                # Afficher le frame
                im = ax.imshow(nca_seq[frame_idx], cmap='hot', vmin=0, vmax=1)
                ax.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=1)
                
                if j == 0:
                    ax.set_ylabel(f'I={intensity:.2f}')
                
                if i == 0:
                    ax.set_title(f't={frame_idx}')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.suptitle("Progression de la Diffusion par Intensité - Stage 4")
        plt.tight_layout()
        
        # Sauvegarde
        multi_path = stage_dir / "multi_intensity_comparison_stage_4.png"
        plt.savefig(multi_path, dpi=150, bbox_inches='tight')
        plt.close()
