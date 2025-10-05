"""
Visualiseur spécialisé pour le Stage 5 (Atténuation Temporelle des Sources).
Génère des visualisations adaptées aux intensités décroissantes dans le temps.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from ..visualizers.base_visualizer import BaseVisualizer


class Stage5Visualizer(BaseVisualizer):
    """
    Visualiseur spécialisé pour le Stage 5 d'atténuation temporelle.
    Génère des visualisations adaptées à la décroissance temporelle des sources.
    """

    def __init__(self):
        super().__init__()
        # Palette de couleurs chaude pour visualiser la chaleur
        self.cmap = LinearSegmentedColormap.from_list('heat',
            [(0, '#000000'), (0.2, '#3b0f0f'), (0.4, '#8b0000'),
             (0.6, '#ff3800'), (0.8, '#ffa500'), (1, '#ffff00')], N=256)
        
        # Configuration matplotlib pour les animations
        matplotlib.rcParams['animation.embed_limit'] = 50  # Pour les grandes animations

    def create_visualizations(self, output_dir: Path,
                             target_sequence: List[torch.Tensor],
                             nca_sequence: List[torch.Tensor],
                             obstacle_mask: torch.Tensor,
                             source_mask: Union[torch.Tensor, float],
                             initial_intensity: float,
                             seed: int,
                             temporal_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Crée les visualisations spécialisées pour le Stage 5.
        
        Args:
            output_dir: Répertoire de sortie
            target_sequence: Séquence cible de diffusion
            nca_sequence: Séquence prédite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source (peut être un tensor ou un float)
            initial_intensity: Intensité initiale de la source
            seed: Graine aléatoire pour reproductibilité
            temporal_data: Données d'atténuation temporelle (optionnel)
        """
        # Convertir les tenseurs en arrays NumPy
        target_np = [t.detach().cpu().numpy() for t in target_sequence]
        nca_np = [n.detach().cpu().numpy() for n in nca_sequence]
        obstacles_np = obstacle_mask.detach().cpu().numpy()
        
        # Vérifier si source_mask est un tensor ou un float
        if isinstance(source_mask, torch.Tensor):
            source_np = source_mask.detach().cpu().numpy()
        else:
            print(f"⚠️ source_mask n'est pas un tensor mais un {type(source_mask).__name__}")
            # Créer un masque basé sur la position la plus chaude dans la première frame
            source_np = np.zeros_like(obstacles_np)
            if len(target_np) > 0:
                # Trouver la position la plus chaude dans la première frame
                max_pos = np.unravel_index(np.argmax(target_np[0]), target_np[0].shape)
                source_np[max_pos] = 1.0  # Marquer cette position comme source
        
        # Maintenant, exécuter chaque méthode de visualisation individuellement
        
        # 1. Créer les animations de base
        self._create_animations_with_intensity(output_dir, target_sequence, nca_sequence,
                                           obstacle_mask, initial_intensity)
        
        # 2. Créer le graphique de convergence
        self._create_convergence_plot_with_intensity(output_dir, target_sequence, nca_sequence,
                                                 seed, initial_intensity)
        
        # 3. Créer la visualisation spéciale d'atténuation temporelle
        if temporal_data:
            self._create_temporal_attenuation_plot(output_dir, temporal_data)
        else:
            self._create_inferred_attenuation_plot(output_dir, target_np, nca_np, source_np)
        
        # 4. Créer la visualisation des profils d'intensité
        self._create_intensity_profile_plot(output_dir, target_np, nca_np, source_np)
        
        # 5. Créer l'animation avancée d'atténuation
        self._create_advanced_attenuation_animation(output_dir, target_np, nca_np,
                                                obstacles_np, source_np, initial_intensity)

    def _create_animations_with_intensity(self, output_dir: Path,
                                       target_sequence: List[torch.Tensor],
                                       nca_sequence: List[torch.Tensor],
                                       obstacle_mask: torch.Tensor,
                                       initial_intensity: float,
                                       suffix: str = "") -> None:
        """
        Crée les animations GIF pour cible et prédiction avec information d'intensité.
        
        Args:
            output_dir: Répertoire de sortie
            target_sequence: Séquence cible de diffusion
            nca_sequence: Séquence prédite par le NCA
            obstacle_mask: Masque des obstacles
            initial_intensity: Intensité initiale de la source
            suffix: Suffixe pour le nom de fichier
        """
        # Convertir pour matplotlib
        target_np = [t.detach().cpu().numpy() for t in target_sequence]
        nca_np = [n.detach().cpu().numpy() for n in nca_sequence]
        obstacles_np = obstacle_mask.detach().cpu().numpy()
        
        # Configuration de l'animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Atténuation Temporelle - Intensité initiale: {initial_intensity:.2f}', fontsize=14)
        
        # Initialisation des images
        im1 = ax1.imshow(target_np[0], cmap=self.cmap, vmin=0, vmax=1)
        im2 = ax2.imshow(nca_np[0], cmap=self.cmap, vmin=0, vmax=1)
        
        # Affichage des obstacles
        obstacles_rgba = np.zeros((*obstacles_np.shape, 4))
        obstacles_rgba[obstacles_np > 0] = [0.5, 0.5, 0.5, 0.7]  # Gris semi-transparent
        ax1.imshow(obstacles_rgba)
        ax2.imshow(obstacles_rgba)
        
        # Titres et légende
        ax1.set_title('Cible (diffusion physique)')
        ax2.set_title('Prédiction (NCA)')
        ax1.axis('off')
        ax2.axis('off')
        
        # Texte pour l'intensité actuelle
        intensity_text = fig.text(0.5, 0.02, f'Intensité: {initial_intensity:.2f}',
                                ha='center', fontsize=12)
        
        # Fonction de mise à jour
        def update(frame):
            im1.set_array(target_np[frame])
            im2.set_array(nca_np[frame])
            
            # Estimer l'intensité de la source (approximation)
            if frame < len(target_np):
                # Trouver la position de la source et son intensité
                source_pos = np.unravel_index(np.argmax(target_np[0]), target_np[0].shape)
                current_intensity = target_np[frame][source_pos]
                intensity_text.set_text(f'Intensité actuelle: {current_intensity:.2f}')
            
            return [im1, im2, intensity_text]
        
        # Création de l'animation
        ani = animation.FuncAnimation(fig, update, frames=min(len(target_np), len(nca_np)),
                                    interval=150, blit=True)
        
        # Sauvegarde
        ani.save(str(output_dir / f'animation_attenuation_temporelle{suffix}.gif'),
                writer='pillow', fps=10)
        
        plt.close(fig)

    def _create_convergence_plot_with_intensity(self, output_dir: Path,
                                             target_sequence: List[torch.Tensor],
                                             nca_sequence: List[torch.Tensor],
                                             seed: int, initial_intensity: float,
                                             threshold: float = 0.002,
                                             suffix: str = "") -> None:
        """
        Crée un graphique de convergence montrant l'erreur MSE au fil du temps.
        
        Args:
            output_dir: Répertoire de sortie
            target_sequence: Séquence cible de diffusion
            nca_sequence: Séquence prédite par le NCA
            seed: Graine aléatoire pour reproductibilité
            initial_intensity: Intensité initiale de la source
            threshold: Seuil de convergence
            suffix: Suffixe pour le nom de fichier
        """
        # Convertir pour calcul
        target_np = [t.detach().cpu().numpy() for t in target_sequence]
        nca_np = [n.detach().cpu().numpy() for n in nca_sequence]
        
        # Calcul de l'erreur MSE au fil du temps
        mse_values = []
        timesteps = min(len(target_np), len(nca_np))
        
        for t in range(timesteps):
            mse = np.mean((target_np[t] - nca_np[t])**2)
            mse_values.append(mse)
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(timesteps), mse_values, 'b-', linewidth=2, label='MSE')
        
        # Ligne de seuil
        ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.7,
                 label=f'Seuil de convergence ({threshold})')
        
        # Formatage
        ax.set_title(f'Convergence - Atténuation temporelle (Intensité initiale: {initial_intensity:.2f})')
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Ajout d'information sur la graine aléatoire
        ax.text(0.02, 0.02, f'Seed: {seed}', transform=ax.transAxes,
              fontsize=8, color='gray', alpha=0.7)
        
        # Sauvegarde
        fig.savefig(str(output_dir / f'convergence_attenuation{suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _create_temporal_attenuation_plot(self, output_dir: Path,
                                       temporal_data: Dict[str, Any]) -> None:
        """
        Crée un graphique spécialisé pour visualiser l'atténuation temporelle.
        
        Args:
            output_dir: Répertoire de sortie
            temporal_data: Données d'atténuation temporelle
        """
        # Extraction des données
        sequences = temporal_data.get('sequences', [])
        if not sequences:
            return
        
        # Sélection de quelques séquences représentatives
        sequence_ids = sorted(sequences.keys())
        if len(sequence_ids) > 5:
            # Sélectionner 5 séquences bien réparties
            sequence_ids = sequence_ids[::max(1, len(sequence_ids) // 5)][:5]
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for seq_id in sequence_ids:
            seq_data = sequences[seq_id]
            initial = seq_data.get('initial_intensity', 0)
            rate = seq_data.get('attenuation_rate', 0)
            values = seq_data.get('sequence', [])
            
            if values:
                ax.plot(range(len(values)), values,
                      label=f'Initial: {initial:.2f}, Rate: {rate:.4f}')
        
        # Formatage
        ax.set_title('Profils d\'Atténuation Temporelle')
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Intensité de la source')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend()
        
        # Sauvegarde
        fig.savefig(str(output_dir / 'temporal_attenuation_profiles.png'),
                  dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _create_inferred_attenuation_plot(self, output_dir: Path,
                                       target_np: List[np.ndarray],
                                       nca_np: List[np.ndarray],
                                       source_np: np.ndarray) -> None:
        """
        Crée un graphique d'atténuation à partir des données inférées.
        
        Args:
            output_dir: Répertoire de sortie
            target_np: Séquence cible en NumPy
            nca_np: Séquence NCA en NumPy
            source_np: Masque de la source en NumPy
        """
        # Trouver la position de la source
        source_pos = np.unravel_index(np.argmax(source_np), source_np.shape)
        
        # Extraire l'intensité de la source au fil du temps
        target_intensities = [t[source_pos] for t in target_np]
        nca_intensities = [n[source_pos] for n in nca_np]
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(range(len(target_intensities)), target_intensities, 'b-',
              linewidth=2, label='Cible (physique)')
        ax.plot(range(len(nca_intensities)), nca_intensities, 'r-',
              linewidth=2, alpha=0.8, label='Prédiction (NCA)')
        
        # Calcul du taux d'atténuation (approximation linéaire)
        if len(target_intensities) > 5:
            initial = target_intensities[0]
            final = target_intensities[-1]
            steps = len(target_intensities) - 1
            avg_rate = (initial - final) / steps if steps > 0 else 0
            
            # Ligne théorique
            theoretical = [max(0, initial - avg_rate * t) for t in range(len(target_intensities))]
            ax.plot(range(len(theoretical)), theoretical, 'g--',
                  linewidth=1.5, alpha=0.6, label=f'Théorique (rate={avg_rate:.4f})')
        
        # Formatage
        ax.set_title('Atténuation Temporelle de la Source')
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Intensité de la source')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.legend()
        
        # Sauvegarde
        fig.savefig(str(output_dir / 'inferred_attenuation.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _create_intensity_profile_plot(self, output_dir: Path,
                                     target_np: List[np.ndarray],
                                     nca_np: List[np.ndarray],
                                     source_np: np.ndarray) -> None:
        """
        Crée un graphique montrant le profil d'intensité le long d'une ligne.
        
        Args:
            output_dir: Répertoire de sortie
            target_np: Séquence cible en NumPy
            nca_np: Séquence NCA en NumPy
            source_np: Masque de la source en NumPy
        """
        # Trouver la position de la source
        source_i, source_j = np.unravel_index(np.argmax(source_np), source_np.shape)
        grid_size = target_np[0].shape[0]
        
        # Sélection des pas de temps à visualiser
        timesteps = min(len(target_np), len(nca_np))
        selected_times = [0, timesteps // 4, timesteps // 2, 3 * timesteps // 4, timesteps - 1]
        selected_times = sorted(list(set(selected_times)))
        
        # Création du graphique
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Profil horizontal
        ax1 = axes[0]
        for t in selected_times:
            if t < timesteps:
                ax1.plot(range(grid_size), target_np[t][source_i, :],
                      label=f'Cible t={t}', linestyle='-')
                ax1.plot(range(grid_size), nca_np[t][source_i, :],
                      label=f'NCA t={t}', linestyle='--')
        
        ax1.axvline(x=source_j, color='k', linestyle=':', alpha=0.5, label='Position source')
        ax1.set_title('Profil d\'intensité horizontal')
        ax1.set_xlabel('Position X')
        ax1.set_ylabel('Intensité')
        ax1.grid(True, alpha=0.3)
        
        # Profil vertical
        ax2 = axes[1]
        for t in selected_times:
            if t < timesteps:
                ax2.plot(range(grid_size), target_np[t][:, source_j],
                      label=f'Cible t={t}', linestyle='-')
                ax2.plot(range(grid_size), nca_np[t][:, source_j],
                      label=f'NCA t={t}', linestyle='--')
        
        ax2.axvline(x=source_i, color='k', linestyle=':', alpha=0.5, label='Position source')
        ax2.set_title('Profil d\'intensité vertical')
        ax2.set_xlabel('Position Y')
        ax2.set_ylabel('Intensité')
        ax2.grid(True, alpha=0.3)
        
        # Légende commune
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))
        
        # Ajustements
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Sauvegarde
        fig.savefig(str(output_dir / 'intensity_profiles.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _create_advanced_attenuation_animation(self, output_dir: Path,
                                            target_np: List[np.ndarray],
                                            nca_np: List[np.ndarray],
                                            obstacles_np: np.ndarray,
                                            source_np: np.ndarray,
                                            initial_intensity: float) -> None:
        """
        Crée une animation avancée avec indicateur d'atténuation.
        
        Args:
            output_dir: Répertoire de sortie
            target_np: Séquence cible en NumPy
            nca_np: Séquence NCA en NumPy
            obstacles_np: Masque des obstacles en NumPy
            source_np: Masque de la source en NumPy
            initial_intensity: Intensité initiale de la source
        """
        # Trouver la position de la source
        source_pos = np.unravel_index(np.argmax(source_np), source_np.shape)
        
        # Extraire l'intensité de la source au fil du temps
        target_intensities = [t[source_pos] for t in target_np]
        nca_intensities = [n[source_pos] for n in nca_np]
        timesteps = min(len(target_np), len(nca_np))
        
        # Configuration de l'animation
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], width_ratios=[1, 1])
        
        ax_target = fig.add_subplot(gs[0, 0])
        ax_nca = fig.add_subplot(gs[0, 1])
        ax_graph = fig.add_subplot(gs[1, :])
        
        fig.suptitle(f'Atténuation Temporelle - Intensité initiale: {initial_intensity:.2f}',
                   fontsize=14)
        
        # Initialisation des images
        im_target = ax_target.imshow(target_np[0], cmap=self.cmap, vmin=0, vmax=1)
        im_nca = ax_nca.imshow(nca_np[0], cmap=self.cmap, vmin=0, vmax=1)
        
        # Affichage des obstacles
        obstacles_rgba = np.zeros((*obstacles_np.shape, 4))
        obstacles_rgba[obstacles_np > 0] = [0.5, 0.5, 0.5, 0.7]  # Gris semi-transparent
        ax_target.imshow(obstacles_rgba)
        ax_nca.imshow(obstacles_rgba)
        
        # Titres et légende
        ax_target.set_title('Cible (diffusion physique)')
        ax_nca.set_title('Prédiction (NCA)')
        ax_target.axis('off')
        ax_nca.axis('off')
        
        # Initialisation du graphique d'atténuation
        line_target, = ax_graph.plot([], [], 'b-', linewidth=2, label='Cible')
        line_nca, = ax_graph.plot([], [], 'r-', linewidth=2, alpha=0.8, label='NCA')
        line_marker_target, = ax_graph.plot([], [], 'bo', markersize=8)
        line_marker_nca, = ax_graph.plot([], [], 'ro', markersize=8)
        
        # Formatage du graphique
        ax_graph.set_xlim(0, timesteps - 1)
        ax_graph.set_ylim(0, initial_intensity * 1.1)
        ax_graph.set_xlabel('Pas de temps')
        ax_graph.set_ylabel('Intensité')
        ax_graph.grid(True, alpha=0.3)
        ax_graph.legend()
        
        # Fonction de mise à jour
        def update(frame):
            # Mise à jour des grilles
            im_target.set_array(target_np[frame])
            im_nca.set_array(nca_np[frame])
            
            # Mise à jour des données du graphique
            line_target.set_data(range(frame + 1), target_intensities[:frame + 1])
            line_nca.set_data(range(frame + 1), nca_intensities[:frame + 1])
            
            # Mise à jour des marqueurs
            if frame < timesteps:
                line_marker_target.set_data([frame], [target_intensities[frame]])
                line_marker_nca.set_data([frame], [nca_intensities[frame]])
            
            return [im_target, im_nca, line_target, line_nca, line_marker_target, line_marker_nca]
        
        # Création de l'animation
        ani = animation.FuncAnimation(fig, update, frames=timesteps,
                                    interval=150, blit=True)
        
        # Sauvegarde
        ani.save(str(output_dir / 'advanced_attenuation_animation.gif'),
                writer='pillow', fps=10)
        
        plt.close(fig)
