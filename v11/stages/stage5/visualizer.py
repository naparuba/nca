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

    def _create_temporal_attenuation_plot(self, output_dir: Path, temporal_data: Dict[str, Any]) -> None:
        """
        Crée un graphique visualisant l'atténuation temporelle des intensités.
        
        Args:
            output_dir: Répertoire de sortie
            temporal_data: Données d'atténuation temporelle
        """
        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'sequence' in temporal_data:
            # Cas où nous avons une séquence complète
            sequence = temporal_data['sequence']
            steps = range(len(sequence))
            
            ax.plot(steps, sequence, 'r-', linewidth=3, label='Atténuation programmée')
            ax.set_title('Profil d\'atténuation temporelle')
            ax.set_xlabel('Pas de temps')
            ax.set_ylabel('Intensité de la source')
            
            # Informations supplémentaires
            if 'initial_intensity' in temporal_data and 'attenuation_rate' in temporal_data:
                initial = temporal_data['initial_intensity']
                rate = temporal_data['attenuation_rate']
                ax.text(0.02, 0.95, f"Intensité initiale: {initial:.3f}\nTaux d'atténuation: {rate:.4f}/pas",
                      transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        else:
            # Cas où nous devons reconstruire à partir des statistiques
            ax.text(0.5, 0.5, "Données d'atténuation temporelle non disponibles",
                  transform=ax.transAxes, ha='center', fontsize=12)
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Sauvegarde
        plt.tight_layout()
        plt.savefig(output_dir / 'attenuation_temporelle.png', dpi=150)
        plt.close()
    
    def _create_inferred_attenuation_plot(self, output_dir: Path,
                                      target_np: List[np.ndarray],
                                      nca_np: List[np.ndarray],
                                      source_np: np.ndarray) -> None:
        """
        Crée un graphique comparant l'atténuation d'intensité entre cible et prédiction.
        Infère l'atténuation à partir des données de la séquence.
        
        Args:
            output_dir: Répertoire de sortie
            target_np: Séquence cible en format numpy
            nca_np: Séquence NCA en format numpy
            source_np: Masque de la source en format numpy
        """
        # Trouver la position de la source
        source_i, source_j = np.where(source_np > 0.5)
        
        if len(source_i) == 0 or len(source_j) == 0:
            # Utiliser la position la plus chaude de la première frame comme approximation
            source_pos = np.unravel_index(np.argmax(target_np[0]), target_np[0].shape)
        else:
            source_pos = (source_i[0], source_j[0])
        
        # Extraire l'évolution de l'intensité à la position de la source
        target_intensities = [frame[source_pos] for frame in target_np]
        nca_intensities = [frame[source_pos] for frame in nca_np if frame.shape == target_np[0].shape]
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracer les courbes
        steps_target = range(len(target_intensities))
        ax.plot(steps_target, target_intensities, 'b-', linewidth=2.5, label='Cible (diffusion)')
        
        steps_nca = range(len(nca_intensities))
        ax.plot(steps_nca, nca_intensities, 'r--', linewidth=2, label='NCA (prédiction)')
        
        # Calcul et affichage du taux d'atténuation moyen
        if len(target_intensities) > 3:
            # Estimation linéaire du taux d'atténuation
            intensity_drops = np.diff(target_intensities)
            avg_attenuation_rate = -np.mean(intensity_drops)
            
            # Texte informatif
            ax.text(0.02, 0.95, f"Taux d'atténuation moyen (cible): {avg_attenuation_rate:.4f}/pas",
                  transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Configuration du graphique
        ax.set_title('Évolution de l\'intensité à la position de la source')
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Intensité')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(bottom=0)
        
        # Sauvegarde
        plt.tight_layout()
        plt.savefig(output_dir / 'comparaison_intensite_source.png', dpi=150)
        plt.close()
    
    def _create_intensity_profile_plot(self, output_dir: Path,
                                   target_np: List[np.ndarray],
                                   nca_np: List[np.ndarray],
                                   source_np: np.ndarray) -> None:
        """
        Crée un graphique montrant les profils d'intensité à différents moments.
        
        Args:
            output_dir: Répertoire de sortie
            target_np: Séquence cible en format numpy
            nca_np: Séquence NCA en format numpy
            source_np: Masque de la source en format numpy
        """
        # Sélection des moments clés (début, milieu et fin)
        if len(target_np) < 3 or len(nca_np) < 3:
            return  # Pas assez de données
            
        key_moments = [0, len(target_np) // 2, len(target_np) - 1]
        
        # Trouver la position de la source
        source_i, source_j = np.where(source_np > 0.5)
        
        if len(source_i) == 0 or len(source_j) == 0:
            # Utiliser la position la plus chaude de la première frame comme approximation
            source_pos = np.unravel_index(np.argmax(target_np[0]), target_np[0].shape)
        else:
            source_pos = (source_i[0], source_j[0])
        
        # Création du graphique à 3 sous-plots (un par moment)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        for i, t in enumerate(key_moments):
            ax = axes[i]
            
            # Extraction des lignes horizontale et verticale passant par la source
            h_slice_target = target_np[t][source_pos[0], :]
            v_slice_target = target_np[t][:, source_pos[1]]
            
            h_slice_nca = nca_np[t][source_pos[0], :] if t < len(nca_np) else np.zeros_like(h_slice_target)
            v_slice_nca = nca_np[t][:, source_pos[1]] if t < len(nca_np) else np.zeros_like(v_slice_target)
            
            # Tracer les profils horizontaux
            ax.plot(h_slice_target, 'b-', linewidth=2, label='Cible - horizontal')
            ax.plot(h_slice_nca, 'r--', linewidth=2, label='NCA - horizontal')
            
            # Tracer les profils verticaux
            ax.plot(v_slice_target, 'g-', linewidth=2, alpha=0.7, label='Cible - vertical')
            ax.plot(v_slice_nca, 'm--', linewidth=2, alpha=0.7, label='NCA - vertical')
            
            # Marquer la position de la source
            ax.axvline(x=source_pos[1], color='k', linestyle=':', alpha=0.5)
            ax.axvline(x=source_pos[0], color='k', linestyle='-.', alpha=0.5)
            
            # Configuration
            ax.set_title(f'Pas de temps {t}')
            ax.set_xlabel('Position')
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.set_ylabel('Intensité')
                ax.legend()
        
        # Titre global
        fig.suptitle('Profils d\'intensité à travers la source', fontsize=16)
        
        # Sauvegarde
        plt.tight_layout()
        plt.savefig(output_dir / 'profils_intensite.png', dpi=150)
        plt.close()
    
    def _create_advanced_attenuation_animation(self, output_dir: Path,
                                           target_np: List[np.ndarray],
                                           nca_np: List[np.ndarray],
                                           obstacles_np: np.ndarray,
                                           source_np: np.ndarray,
                                           initial_intensity: float) -> None:
        """
        Crée une animation avancée montrant l'atténuation temporelle avec graphique intégré.
        
        Args:
            output_dir: Répertoire de sortie
            target_np: Séquence cible en format numpy
            nca_np: Séquence NCA en format numpy
            obstacles_np: Masque des obstacles en format numpy
            source_np: Masque de la source en format numpy
            initial_intensity: Intensité initiale de la source
        """
        # Vérifier s'il y a suffisamment de données
        if len(target_np) < 5 or len(nca_np) < 5:
            return
        
        # Configuration de la figure
        fig = plt.figure(figsize=(16, 8))
        
        # Grilles pour les images
        gs = fig.add_gridspec(2, 3)
        
        # Création des axes
        ax_target = fig.add_subplot(gs[0, 0])  # Haut gauche - Cible
        ax_nca = fig.add_subplot(gs[0, 1])     # Haut milieu - NCA
        ax_diff = fig.add_subplot(gs[0, 2])    # Haut droite - Différence
        ax_plot = fig.add_subplot(gs[1, :])    # Bas - Graphique d'intensité
        
        # Trouver la position de la source
        source_i, source_j = np.where(source_np > 0.5)
        if len(source_i) == 0 or len(source_j) == 0:
            source_pos = np.unravel_index(np.argmax(target_np[0]), target_np[0].shape)
        else:
            source_pos = (source_i[0], source_j[0])
        
        # Initialisation
        im_target = ax_target.imshow(target_np[0], cmap=self.cmap, vmin=0, vmax=1)
        im_nca = ax_nca.imshow(nca_np[0], cmap=self.cmap, vmin=0, vmax=1)
        im_diff = ax_diff.imshow(np.abs(target_np[0] - nca_np[0]), cmap='viridis', vmin=0, vmax=0.5)
        
        # Afficher les obstacles
        obstacles_rgba = np.zeros((*obstacles_np.shape, 4))
        obstacles_rgba[obstacles_np > 0] = [0.5, 0.5, 0.5, 0.7]  # Gris semi-transparent
        ax_target.imshow(obstacles_rgba)
        ax_nca.imshow(obstacles_rgba)
        
        # Ajouter les marqueurs de source
        ax_target.plot(source_pos[1], source_pos[0], 'co', markersize=8)
        ax_nca.plot(source_pos[1], source_pos[0], 'co', markersize=8)
        
        # Configuration des axes
        ax_target.set_title('Cible (diffusion)')
        ax_nca.set_title('Prédiction (NCA)')
        ax_diff.set_title('Différence absolue')
        
        for ax in [ax_target, ax_nca, ax_diff]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Initialisation du graphique d'intensité
        target_intensities = [frame[source_pos] for frame in target_np]
        nca_intensities = [frame[source_pos] for frame in nca_np[:len(target_np)]]
        
        line_target, = ax_plot.plot([], [], 'b-', linewidth=2, label='Cible')
        line_nca, = ax_plot.plot([], [], 'r--', linewidth=2, label='NCA')
        
        # Point mobile sur les courbes
        point_target, = ax_plot.plot([], [], 'bo', markersize=8)
        point_nca, = ax_plot.plot([], [], 'ro', markersize=8)
        
        # Configuration du graphique
        ax_plot.set_xlim(0, len(target_np) - 1)
        ax_plot.set_ylim(0, initial_intensity * 1.1)
        ax_plot.set_xlabel('Pas de temps')
        ax_plot.set_ylabel('Intensité à la source')
        ax_plot.grid(True, alpha=0.3)
        ax_plot.legend(loc='upper right')
        
        # Texte pour l'étape actuelle
        time_text = fig.text(0.5, 0.01, '', ha='center', fontsize=12)
        
        # Colorbar
        cbar = fig.colorbar(im_target, ax=[ax_target, ax_nca, ax_diff], shrink=0.8)
        cbar.set_label('Intensité de chaleur')
        
        # Fonction d'animation
        def animate(i):
            # Mise à jour des images
            im_target.set_array(target_np[i])
            
            # Gérer les cas où nca_np est plus court que target_np
            if i < len(nca_np):
                im_nca.set_array(nca_np[i])
                im_diff.set_array(np.abs(target_np[i] - nca_np[i]))
            
            # Mise à jour des courbes
            line_target.set_data(range(i+1), target_intensities[:i+1])
            line_nca.set_data(range(min(i+1, len(nca_intensities))), nca_intensities[:i+1])
            
            # Mise à jour des points mobiles
            point_target.set_data([i], [target_intensities[i]])
            if i < len(nca_intensities):
                point_nca.set_data([i], [nca_intensities[i]])
            
            # Mise à jour du texte
            time_text.set_text(f'Pas de temps: {i}')
            
            return [im_target, im_nca, im_diff, line_target, line_nca, point_target, point_nca, time_text]
        
        # Création de l'animation
        ani = animation.FuncAnimation(
            fig, animate, frames=min(len(target_np), 50),  # Limite à 50 images pour des GIFs raisonnables
            interval=200, blit=True
        )
        
        # Sauvegarde
        ani.save(str(output_dir / 'animation_attenuation_avancee.gif'), writer='pillow', fps=5)
        plt.close(fig)

    def _finalize_convergence_plot(self, ax, output_dir: Path, seed: int, initial_intensity: float, suffix: str = ""):
        """
        Finalise et sauvegarde un graphique de convergence.
        
        Args:
            ax: Axe matplotlib à finaliser
            output_dir: Répertoire de sortie
            seed: Graine aléatoire pour reproductibilité
            initial_intensity: Intensité initiale de la source
            suffix: Suffixe pour le nom de fichier
        """
        # Configuration du graphique
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('MSE')
        ax.set_yscale('log')  # Échelle logarithmique pour mieux voir les détails
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Titre avec information d'intensité
        ax.set_title(f'Convergence - Intensité initiale: {initial_intensity:.2f} - Seed: {seed}')
        
        # Sauvegarde
        plt.tight_layout()
        plt.savefig(output_dir / f'convergence_temporelle{suffix}.png', dpi=150)
        plt.close()
