"""
Stage 6 Visualizer : Visualisation spécialisée pour les sources multiples.
Permet de visualiser les interactions entre sources avec caractéristiques indépendantes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import matplotlib.cm as cm
from matplotlib.colors import Normalize

class Stage6Visualizer:
    """
    Visualiseur spécialisé pour le Stage 6.
    Crée des visualisations adaptées aux sources multiples avec caractéristiques indépendantes.
    """
    
    def __init__(self):
        # Configuration de matplotlib
        plt.style.use('dark_background')
        self.cmap_main = 'hot'
        self.cmap_sources = plt.cm.viridis
        self.norm_sources = Normalize(vmin=0, vmax=1)
        
    def create_visualizations(self, output_dir: Path, target_seq: List[torch.Tensor],
                            nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                            source_mask: torch.Tensor, source_info: Any,
                            seed: int, additional_data: Any = None) -> None:
        """
        Crée l'ensemble des visualisations pour le Stage 6.
        
        Args:
            output_dir: Répertoire de sortie
            target_seq: Séquence cible
            nca_seq: Séquence produite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque des sources (obsolète pour Stage 6, sources multiples)
            source_info: Informations sur les sources ou intensité initiale
            seed: Graine aléatoire utilisée
            additional_data: Données supplémentaires (positions et intensités des sources)
        """
        print(f"    🎨 Création des visualisations Stage 6...")
        
        # Récupération des sources depuis les données additionnelles
        sources_data = self._extract_sources_data(target_seq, additional_data)
        if not sources_data:
            print(f"    ⚠️ Aucune information sur les sources multiples disponible, utilisation de visualisations génériques")
            return self._create_generic_visualizations(output_dir, target_seq, nca_seq, obstacle_mask, source_mask, seed)
        
        # Création des visualisations spécialisées
        self._create_multi_source_animation(output_dir, target_seq, nca_seq, obstacle_mask, sources_data)
        self._create_multi_source_convergence_plot(output_dir, target_seq, nca_seq, seed, sources_data)
        self._create_source_intensity_plots(output_dir, sources_data)
        self._create_source_interaction_visualization(output_dir, target_seq, nca_seq, obstacle_mask, sources_data)
        
        print(f"    ✅ Visualisations Stage 6 terminées")
        
    def _extract_sources_data(self, target_seq: List[torch.Tensor],
                            additional_data: Any) -> Dict[str, Any]:
        """
        Extrait les informations sur les sources à partir des données additionnelles.
        
        Args:
            target_seq: Séquence cible
            additional_data: Données supplémentaires
            
        Returns:
            Dictionnaire contenant les positions et caractéristiques des sources
        """
        if additional_data is None or not isinstance(additional_data, dict):
            return None
        
        # Structure attendue des données additionnelles
        expected_keys = ['positions', 'initial_intensities', 'attenuation_rates', 'intensities_over_time']
        for key in expected_keys:
            if key not in additional_data:
                return None
                
        # Vérification de la cohérence
        positions = additional_data['positions']
        if not positions or len(positions) == 0:
            return None
            
        # Construction du dictionnaire de retour
        sources_data = {
            'n_sources': len(positions),
            'positions': positions,
            'initial_intensities': additional_data['initial_intensities'],
            'attenuation_rates': additional_data['attenuation_rates'],
            'intensities_over_time': additional_data['intensities_over_time'],
            'n_steps': len(target_seq),
            'grid_size': target_seq[0].shape[0]
        }
        
        return sources_data
    
    def _create_generic_visualizations(self, output_dir: Path, target_seq: List[torch.Tensor],
                                     nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                                     source_mask: torch.Tensor, seed: int) -> None:
        """
        Crée des visualisations génériques lorsque les données de sources multiples ne sont pas disponibles.
        
        Args:
            output_dir: Répertoire de sortie
            target_seq: Séquence cible
            nca_seq: Séquence produite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source
            seed: Graine aléatoire
        """
        self._create_standard_animation(output_dir, target_seq, nca_seq, obstacle_mask, source_mask)
        self._create_standard_convergence_plot(output_dir, target_seq, nca_seq, seed)
    
    def _create_multi_source_animation(self, output_dir: Path, target_seq: List[torch.Tensor],
                                     nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                                     sources_data: Dict[str, Any]) -> None:
        """
        Crée une animation comparative montrant les sources multiples et leur évolution.
        
        Args:
            output_dir: Répertoire de sortie
            target_seq: Séquence cible
            nca_seq: Séquence produite par le NCA
            obstacle_mask: Masque des obstacles
            sources_data: Données des sources
        """
        # Conversion en numpy
        target_np = [t.detach().cpu().numpy() for t in target_seq]
        nca_np = [n.detach().cpu().numpy() for n in nca_seq]
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        
        # Récupération des informations sur les sources
        positions = sources_data['positions']
        intensities_over_time = sources_data['intensities_over_time']
        n_sources = sources_data['n_sources']
        
        # Création des couleurs distinctes pour chaque source
        source_colors = plt.cm.tab10(np.linspace(0, 1, n_sources))
        
        # Animation comparative
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Stage 6: Sources Multiples avec Caractéristiques Indépendantes ({n_sources} sources)', fontsize=14)
        
        def animate_comparison(frame):
            ax1.clear()
            ax2.clear()
            
            # Cible
            im1 = ax1.imshow(target_np[frame], cmap=self.cmap_main, vmin=0, vmax=1)
            ax1.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            
            # Marquage des sources avec couleurs distinctes
            for i, (pos, intensities) in enumerate(zip(positions, intensities_over_time)):
                if frame < len(intensities):
                    intensity = intensities[frame]
                    if intensity > 0.01:  # Ne pas afficher les sources éteintes
                        circle = plt.Circle((pos[1], pos[0]), 0.5, color=source_colors[i],
                                          alpha=min(0.8, intensity))
                        ax1.add_artist(circle)
                        ax1.text(pos[1], pos[0], f"{i+1}", ha='center', va='center',
                               color='white', fontsize=8, fontweight='bold')
            
            ax1.set_title(f'Cible - t={frame}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # NCA
            im2 = ax2.imshow(nca_np[frame], cmap=self.cmap_main, vmin=0, vmax=1)
            ax2.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            
            # Marquage des sources identique
            for i, (pos, intensities) in enumerate(zip(positions, intensities_over_time)):
                if frame < len(intensities):
                    intensity = intensities[frame]
                    if intensity > 0.01:
                        circle = plt.Circle((pos[1], pos[0]), 0.5, color=source_colors[i],
                                          alpha=min(0.8, intensity))
                        ax2.add_artist(circle)
                        ax2.text(pos[1], pos[0], f"{i+1}", ha='center', va='center',
                               color='white', fontsize=8, fontweight='bold')
            
            ax2.set_title(f'NCA - t={frame}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            # Légende des sources
            if frame == 0:
                for i in range(n_sources):
                    initial = sources_data['initial_intensities'][i]
                    rate = sources_data['attenuation_rates'][i]
                    fig.text(0.02 + i*0.25, 0.02,
                           f"Source {i+1}: I₀={initial:.2f}, τ={rate:.4f}",
                           color=source_colors[i], fontsize=9, transform=fig.transFigure)
            
            return [im1, im2]
        
        n_frames = min(len(target_np), len(nca_np))
        ani = animation.FuncAnimation(
            fig, animate_comparison, frames=n_frames, interval=200, blit=False
        )
        
        # Sauvegarde de l'animation
        animation_path = output_dir / "animation_multi_sources.gif"
        ani.save(animation_path, writer='pillow', fps=5, dpi=100)
        plt.close()
        
    def _create_multi_source_convergence_plot(self, output_dir: Path, target_seq: List[torch.Tensor],
                                           nca_seq: List[torch.Tensor], seed: int,
                                           sources_data: Dict[str, Any]) -> None:
        """
        Crée un graphique de convergence adapté aux sources multiples.
        
        Args:
            output_dir: Répertoire de sortie
            target_seq: Séquence cible
            nca_seq: Séquence produite par le NCA
            seed: Graine aléatoire
            sources_data: Données des sources
        """
        # Calcul de l'erreur temporelle
        errors = []
        for t in range(min(len(target_seq), len(nca_seq))):
            target_np = target_seq[t].detach().cpu().numpy()
            nca_np = nca_seq[t].detach().cpu().numpy()
            error = np.mean((target_np - nca_np) ** 2)
            errors.append(error)
        
        # Calcul des erreurs par source
        source_errors = []
        positions = sources_data['positions']
        n_sources = len(positions)
        grid_size = target_seq[0].shape[0]
        
        # Pour chaque source, calculer l'erreur dans sa zone d'influence
        for i, pos in enumerate(positions):
            pos_errors = []
            for t in range(min(len(target_seq), len(nca_seq))):
                target_np = target_seq[t].detach().cpu().numpy()
                nca_np = nca_seq[t].detach().cpu().numpy()
                
                # Définition d'un masque pour la zone d'influence (cercle de rayon 3)
                mask = np.zeros_like(target_np, dtype=bool)
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        ni, nj = pos[0] + di, pos[1] + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            if di**2 + dj**2 <= 9:  # Rayon 3
                                mask[ni, nj] = True
                
                # Calcul de l'erreur dans la zone d'influence
                local_error = np.mean((target_np[mask] - nca_np[mask]) ** 2) if np.any(mask) else 0
                pos_errors.append(local_error)
            
            source_errors.append(pos_errors)
        
        # Création du graphique
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        fig.suptitle(f'Stage 6: Convergence avec Sources Multiples - Seed {seed}', fontsize=14)
        
        # Graphique principal des erreurs
        ax1.plot(errors, 'b-', linewidth=2, label='Erreur MSE Globale')
        ax1.axhline(y=0.000005, color='r', linestyle='--', label='Seuil de convergence (5e-6)')
        ax1.set_xlabel('Pas de temps')
        ax1.set_ylabel('Erreur MSE')
        ax1.set_title('Convergence Globale')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Graphique des erreurs par source
        colors = plt.cm.tab10(np.linspace(0, 1, n_sources))
        for i in range(n_sources):
            label = f"Source {i+1} (I₀={sources_data['initial_intensities'][i]:.2f}, τ={sources_data['attenuation_rates'][i]:.4f})"
            ax2.plot(source_errors[i], color=colors[i], label=label, linewidth=1.5)
        
        ax2.set_xlabel('Pas de temps')
        ax2.set_ylabel('Erreur MSE Locale')
        ax2.set_title('Convergence par Zone de Source')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = output_dir / "convergence_multi_sources.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _create_source_intensity_plots(self, output_dir: Path, sources_data: Dict[str, Any]) -> None:
        """
        Crée des graphiques montrant l'évolution des intensités des sources au cours du temps.
        
        Args:
            output_dir: Répertoire de sortie
            sources_data: Données des sources
        """
        intensities_over_time = sources_data['intensities_over_time']
        initial_intensities = sources_data['initial_intensities']
        attenuation_rates = sources_data['attenuation_rates']
        n_sources = len(intensities_over_time)
        
        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Évolution des Intensités des Sources au Cours du Temps', fontsize=14)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_sources))
        for i, intensity_seq in enumerate(intensities_over_time):
            label = f"Source {i+1} (I₀={initial_intensities[i]:.2f}, τ={attenuation_rates[i]:.4f})"
            ax.plot(intensity_seq, color=colors[i], label=label, linewidth=2)
        
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Intensité')
        ax.set_title('Atténuation Temporelle des Sources')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        intensity_path = output_dir / "intensities_evolution.png"
        plt.savefig(intensity_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _create_source_interaction_visualization(self, output_dir: Path, target_seq: List[torch.Tensor],
                                              nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                                              sources_data: Dict[str, Any]) -> None:
        """
        Crée une visualisation spécifique des zones d'interaction entre sources.
        
        Args:
            output_dir: Répertoire de sortie
            target_seq: Séquence cible
            nca_seq: Séquence produite par le NCA
            obstacle_mask: Masque des obstacles
            sources_data: Données des sources
        """
        # Sélection d'un pas de temps représentatif (25% de la séquence)
        time_step = min(len(target_seq), len(nca_seq)) // 4
        
        # Conversion en numpy
        target_np = target_seq[time_step].detach().cpu().numpy()
        nca_np = nca_seq[time_step].detach().cpu().numpy()
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        
        # Calcul de la différence
        diff = np.abs(target_np - nca_np)
        
        # Positions des sources
        positions = sources_data['positions']
        n_sources = len(positions)
        intensities = [seq[time_step] if time_step < len(seq) else 0 for seq in sources_data['intensities_over_time']]
        
        # Création du graphique
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Analyse des Interactions entre Sources - Pas de temps {time_step}', fontsize=14)
        
        # Image cible
        im1 = axes[0].imshow(target_np, cmap=self.cmap_main, vmin=0, vmax=1)
        axes[0].contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
        axes[0].set_title('Diffusion Cible')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Prédiction NCA
        im2 = axes[1].imshow(nca_np, cmap=self.cmap_main, vmin=0, vmax=1)
        axes[1].contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
        axes[1].set_title('Prédiction NCA')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        # Carte de différence
        im3 = axes[2].imshow(diff, cmap='viridis', vmin=0, vmax=0.1)
        axes[2].contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
        axes[2].set_title('Différence Absolue')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        plt.colorbar(im3, ax=axes[2], label='Différence')
        
        # Marquage des sources avec couleurs distinctes
        colors = plt.cm.tab10(np.linspace(0, 1, n_sources))
        for ax_idx, ax in enumerate(axes):
            for i, (pos, intensity) in enumerate(zip(positions, intensities)):
                if intensity > 0.01:  # Ne pas afficher les sources éteintes
                    circle = plt.Circle((pos[1], pos[0]), 0.5, color=colors[i],
                                      alpha=min(0.8, intensity))
                    ax.add_artist(circle)
                    ax.text(pos[1], pos[0], f"{i+1}", ha='center', va='center',
                          color='white', fontsize=8, fontweight='bold')
        
        # Légende des sources
        for i in range(n_sources):
            initial = sources_data['initial_intensities'][i]
            rate = sources_data['attenuation_rates'][i]
            fig.text(0.02 + i*0.25, 0.02,
                   f"Source {i+1}: I₀={initial:.2f}, τ={rate:.4f}, I(t)={intensities[i]:.2f}",
                   color=colors[i], fontsize=9, transform=fig.transFigure)
        
        plt.tight_layout()
        interaction_path = output_dir / "source_interactions.png"
        plt.savefig(interaction_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    def _create_standard_animation(self, output_dir: Path, target_seq: List[torch.Tensor],
                                 nca_seq: List[torch.Tensor], obstacle_mask: torch.Tensor,
                                 source_mask: torch.Tensor) -> None:
        """
        Crée une animation standard sans données spécialisées sur les sources multiples.
        
        Args:
            output_dir: Répertoire de sortie
            target_seq: Séquence cible
            nca_seq: Séquence produite par le NCA
            obstacle_mask: Masque des obstacles
            source_mask: Masque de la source
        """
        # Conversion en numpy
        target_np = [t.detach().cpu().numpy() for t in target_seq]
        nca_np = [n.detach().cpu().numpy() for n in nca_seq]
        obstacle_np = obstacle_mask.detach().cpu().numpy()
        source_np = source_mask.detach().cpu().numpy()
        
        # Animation comparative
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Stage 6: Animation Générique (données sources multiples non disponibles)', fontsize=14)
        
        def animate_comparison(frame):
            ax1.clear()
            ax2.clear()
            
            # Cible
            im1 = ax1.imshow(target_np[frame], cmap=self.cmap_main, vmin=0, vmax=1)
            ax1.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax1.contour(source_np, levels=[0.5], colors='white', linewidths=1)
            ax1.set_title(f'Cible - t={frame}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # NCA
            im2 = ax2.imshow(nca_np[frame], cmap=self.cmap_main, vmin=0, vmax=1)
            ax2.contour(obstacle_np, levels=[0.5], colors='cyan', linewidths=2)
            ax2.contour(source_np, levels=[0.5], colors='white', linewidths=1)
            ax2.set_title(f'NCA - t={frame}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            return [im1, im2]
        
        n_frames = min(len(target_np), len(nca_np))
        ani = animation.FuncAnimation(
            fig, animate_comparison, frames=n_frames, interval=200, blit=False
        )
        
        # Sauvegarde de l'animation
        animation_path = output_dir / "animation_stage6_generic.gif"
        ani.save(animation_path, writer='pillow', fps=5, dpi=100)
        plt.close()
        
    def _create_standard_convergence_plot(self, output_dir: Path, target_seq: List[torch.Tensor],
                                        nca_seq: List[torch.Tensor], seed: int) -> None:
        """
        Crée un graphique de convergence standard sans données spécialisées.
        
        Args:
            output_dir: Répertoire de sortie
            target_seq: Séquence cible
            nca_seq: Séquence produite par le NCA
            seed: Graine aléatoire
        """
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
        ax.axhline(y=0.000005, color='r', linestyle='--', label='Seuil convergence Stage 6')
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_title(f'Convergence Stage 6 (Générique) - Seed {seed}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = output_dir / "convergence_stage6_generic.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
