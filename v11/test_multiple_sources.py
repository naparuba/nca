"""
Test NCA avec sources multiples à intensité maximale et faible atténuation.
Charge un modèle entraîné et visualise son comportement avec plusieurs sources.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
import os
import argparse
from typing import List, Tuple, Dict, Any, Optional, Union

# Réutilisation du modèle NCA
class ImprovedNCA(torch.nn.Module):
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
            layers.append(torch.nn.Linear(current_size, hidden_size))
            layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.1))
            current_size = hidden_size

        layers.append(torch.nn.Linear(hidden_size, 1))
        layers.append(torch.nn.Tanh())

        self.network = torch.nn.Sequential(*layers)
        self.delta_scale = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.network(x)
        return delta * self.delta_scale


class MultiSourceSimulator:
    """Simulateur de diffusion avec sources multiples."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0

    def generate_grid(self, size: int, sources: List[Tuple[int, int, float]],
                      obstacles: Optional[List[Tuple[int, int, int, int]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Génère une grille avec les sources et obstacles spécifiés.
        
        Args:
            size: Taille de la grille (carrée)
            sources: Liste de tuples (i, j, intensité) pour chaque source
            obstacles: Liste optionnelle de tuples (i, j, hauteur, largeur) pour les obstacles
            
        Returns:
            (grille, masque_sources, masque_obstacles)
        """
        # Initialisation des tenseurs
        grid = torch.zeros((size, size), device=self.device)
        source_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)
        
        # Placement des sources
        for i, j, intensity in sources:
            grid[i, j] = intensity
            source_mask[i, j] = True
        
        # Placement des obstacles (rectangles)
        if obstacles:
            for i, j, height, width in obstacles:
                obstacle_mask[i:i+height, j:j+width] = True
                
        # S'assurer que les obstacles ne chevauchent pas les sources
        obstacle_mask[source_mask] = False
        
        return grid, source_mask, obstacle_mask

    def step(self, grid: torch.Tensor, source_mask: torch.Tensor,
            obstacle_mask: torch.Tensor, source_intensities: List[Tuple[Tuple[int, int], float]]) -> torch.Tensor:
        """
        Effectue un pas de diffusion avec gestion des intensités par source.
        
        Args:
            grid: Grille actuelle
            source_mask: Masque des sources
            obstacle_mask: Masque des obstacles
            source_intensities: Liste de tuples ((i, j), intensité) pour chaque source
            
        Returns:
            Nouvelle grille après diffusion
        """
        # Diffusion par convolution
        x = grid.unsqueeze(0).unsqueeze(0)
        new_grid = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)

        # Application des contraintes d'obstacles
        new_grid[obstacle_mask] = 0.0

        # Application des intensités aux sources
        for (i, j), intensity in source_intensities:
            new_grid[i, j] = intensity

        return new_grid


class NCAUpdater:
    """
    Classe pour appliquer le modèle NCA sur une grille avec plusieurs sources.
    """
    
    def __init__(self, model: ImprovedNCA, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device

    def step(self, grid: torch.Tensor, source_mask: torch.Tensor,
             obstacle_mask: torch.Tensor, source_intensities: List[Tuple[Tuple[int, int], float]]) -> torch.Tensor:
        """
        Applique un pas du modèle NCA.
        
        Args:
            grid: Grille actuelle
            source_mask: Masque des sources
            obstacle_mask: Masque des obstacles
            source_intensities: Liste de tuples ((i, j), intensité) pour chaque source
            
        Returns:
            Nouvelle grille après application du NCA
        """
        H, W = grid.shape

        # Extraction vectorisée des patches 3x3
        grid_padded = F.pad(grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        patches = F.unfold(grid_padded, kernel_size=3, stride=1)
        patches = patches.squeeze(0).transpose(0, 1)

        # Features additionnelles
        source_flat = source_mask.flatten().float().unsqueeze(1)
        obstacle_flat = obstacle_mask.flatten().float().unsqueeze(1)
        
        # Préparation des entrées du réseau
        full_patches = torch.cat([patches, source_flat, obstacle_flat], dim=1)

        # Application du NCA sur les positions valides
        valid_mask = ~obstacle_mask.flatten()

        if valid_mask.any():
            valid_patches = full_patches[valid_mask]
            deltas = self.model(valid_patches)

            new_grid = grid.clone().flatten()
            new_grid[valid_mask] += deltas.squeeze()
            new_grid = torch.clamp(new_grid, 0.0, 1.0).reshape(H, W)
        else:
            new_grid = grid.clone()

        # Application des contraintes finales
        new_grid[obstacle_mask] = 0.0
        
        # Mise à jour des intensités pour chaque source
        for (i, j), intensity in source_intensities:
            new_grid[i, j] = intensity

        return new_grid


def run_multi_source_simulation(model_path: str, output_dir: str,
                              n_steps: int = 50,
                              attenuation_rate: float = 0.001,
                              save_animation: bool = True):
    """
    Exécute une simulation avec plusieurs sources et faible atténuation.
    
    Args:
        model_path: Chemin vers le fichier du modèle entraîné
        output_dir: Répertoire de sortie pour les animations et images
        n_steps: Nombre de pas de simulation
        attenuation_rate: Taux d'atténuation des sources (très faible par défaut)
        save_animation: Indique si l'animation doit être sauvegardée
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Utilisation du device: {device}")
    
    # Création du répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Chargement du modèle
    print(f"📂 Chargement du modèle depuis {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = ImprovedNCA(input_size=11).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modèle chargé avec succès")
    
    # Configuration de la simulation
    simulator = MultiSourceSimulator(device)
    updater = NCAUpdater(model, device)
    grid_size = 32  # Grille plus grande pour mieux voir les interactions
    
    # Définition des sources (au moins 3 avec intensité maximale)
    sources = [
        (10, 10, 1.0),    # Source 1 - coin supérieur gauche
        (8, 16, 1.0),   # Source 2 - coin supérieur droit
        #(24, 16, 1.0),  # Source 3 - milieu bas
        #(16, 16, 0.7)   # Source 4 - centre (intensité légèrement plus faible)
    ]
    
    # Définition de quelques obstacles pour rendre la simulation plus intéressante
    obstacles = [
        (15, 0, 2, 10),    # Obstacle horizontal supérieur gauche
        (15, 22, 2, 10),   # Obstacle horizontal supérieur droit
        (13, 15, 6, 2)     # Obstacle vertical central
    ]
    
    # Initialisation de la grille et des masques
    grid, source_mask, obstacle_mask = simulator.generate_grid(grid_size, sources, obstacles)
    
    # Préparation de la séquence d'intensités avec atténuation faible
    source_positions = [(i, j) for i, j, _ in sources]
    source_intensities = [intensity for _, _, intensity in sources]
    
    # Séquences d'intensités avec atténuation lente
    intensity_sequences = []
    for intensity in source_intensities:
        sequence = [intensity]
        for _ in range(1, n_steps):
            new_intensity = max(0.0, sequence[-1] - attenuation_rate)
            sequence.append(new_intensity)
        intensity_sequences.append(sequence)
    
    # Exécution de la simulation - cas de référence avec diffusion simple
    diffusion_sequence = [grid.clone()]
    for step in range(n_steps):
        # Préparation des intensités pour ce pas
        step_intensities = [((i, j), intensity_sequences[idx][step])
                           for idx, (i, j) in enumerate(source_positions)]
        
        # Application de la diffusion
        grid = simulator.step(grid, source_mask, obstacle_mask, step_intensities)
        diffusion_sequence.append(grid.clone())
    
    # Réinitialisation pour la simulation avec NCA
    grid, _, _ = simulator.generate_grid(grid_size, sources, obstacles)
    
    # Exécution avec le modèle NCA
    nca_sequence = [grid.clone()]
    with torch.no_grad():
        for step in range(n_steps):
            # Préparation des intensités pour ce pas
            step_intensities = [((i, j), intensity_sequences[idx][step])
                              for idx, (i, j) in enumerate(source_positions)]
            
            # Application du NCA
            grid = updater.step(grid, source_mask, obstacle_mask, step_intensities)
            nca_sequence.append(grid.clone())
    
    # Conversion pour visualisation
    diffusion_np = [t.cpu().numpy() for t in diffusion_sequence]
    nca_np = [t.cpu().numpy() for t in nca_sequence]
    obstacle_np = obstacle_mask.cpu().numpy()
    
    # Visualisation comparative entre diffusion et NCA
    if save_animation:
        create_comparison_animation(diffusion_np, nca_np, obstacle_np, source_positions,
                                  intensity_sequences, output_dir)
    
    # Visualisation de quelques étapes clés
    create_snapshot_comparison(diffusion_np, nca_np, obstacle_np, source_positions,
                             [0, n_steps//4, n_steps//2, n_steps-1], output_dir)
    
    print(f"✅ Simulation terminée avec succès")
    print(f"📊 Visualisations sauvegardées dans {output_dir}")


def create_comparison_animation(diffusion_seq, nca_seq,
                              obstacle_mask, source_positions,
                              intensity_sequences, output_dir):
    """
    Crée une animation comparative entre diffusion simple et NCA.
    
    Args:
        diffusion_seq: Séquence de diffusion simple
        nca_seq: Séquence NCA
        obstacle_mask: Masque des obstacles
        source_positions: Positions des sources
        intensity_sequences: Séquences d'intensités pour chaque source
        output_dir: Répertoire de sortie
    """
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    def animate_comparison(frame):
        ax1.clear()
        ax2.clear()
        
        # Diffusion simple
        im1 = ax1.imshow(diffusion_seq[frame], cmap='hot', vmin=0, vmax=1)
        ax1.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
        ax1.set_title(f'Diffusion - Étape {frame}')
        
        # Affichage des sources avec leur intensité actuelle
        for idx, (i, j) in enumerate(source_positions):
            intensity = intensity_sequences[idx][min(frame, len(intensity_sequences[idx])-1)]
            ax1.plot(j, i, 'o', markersize=5,
                   color=plt.cm.viridis(intensity),
                   markeredgecolor='white')
        
        # NCA
        im2 = ax2.imshow(nca_seq[frame], cmap='hot', vmin=0, vmax=1)
        ax2.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
        ax2.set_title(f'NCA - Étape {frame}')
        
        # Affichage des sources avec leur intensité actuelle
        for idx, (i, j) in enumerate(source_positions):
            intensity = intensity_sequences[idx][min(frame, len(intensity_sequences[idx])-1)]
            ax2.plot(j, i, 'o', markersize=5,
                   color=plt.cm.viridis(intensity),
                   markeredgecolor='white')
        
        # Supprimer les axes
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        return [im1, im2]
    
    # Création de l'animation
    n_frames = min(len(diffusion_seq), len(nca_seq))
    ani = animation.FuncAnimation(
        fig, animate_comparison, frames=n_frames, interval=100, blit=False
    )
    
    # Sauvegarde de l'animation au format GIF au lieu de MP4
    animation_path = os.path.join(output_dir, "multi_source_animation.gif")
    ani.save(animation_path, writer='pillow', fps=5, dpi=100)
    plt.close(fig)
    
    print(f"🎬 Animation sauvegardée dans {animation_path}")
    
    # Création d'une animation NCA uniquement (plus légère et plus fluide)
    print(f"🎬 Création d'une animation NCA uniquement...")
    fig_nca, ax_nca = plt.subplots(figsize=(8, 8))
    
    def animate_nca_only(frame):
        ax_nca.clear()
        im = ax_nca.imshow(nca_seq[frame], cmap='hot', vmin=0, vmax=1)
        ax_nca.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
        ax_nca.set_title(f'NCA - Étape {frame}')
        
        # Affichage des sources avec leur intensité actuelle
        for idx, (i, j) in enumerate(source_positions):
            intensity = intensity_sequences[idx][min(frame, len(intensity_sequences[idx])-1)]
            ax_nca.plot(j, i, 'o', markersize=5,
                      color=plt.cm.viridis(intensity),
                      markeredgecolor='white')
        
        ax_nca.set_xticks([])
        ax_nca.set_yticks([])
        return [im]
    
    ani_nca = animation.FuncAnimation(
        fig_nca, animate_nca_only, frames=n_frames, interval=100, blit=False
    )
    
    nca_animation_path = os.path.join(output_dir, "nca_only_animation.gif")
    ani_nca.save(nca_animation_path, writer='pillow', fps=5, dpi=120)
    plt.close(fig_nca)
    
    print(f"🎬 Animation NCA uniquement sauvegardée dans {nca_animation_path}")


def create_snapshot_comparison(diffusion_seq, nca_seq,
                             obstacle_mask, source_positions,
                             steps, output_dir):
    """
    Crée une visualisation des étapes clés pour comparer diffusion et NCA.
    
    Args:
        diffusion_seq: Séquence de diffusion simple
        nca_seq: Séquence NCA
        obstacle_mask: Masque des obstacles
        source_positions: Positions des sources
        steps: Liste des indices d'étapes à visualiser
        output_dir: Répertoire de sortie
    """
    plt.style.use('dark_background')
    n_steps = len(steps)
    fig, axes = plt.subplots(2, n_steps, figsize=(n_steps*4, 8))
    
    for col, step in enumerate(steps):
        # Diffusion
        axes[0, col].imshow(diffusion_seq[step], cmap='hot', vmin=0, vmax=1)
        axes[0, col].contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
        axes[0, col].set_title(f'Diffusion - Étape {step}')
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        
        # Sources
        for i, j in source_positions:
            axes[0, col].plot(j, i, 'o', markersize=5, color='yellow', markeredgecolor='white')
        
        # NCA
        axes[1, col].imshow(nca_seq[step], cmap='hot', vmin=0, vmax=1)
        axes[1, col].contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
        axes[1, col].set_title(f'NCA - Étape {step}')
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        
        # Sources
        for i, j in source_positions:
            axes[1, col].plot(j, i, 'o', markersize=5, color='yellow', markeredgecolor='white')
    
    plt.tight_layout()
    snapshot_path = os.path.join(output_dir, "multi_source_snapshots.png")
    plt.savefig(snapshot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"📸 Comparaison d'étapes sauvegardée dans {snapshot_path}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Test NCA avec sources multiples')
    parser.add_argument('--model', type=str, default='nca_outputs_modular_progressive_obstacles_variable_intensity_seed_123/final_model.pth',
                      help='Chemin du modèle NCA entraîné')
    parser.add_argument('--output', type=str, default='multi_source_test_output',
                      help='Répertoire de sortie pour les visualisations')
    parser.add_argument('--steps', type=int, default=50,
                      help='Nombre de pas de simulation')
    parser.add_argument('--attenuation', type=float, default=0.001,
                      help='Taux d\'atténuation des sources (plus petit = plus lent)')
    
    args = parser.parse_args()
    
    print(f"\n" + "="*80)
    print(f"🚀 TEST NCA AVEC SOURCES MULTIPLES")
    print(f"="*80)
    
    run_multi_source_simulation(
        model_path=args.model,
        output_dir=args.output,
        n_steps=args.steps,
        attenuation_rate=args.attenuation
    )
    
    print(f"\n" + "="*80)
    print(f"✅ TEST TERMINÉ AVEC SUCCÈS")
    print(f"="*80)


if __name__ == "__main__":
    main()
