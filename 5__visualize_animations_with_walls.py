#!/usr/bin/env python3
"""
Script de visualisation des animations NCA sauvegardées avec obstacles.
Permet de voir les résultats de l'entraînement et comparer avec la simulation cible.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse
from pathlib import Path
import glob

def parse_arguments():
    """
    Parse les arguments de ligne de commande pour la visualisation.

    Returns:
        Namespace avec les arguments parsés
    """
    parser = argparse.ArgumentParser(
        description='Visualiseur d\'animations NCA avec obstacles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed spécifique à visualiser (si None, cherche tous les répertoires)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Répertoire de sortie spécifique à analyser'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=8,
        help='Images par seconde pour les GIFs'
    )

    return parser.parse_args()

def find_output_directories(seed=None):
    """
    Trouve tous les répertoires de sortie NCA disponibles.

    Args:
        seed: Seed spécifique à chercher (optionnel)

    Returns:
        Liste des chemins de répertoires trouvés
    """
    if seed is not None:
        # Cherche un répertoire spécifique
        pattern = f"nca_outputs_with_walls_seed_{seed}"
        dirs = [Path(pattern)] if Path(pattern).exists() else []
    else:
        # Cherche tous les répertoires avec pattern
        pattern = "nca_outputs_with_walls_seed_*"
        dirs = [Path(d) for d in glob.glob(pattern) if Path(d).is_dir()]

        # Ajoute aussi l'ancien format sans seed
        old_format = Path("nca_outputs_with_walls")
        if old_format.exists():
            dirs.append(old_format)

    return sorted(dirs)

def load_animation_data(filepath):
    """
    Charge les données d'animation depuis un fichier .npy
    
    Args:
        filepath: Chemin vers le fichier d'animation
    
    Returns:
        Liste des frames avec données NCA, cible et obstacles
    """
    try:
        frames = np.load(filepath, allow_pickle=True)
        print(f"✅ Chargé {len(frames)} frames depuis {filepath}")
        return frames
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {filepath}: {e}")
        return []

def create_comparison_gif(frames_data, output_path, title, fps=10):
    """
    Crée un GIF animé comparant NCA et simulation cible avec obstacles

    Args:
        frames_data: Données des frames
        output_path: Chemin de sortie pour le GIF
        title: Titre de l'animation
        fps: Images par seconde
    """
    if len(frames_data) == 0:
        print("❌ Pas de données à animer")
        return
    
    # Configuration de la figure avec 4 panneaux
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Initialisation des images
    frame0 = frames_data[0]
    nca_grid = frame0['nca_grid']
    target_grid = frame0['target_grid']
    
    # Vérification de la présence des obstacles
    has_obstacles = 'obstacle_mask' in frame0
    if has_obstacles:
        obstacle_mask = frame0['obstacle_mask']
        source_mask = frame0['source_mask'] if 'source_mask' in frame0 else np.zeros_like(obstacle_mask)

    # Panneau 1: NCA
    im1 = axes[0].imshow(nca_grid, cmap='plasma', vmin=0, vmax=1, animated=True)
    axes[0].set_title('NCA (Réseau)')
    axes[0].set_xlabel('Position X')
    axes[0].set_ylabel('Position Y')

    # Panneau 2: Cible
    im2 = axes[1].imshow(target_grid, cmap='plasma', vmin=0, vmax=1, animated=True)
    axes[1].set_title('Cible (Simulation physique)')
    axes[1].set_xlabel('Position X')
    axes[1].set_ylabel('Position Y')

    # Panneau 3: Différence absolue
    diff = np.abs(nca_grid - target_grid)
    im3 = axes[2].imshow(diff, cmap='viridis', vmin=0, vmax=0.5, animated=True)
    axes[2].set_title('Différence |NCA - Cible|')
    axes[2].set_xlabel('Position X')
    axes[2].set_ylabel('Position Y')

    # Panneau 4: Obstacles et sources (si disponibles)
    if has_obstacles:
        # Création d'une image composite pour obstacles et sources
        composite = np.zeros((*obstacle_mask.shape, 3))  # RGB
        # Obstacles en gris foncé
        composite[obstacle_mask] = [0.3, 0.3, 0.3]
        # Sources en rouge
        composite[source_mask] = [1.0, 0.0, 0.0]
        # Zones libres en transparent (blanc très clair)
        free_areas = ~(obstacle_mask | source_mask)
        composite[free_areas] = [0.95, 0.95, 0.95]

        im4 = axes[3].imshow(composite, animated=False)  # Statique
        axes[3].set_title('Obstacles (gris) et Source (rouge)')
    else:
        # Si pas d'obstacles, afficher juste la différence à nouveau
        im4 = axes[3].imshow(diff, cmap='viridis', vmin=0, vmax=0.5, animated=True)
        axes[3].set_title('Différence (bis)')

    axes[3].set_xlabel('Position X')
    axes[3].set_ylabel('Position Y')

    # Barres de couleur
    plt.colorbar(im1, ax=axes[0], label='Intensité')
    plt.colorbar(im2, ax=axes[1], label='Intensité')
    plt.colorbar(im3, ax=axes[2], label='Erreur absolue')

    plt.tight_layout()
    
    # Fonction d'animation
    def animate(frame_idx):
        if frame_idx < len(frames_data):
            frame = frames_data[frame_idx]
            nca_grid = frame['nca_grid']
            target_grid = frame['target_grid']
            diff = np.abs(nca_grid - target_grid)
            
            im1.set_data(nca_grid)
            im2.set_data(target_grid)
            im3.set_data(diff)
            
            if not has_obstacles:
                im4.set_data(diff)

            # Calcul des métriques
            mse = np.mean((nca_grid - target_grid) ** 2)
            max_diff = np.max(diff)
            
            title_text = f"{title} - Étape {frame_idx+1}/{len(frames_data)}\n"
            title_text += f"MSE: {mse:.6f} | Erreur max: {max_diff:.4f}"

            if has_obstacles:
                n_obstacles = np.sum(obstacle_mask)
                title_text += f" | Obstacles: {n_obstacles} cellules"

            plt.suptitle(title_text)

        return [im1, im2, im3] + ([im4] if not has_obstacles else [])

    # Création de l'animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames_data), 
        interval=1000//fps, blit=True, repeat=True
    )
    
    # Sauvegarde
    print(f"💾 Création du GIF: {output_path}")
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"✅ GIF sauvegardé: {output_path}")

def create_static_comparison(frames_data, output_path, title, steps_to_show=[0, 10, 20, -1]):
    """
    Crée une comparaison statique montrant plusieurs étapes clés avec obstacles

    Args:
        frames_data: Données des frames
        output_path: Chemin de sortie
        title: Titre
        steps_to_show: Indices des étapes à montrer
    """
    if len(frames_data) == 0:
        return
    
    n_steps = len(steps_to_show)
    has_obstacles = 'obstacle_mask' in frames_data[0]
    n_rows = 4 if has_obstacles else 3

    fig, axes = plt.subplots(n_rows, n_steps, figsize=(4*n_steps, 3*n_rows))

    if n_steps == 1:
        axes = axes.reshape(-1, 1)
    
    for i, step_idx in enumerate(steps_to_show):
        if step_idx == -1:
            step_idx = len(frames_data) - 1
        
        if step_idx >= len(frames_data):
            continue
            
        frame = frames_data[step_idx]
        nca_grid = frame['nca_grid']
        target_grid = frame['target_grid']
        diff = np.abs(nca_grid - target_grid)
        
        # NCA
        im1 = axes[0, i].imshow(nca_grid, cmap='plasma', vmin=0, vmax=1)
        axes[0, i].set_title(f'NCA - Étape {step_idx+1}')
        axes[0, i].set_xlabel('Position X')
        if i == 0:
            axes[0, i].set_ylabel('Position Y')
        
        # Cible
        im2 = axes[1, i].imshow(target_grid, cmap='plasma', vmin=0, vmax=1)
        axes[1, i].set_title(f'Cible - Étape {step_idx+1}')
        axes[1, i].set_xlabel('Position X')
        if i == 0:
            axes[1, i].set_ylabel('Position Y')
        
        # Différence
        im3 = axes[2, i].imshow(diff, cmap='viridis', vmin=0, vmax=0.5)
        axes[2, i].set_title(f'Différence - Étape {step_idx+1}')
        axes[2, i].set_xlabel('Position X')
        if i == 0:
            axes[2, i].set_ylabel('Position Y')
        
        # Obstacles et sources (si disponibles)
        if has_obstacles:
            obstacle_mask = frame['obstacle_mask']
            source_mask = frame['source_mask'] if 'source_mask' in frame else np.zeros_like(obstacle_mask)

            # Image composite
            composite = np.zeros((*obstacle_mask.shape, 3))
            composite[obstacle_mask] = [0.3, 0.3, 0.3]  # Gris foncé pour obstacles
            composite[source_mask] = [1.0, 0.0, 0.0]    # Rouge pour sources
            free_areas = ~(obstacle_mask | source_mask)
            composite[free_areas] = [0.95, 0.95, 0.95]  # Blanc cassé pour zones libres

            axes[3, i].imshow(composite)
            axes[3, i].set_title(f'Obstacles & Source - Étape {step_idx+1}')
            axes[3, i].set_xlabel('Position X')
            if i == 0:
                axes[3, i].set_ylabel('Position Y')

        # Métriques
        mse = np.mean((nca_grid - target_grid) ** 2)
        max_diff = np.max(diff)
        metric_text = f'MSE: {mse:.4f}\nMax: {max_diff:.3f}'

        if has_obstacles:
            n_obstacles = np.sum(obstacle_mask)
            metric_text += f'\nObst.: {n_obstacles}'

        axes[2, i].text(0.02, 0.98, metric_text,
                       transform=axes[2, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Barres de couleur
    plt.colorbar(im1, ax=axes[0, :], label='Intensité', shrink=0.8)
    plt.colorbar(im2, ax=axes[1, :], label='Intensité', shrink=0.8)
    plt.colorbar(im3, ax=axes[2, :], label='Erreur absolue', shrink=0.8)
    
    title_suffix = " avec obstacles" if has_obstacles else ""
    plt.suptitle(f"{title}{title_suffix} - Comparaison à différentes étapes", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Comparaison statique sauvegardée: {output_path}")

def analyze_convergence(frames_data, title, output_dir):
    """
    Analyse la convergence du NCA vers la cible avec obstacles

    Args:
        frames_data: Données des frames
        title: Titre pour l'affichage
        output_dir: Répertoire de sortie
    """
    if len(frames_data) == 0:
        return
    
    mse_values = []
    max_errors = []
    has_obstacles = 'obstacle_mask' in frames_data[0]

    if has_obstacles:
        obstacle_coverage = []

    for frame in frames_data:
        nca_grid = frame['nca_grid']
        target_grid = frame['target_grid']
        
        mse = np.mean((nca_grid - target_grid) ** 2)
        max_err = np.max(np.abs(nca_grid - target_grid))
        
        mse_values.append(mse)
        max_errors.append(max_err)

        if has_obstacles:
            obstacle_mask = frame['obstacle_mask']
            coverage = np.sum(obstacle_mask) / obstacle_mask.size
            obstacle_coverage.append(coverage)

    n_plots = 3 if has_obstacles else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4))

    if n_plots == 2:
        axes = [axes[0], axes[1]]

    # MSE au cours du temps
    axes[0].plot(mse_values, linewidth=2, color='blue')
    axes[0].set_xlabel('Étape temporelle')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Erreur quadratique moyenne')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Erreur maximale
    axes[1].plot(max_errors, linewidth=2, color='red')
    axes[1].set_xlabel('Étape temporelle')
    axes[1].set_ylabel('Erreur maximale')
    axes[1].set_title('Erreur absolue maximale')
    axes[1].grid(True, alpha=0.3)

    # Couverture d'obstacles (si applicable)
    if has_obstacles:
        axes[2].plot(obstacle_coverage, linewidth=2, color='gray')
        axes[2].set_xlabel('Étape temporelle')
        axes[2].set_ylabel('Couverture d\'obstacles (%)')
        axes[2].set_title('Proportion d\'obstacles')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)

    title_suffix = " avec obstacles" if has_obstacles else ""
    plt.suptitle(f"{title}{title_suffix} - Analyse de convergence")
    plt.tight_layout()
    
    output_path = f"{output_dir}/convergence_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Analyse de convergence sauvegardée: {output_path}")
    
    # Statistiques finales
    final_mse = mse_values[-1] if mse_values else 0
    final_max_err = max_errors[-1] if max_errors else 0
    stats_text = f"📊 {title} - MSE finale: {final_mse:.6f} | Erreur max finale: {final_max_err:.4f}"

    if has_obstacles:
        avg_obstacle_coverage = np.mean(obstacle_coverage) if obstacle_coverage else 0
        stats_text += f" | Couverture obstacles: {avg_obstacle_coverage:.2%}"

    print(stats_text)

def main():
    """
    Fonction principale de visualisation avec support des obstacles et seed.
    """
    print("=" * 60)
    print("🎬 Visualiseur d'animations NCA avec obstacles")
    print("=" * 60)
    
    # Parse des arguments
    args = parse_arguments()

    # Trouve les répertoires de sortie
    if args.output_dir:
        output_dirs = [Path(args.output_dir)]
    else:
        output_dirs = find_output_directories(args.seed)

    if not output_dirs:
        if args.seed:
            print(f"❌ Aucun répertoire trouvé pour la seed {args.seed}")
            print(f"   Cherché: nca_outputs_with_walls_seed_{args.seed}")
        else:
            print("❌ Aucun répertoire de sortie NCA trouvé.")
            print("   Patterns cherchés: nca_outputs_with_walls_seed_*, nca_outputs_with_walls")
        print("   Lancez d'abord l'entraînement avec : python nca_light_diffuse_with_walls.py")
        return
    
    print(f"📁 Trouvé {len(output_dirs)} répertoire(s) de sortie:")
    for output_dir in output_dirs:
        print(f"   - {output_dir}")

    # Traite chaque répertoire
    for output_dir in output_dirs:
        print(f"\n📂 Traitement du répertoire: {output_dir}")

        # Extraction de la seed depuis le nom du répertoire
        if "seed_" in str(output_dir):
            seed_info = str(output_dir).split("seed_")[-1]
            print(f"🌱 Seed détectée: {seed_info}")

        # Recherche des fichiers d'animation
        animation_files = list(output_dir.glob("animation_*.npy"))

        if not animation_files:
            print(f"❌ Aucune animation trouvée dans {output_dir}/")
            continue

        print(f"🎭 Trouvé {len(animation_files)} animation(s)")

        for anim_file in animation_files:
            print(f"\n🎬 Traitement de {anim_file.name}")

            # Chargement des données
            frames_data = load_animation_data(anim_file)
            if len(frames_data) == 0:
                continue

            # Vérification de la présence d'obstacles
            has_obstacles = 'obstacle_mask' in frames_data[0]
            obstacle_info = " avec obstacles" if has_obstacles else ""
            print(f"🧱 Obstacles détectés: {'Oui' if has_obstacles else 'Non'}")

            # Extraction du titre à partir du nom de fichier
            base_name = anim_file.stem.replace("animation_", "").replace("_", " ").title()

            # Ajout de l'info seed au titre
            if "seed_" in str(output_dir):
                seed_info = str(output_dir).split("seed_")[-1]
                base_name += f" (Seed {seed_info})"

            # Création du GIF animé
            gif_path = str(output_dir / f"gif_{anim_file.stem}.gif")
            create_comparison_gif(frames_data, gif_path, base_name + obstacle_info, fps=args.fps)

            # Comparaison statique
            static_path = str(output_dir / f"static_{anim_file.stem}.png")
            create_static_comparison(frames_data, static_path, base_name)

            # Analyse de convergence
            analyze_convergence(frames_data, base_name, str(output_dir))

    print(f"\n✨ Visualisation terminée !")
    print("📁 Fichiers générés :")
    print("   - GIFs animés (.gif) : animations complètes avec 4 panneaux")
    print("   - Images statiques (.png) : comparaisons par étapes")
    print("   - Analyses de convergence (.png) : évolution des erreurs")
    if any(output_dirs):
        print("   - Support complet des obstacles et sources")
        print("   - Support multi-seed pour comparaisons")

if __name__ == "__main__":
    main()
