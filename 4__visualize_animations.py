#!/usr/bin/env python3
"""
Script de visualisation des animations NCA sauvegardées.
Permet de voir les résultats de l'entraînement et comparer avec la simulation cible.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from pathlib import Path

def load_animation_data(filepath):
    """
    Charge les données d'animation depuis un fichier .npy
    
    Args:
        filepath: Chemin vers le fichier d'animation
    
    Returns:
        Liste des frames avec données NCA et cible
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
    Crée un GIF animé comparant NCA et simulation cible
    
    Args:
        frames_data: Données des frames
        output_path: Chemin de sortie pour le GIF
        title: Titre de l'animation
        fps: Images par seconde
    """
    if len(frames_data) == 0:
        print("❌ Pas de données à animer")
        return
    
    # Configuration de la figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initialisation des images
    frame0 = frames_data[0]
    nca_grid = frame0['nca_grid']
    target_grid = frame0['target_grid']
    
    # Images initiales
    im1 = ax1.imshow(nca_grid, cmap='plasma', vmin=0, vmax=1, animated=True)
    ax1.set_title('NCA (Réseau)')
    ax1.set_xlabel('Position X')
    ax1.set_ylabel('Position Y')
    
    im2 = ax2.imshow(target_grid, cmap='plasma', vmin=0, vmax=1, animated=True)
    ax2.set_title('Cible (Simulation physique)')
    ax2.set_xlabel('Position X')
    ax2.set_ylabel('Position Y')
    
    # Différence absolue
    diff = np.abs(nca_grid - target_grid)
    im3 = ax3.imshow(diff, cmap='viridis', vmin=0, vmax=0.5, animated=True)
    ax3.set_title('Différence |NCA - Cible|')
    ax3.set_xlabel('Position X')
    ax3.set_ylabel('Position Y')
    
    # Barres de couleur
    plt.colorbar(im1, ax=ax1, label='Intensité')
    plt.colorbar(im2, ax=ax2, label='Intensité')
    plt.colorbar(im3, ax=ax3, label='Erreur absolue')
    
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
            
            # Calcul des métriques
            mse = np.mean((nca_grid - target_grid) ** 2)
            max_diff = np.max(diff)
            
            plt.suptitle(f"{title} - Étape {frame_idx+1}/{len(frames_data)}\n"
                        f"MSE: {mse:.6f} | Erreur max: {max_diff:.4f}")
        
        return [im1, im2, im3]
    
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
    Crée une comparaison statique montrant plusieurs étapes clés
    
    Args:
        frames_data: Données des frames
        output_path: Chemin de sortie
        title: Titre
        steps_to_show: Indices des étapes à montrer
    """
    if len(frames_data) == 0:
        return
    
    n_steps = len(steps_to_show)
    fig, axes = plt.subplots(3, n_steps, figsize=(4*n_steps, 12))
    
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
        
        # Métriques
        mse = np.mean((nca_grid - target_grid) ** 2)
        max_diff = np.max(diff)
        axes[2, i].text(0.02, 0.98, f'MSE: {mse:.4f}\nMax: {max_diff:.3f}', 
                       transform=axes[2, i].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Barres de couleur
    plt.colorbar(im1, ax=axes[0, :], label='Intensité', shrink=0.8)
    plt.colorbar(im2, ax=axes[1, :], label='Intensité', shrink=0.8)
    plt.colorbar(im3, ax=axes[2, :], label='Erreur absolue', shrink=0.8)
    
    plt.suptitle(f"{title} - Comparaison à différentes étapes", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Comparaison statique sauvegardée: {output_path}")

def analyze_convergence(frames_data, title):
    """
    Analyse la convergence du NCA vers la cible
    
    Args:
        frames_data: Données des frames
        title: Titre pour l'affichage
    """
    if len(frames_data) == 0:
        return
    
    mse_values = []
    max_errors = []
    
    for frame in frames_data:
        nca_grid = frame['nca_grid']
        target_grid = frame['target_grid']
        
        mse = np.mean((nca_grid - target_grid) ** 2)
        max_err = np.max(np.abs(nca_grid - target_grid))
        
        mse_values.append(mse)
        max_errors.append(max_err)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # MSE au cours du temps
    ax1.plot(mse_values, linewidth=2, color='blue')
    ax1.set_xlabel('Étape temporelle')
    ax1.set_ylabel('MSE')
    ax1.set_title('Erreur quadratique moyenne')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Erreur maximale
    ax2.plot(max_errors, linewidth=2, color='red')
    ax2.set_xlabel('Étape temporelle')
    ax2.set_ylabel('Erreur maximale')
    ax2.set_title('Erreur absolue maximale')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"{title} - Analyse de convergence")
    plt.tight_layout()
    
    output_path = f"nca_outputs/convergence_{title.lower().replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Analyse de convergence sauvegardée: {output_path}")
    
    # Statistiques finales
    final_mse = mse_values[-1] if mse_values else 0
    final_max_err = max_errors[-1] if max_errors else 0
    print(f"📊 {title} - MSE finale: {final_mse:.6f} | Erreur max finale: {final_max_err:.4f}")

def main():
    """
    Fonction principale de visualisation
    """
    print("=" * 60)
    print("🎬 Visualiseur d'animations NCA")
    print("=" * 60)
    
    output_dir = Path("nca_outputs")
    if not output_dir.exists():
        print("❌ Dossier nca_outputs non trouvé. Lancez d'abord l'entraînement.")
        return
    
    # Recherche des fichiers d'animation
    animation_files = list(output_dir.glob("animation_*.npy"))
    
    if not animation_files:
        print("❌ Aucune animation trouvée dans nca_outputs/")
        return
    
    print(f"📁 Trouvé {len(animation_files)} animation(s)")
    
    for anim_file in animation_files:
        print(f"\n🎭 Traitement de {anim_file.name}")
        
        # Chargement des données
        frames_data = load_animation_data(anim_file)
        if len(frames_data) == 0:
            continue
        
        # Extraction du titre à partir du nom de fichier
        base_name = anim_file.stem.replace("animation_", "").replace("_", " ").title()
        
        # Création du GIF animé
        gif_path = str(output_dir / f"gif_{anim_file.stem}.gif")
        create_comparison_gif(frames_data, gif_path, base_name, fps=8)
        
        # Comparaison statique
        static_path = str(output_dir / f"static_{anim_file.stem}.png")
        create_static_comparison(frames_data, static_path, base_name)
        
        # Analyse de convergence
        analyze_convergence(frames_data, base_name)
    
    print(f"\n✨ Visualisation terminée ! Consultez le dossier {output_dir}/")
    print("📁 Fichiers générés :")
    print("   - GIFs animés (.gif) : animations complètes")
    print("   - Images statiques (.png) : comparaisons par étapes")
    print("   - Analyses de convergence (.png) : évolution des erreurs")

if __name__ == "__main__":
    main()
