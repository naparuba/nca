from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from matplotlib import pyplot as plt

from config import CONFIG
from nca_model import ImprovedNCA
from simulator import get_simulator
from stage_manager import STAGE_MANAGER
from updater import OptimizedNCAUpdater


class ProgressiveVisualizer:
    """
    Système de visualisation avancé pour l'apprentissage modulaire.
    Génère des animations et graphiques comparatifs par étape.
    """
    
    
    def __init__(self):
        self.frame_data = {}  # Données par étape
    
    
    def visualize_stage_results(self, model: ImprovedNCA, stage_nb: int) -> None:
        """
        Visualise les résultats d'une étape spécifique.

        Args:
            model: Modèle NCA entraîné
            stage_nb: Numéro d'étape à visualiser
        Returns:
            Dictionnaire avec les données de visualisation
        """
        print(f"\n🎨 Génération des visualisations pour l'étape {stage_nb}...")
        
        # Génération de la séquence de test avec seed fixe
        torch.manual_seed(CONFIG.VISUALIZATION_SEED)
        np.random.seed(CONFIG.VISUALIZATION_SEED)
        simulator = get_simulator()
        target_seq, source_mask, obstacle_mask = simulator.generate_stage_sequence(
                stage_nb=stage_nb,
                n_steps=CONFIG.POSTVIS_STEPS,
                size=CONFIG.GRID_SIZE
        )
        
        # Prédiction du modèle
        model.eval()
        updater = OptimizedNCAUpdater(model)
        
        # Simulation NCA avec torch.no_grad() pour éviter le gradient
        nca_sequence = []
        grid_pred = torch.zeros_like(target_seq[0])
        grid_pred[source_mask] = CONFIG.SOURCE_INTENSITY
        nca_sequence.append(grid_pred.clone())
        
        with torch.no_grad():  # Désactive le calcul de gradient pour les visualisations
            for _ in range(CONFIG.POSTVIS_STEPS):
                grid_pred = updater.step(grid_pred, source_mask, obstacle_mask)
                nca_sequence.append(grid_pred.clone())
        
        # Création des visualisations avec .detach() pour sécurité
        vis_data = {
            'stage_nb':        stage_nb,
            'target_sequence': [t.detach().cpu().numpy() for t in target_seq],
            'nca_sequence':    [t.detach().cpu().numpy() for t in nca_sequence],
            'source_mask':     source_mask.detach().cpu().numpy(),
            'obstacle_mask':   obstacle_mask.detach().cpu().numpy(),
            'vis_seed':        CONFIG.VISUALIZATION_SEED,
        }
        
        # Sauvegarde des animations
        self._create_stage_animations(vis_data)
        self._create_stage_convergence_plot(vis_data)
        
        model.train()
        return
    
    
    def _create_stage_animations(self, vis_data: Dict[str, Any]):
        """Crée les animations GIF pour une étape."""
        stage_nb = vis_data['stage_nb']
        stage_dir = Path(CONFIG.OUTPUT_DIR) / f"stage_{stage_nb}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Animation comparative
        self._save_comparison_gif(
                vis_data['target_sequence'],
                vis_data['nca_sequence'],
                vis_data['obstacle_mask'],
                stage_dir / f"animation_comparaison_étape_{stage_nb}.gif",
                f"Étape {stage_nb} - Comparaison Cible vs NCA"
        )
        
        print(f"✅ Animations étape {stage_nb} sauvegardées dans {stage_dir}")
    
    
    def _create_stage_convergence_plot(self, vis_data: Dict[str, Any]):
        """Crée le graphique de convergence pour une étape."""
        stage_nb = vis_data['stage_nb']
        stage_dir = Path(CONFIG.OUTPUT_DIR) / f"stage_{stage_nb}"
        
        target_seq = vis_data['target_sequence']
        nca_seq = vis_data['nca_sequence']
        
        # Calcul de l'erreur temporelle
        errors = []
        for t in range(min(len(target_seq), len(nca_seq))):
            error = np.mean((target_seq[t] - nca_seq[t]) ** 2)
            errors.append(error)
        
        # Graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(errors, 'b-', linewidth=2, label='Erreur MSE')
        
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_title(f'Convergence Étape {stage_nb} - Seed {vis_data["vis_seed"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = stage_dir / f"convergence_étape_{stage_nb}.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graphique de convergence étape {stage_nb} sauvegardé: {convergence_path}")
    
    
    def _save_comparison_gif(self, target_seq: List[np.ndarray], nca_seq: List[np.ndarray],
                             obstacle_mask: np.ndarray, filepath: Path, title: str):
        """Sauvegarde un GIF de comparaison côte à côte."""
        import matplotlib.animation as animation
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Cible
            im1 = ax1.imshow(target_seq[frame], cmap='hot', vmin=0, vmax=1)
            ax1.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax1.set_title(f'Cible - t={frame}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # NCA
            im2 = ax2.imshow(nca_seq[frame], cmap='hot', vmin=0, vmax=1)
            ax2.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax2.set_title(f'NCA - t={frame}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            return [im1, im2]
        
        
        n_frames = min(len(target_seq), len(nca_seq))
        ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200, blit=False)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()
    
    
    def create_curriculum_summary(self):
        """Crée un résumé visuel complet du curriculum d'apprentissage."""
        print("\n🎨 Génération du résumé visuel du curriculum...")
        
        # Graphique de progression globale
        self._plot_curriculum_progression()
        
        # Comparaison inter-étapes
        self._plot_stage_comparison()
        
        print("✅ Résumé visuel complet généré")
    
    
    def _plot_curriculum_progression(self):
        """Graphique de la progression globale du curriculum."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 15))
        
        for stage in STAGE_MANAGER.get_stages():
            stage_nb = stage.get_stage_nb()
            loss_history = stage.get_loss_history()
            stage_epochs = list(range((stage_nb - 1) * CONFIG.NB_EPOCHS_BY_STAGE, stage_nb * CONFIG.NB_EPOCHS_BY_STAGE))
            
            if loss_history:
                ax1.plot(stage_epochs, loss_history,
                         color=stage.get_color(),
                         label=f'Étape {stage_nb}',
                         linewidth=2)
        
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Perte MSE')
        ax1.set_title('Progression du Curriculum d\'Apprentissage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Learning rate par étape
        for stage in STAGE_MANAGER.get_stages():
            stage_nb = stage.get_stage_nb()
            
            lrs = stage.get_metrics_lrs()
            if lrs:
                stage_epochs = list(range((stage_nb - 1) * CONFIG.NB_EPOCHS_BY_STAGE, stage_nb * CONFIG.NB_EPOCHS_BY_STAGE))
                ax2.plot(stage_epochs, lrs,
                         color=stage.get_color(),
                         label=f'LR Étape {stage_nb}',
                         linewidth=2)
        
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Évolution du Learning Rate par Étape')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(Path(CONFIG.OUTPUT_DIR) / "curriculum_progression.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    
    def _plot_stage_comparison(self):
        """Graphique de comparaison entre étapes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12))
        
        stages = [1, 2, 3]
        stage_names = ["Sans obstacles", "Un obstacle", "Obstacles multiples"]
        
        ax1.set_ylabel('Perte finale')
        ax1.set_title('Perte Finale par Étape')
        ax1.set_yscale('log')
        
        # Époques utilisées par étape
        epochs_used = CONFIG.NB_EPOCHS_BY_STAGE
        
        x = np.arange(len(stages))
        width = 0.35
        
        ax2.bar(x + width / 2, epochs_used, width, label='Utilisées', alpha=0.7, color='darkblue')
        
        ax2.set_xlabel('Étape')
        ax2.set_ylabel('Nombre d\'époques')
        ax2.set_title('Époques Prévues vs Utilisées')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stage_names, rotation=15)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(Path(CONFIG.OUTPUT_DIR) / "stage_comparison.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
