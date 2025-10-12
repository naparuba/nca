from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from matplotlib import pyplot as plt

from config import CONFIG
from train import ImprovedNCA, simulator, OptimizedNCAUpdater


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
        
        # Animation NCA seule
        self._save_single_gif(
                vis_data['nca_sequence'],
                vis_data['obstacle_mask'],
                stage_dir / f"animation_nca_étape_{stage_nb}.gif",
                f"Étape {stage_nb} - Prédiction NCA"
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
    
    
    def _save_single_gif(self, sequence: List[np.ndarray], obstacle_mask: np.ndarray,
                         filepath: Path, title: str):
        """Sauvegarde un GIF d'une séquence unique."""
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        
        def animate(frame):
            ax.clear()
            im = ax.imshow(sequence[frame], cmap='hot', vmin=0, vmax=1)
            ax.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax.set_title(f'{title} - t={frame}')
            ax.set_xticks([])
            ax.set_yticks([])
            return [im]
        
        
        ani = animation.FuncAnimation(fig, animate, frames=len(sequence), interval=200, blit=False)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()
    
    
    def create_curriculum_summary(self, global_metrics: Dict[str, Any]):
        """Crée un résumé visuel complet du curriculum d'apprentissage."""
        print("\n🎨 Génération du résumé visuel du curriculum...")
        
        # Graphique de progression globale
        self._plot_curriculum_progression(global_metrics)
        
        # Comparaison inter-étapes
        self._plot_stage_comparison(global_metrics)
        
        # Métriques de performance
        self._plot_performance_metrics(global_metrics)
        
        print("✅ Résumé visuel complet généré")
    
    
    def _plot_curriculum_progression(self, metrics: Dict[str, Any]):
        """Graphique de la progression globale du curriculum."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15))
        
        # Historique des pertes avec codes couleur par étape
        losses = metrics['global_history']['losses']
        stages = metrics['global_history']['stages']
        epochs = metrics['global_history']['epochs']
        
        stage_colors = {1: 'green', 2: 'orange', 3: 'red'}
        
        for stage_nb in [1, 2, 3]:
            stage_indices = [i for i, s in enumerate(stages) if s == stage_nb]
            stage_losses = [losses[i] for i in stage_indices]
            stage_epochs = [epochs[i] for i in stage_indices]
            
            if stage_losses:
                ax1.plot(stage_epochs, stage_losses,
                         color=stage_colors[stage_nb],
                         label=f'Étape {stage_nb}',
                         linewidth=2)
        
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Perte MSE')
        ax1.set_title('Progression du Curriculum d\'Apprentissage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Learning rate par étape
        for stage_nb in [1, 2, 3]:
            stage_history = metrics['stage_histories'][stage_nb]
            if stage_history['lr']:
                stage_epochs_local = [metrics['stage_start_epochs'].get(stage_nb, 0) + e
                                      for e in stage_history['epochs']]
                ax2.plot(stage_epochs_local, stage_history['lr'],
                         color=stage_colors[stage_nb],
                         label=f'LR Étape {stage_nb}',
                         linewidth=2)
        
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Évolution du Learning Rate par Étape')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Accélération du Learning Rate (dérivée seconde) pour détecter les changements d'accélération
        for stage_nb in [1, 2, 3]:
            stage_history = metrics['stage_histories'][stage_nb]
            if stage_history['lr'] and len(stage_history['lr']) > 2:  # Besoin d'au moins 3 points pour dérivée seconde
                # Calcul de la dérivée première (vitesse)
                lr_values = stage_history['lr']
                lr_velocity = []
                
                for i in range(1, len(lr_values)):
                    velocity = lr_values[i] - lr_values[i - 1]
                    lr_velocity.append(velocity)
                
                # Calcul de la dérivée seconde (accélération de l'accélération)
                lr_acceleration = []
                for i in range(1, len(lr_velocity)):
                    acceleration = lr_velocity[i] - lr_velocity[i - 1]
                    lr_acceleration.append(acceleration)
                
                # Époques correspondantes (on commence à l'époque 2 car on a besoin de 3 points pour la dérivée seconde)
                stage_epochs_acceleration = [metrics['stage_start_epochs'].get(stage_nb, 0) + e
                                             for e in stage_history['epochs'][2:]]
                
                if lr_acceleration:
                    ax3.plot(stage_epochs_acceleration, lr_acceleration,
                             color=stage_colors[stage_nb],
                             label=f'Accélération LR Étape {stage_nb}',
                             linewidth=2,
                             marker='o', markersize=3, alpha=0.7)
                    
                    # Ligne de référence à zéro pour identifier les changements d'accélération
                    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
                    
                    # Zone négative (décélération) en rouge transparent
                    negative_mask = [a < 0 for a in lr_acceleration]
                    if any(negative_mask):
                        ax3.fill_between(stage_epochs_acceleration, lr_acceleration, 0,
                                         where=negative_mask,
                                         alpha=0.2, color='red',
                                         label='Zone de décélération' if stage_nb == 1 else "")
                    
                    # Zone positive (accélération croissante) en vert transparent
                    positive_mask = [a > 0 for a in lr_acceleration]
                    if any(positive_mask):
                        ax3.fill_between(stage_epochs_acceleration, lr_acceleration, 0,
                                         where=positive_mask,
                                         alpha=0.2, color='green',
                                         label='Zone d\'accélération' if stage_nb == 1 else "")
                    
                    # Détection et marquage des points d'inflexion
                    inflection_points_epochs = []
                    inflection_points_values = []
                    
                    for i in range(1, len(lr_acceleration)):
                        # Point d'inflexion = changement de signe dans l'accélération
                        prev_accel = lr_acceleration[i - 1]
                        curr_accel = lr_acceleration[i]
                        
                        # Vérifier si on traverse zéro (changement de signe)
                        if (prev_accel > 0 and curr_accel < 0) or (prev_accel < 0 and curr_accel > 0):
                            # Filtre très léger pour éviter seulement le bruit extrême
                            if abs(prev_accel) > 1e-12 or abs(curr_accel) > 1e-12:
                                inflection_epoch = stage_epochs_acceleration[i]
                                inflection_value = curr_accel
                                inflection_points_epochs.append(inflection_epoch)
                                inflection_points_values.append(inflection_value)
                    
                    # Marquer les points d'inflexion sur le graphique
                    if inflection_points_epochs:
                        ax3.scatter(inflection_points_epochs, inflection_points_values,
                                    color=stage_colors[stage_nb],
                                    s=80, marker='X',
                                    edgecolors='black', linewidth=2,
                                    label=f'Points d\'inflexion Étape {stage_nb}' if stage_nb == 1 else "",
                                    zorder=5, alpha=0.9)
                        
                        # Annotations pour les points d'inflexion les plus significatifs
                        for i, (epoch, value) in enumerate(zip(inflection_points_epochs, inflection_points_values)):
                            if i < 3:  # Limite à 3 annotations par étape pour éviter l'encombrement
                                ax3.annotate(f'Inflexion\nÉ{epoch}',
                                             xy=(epoch, value),
                                             xytext=(10, 20 if value > 0 else -30),
                                             textcoords='offset points',
                                             fontsize=8,
                                             color=stage_colors[stage_nb],
                                             bbox=dict(boxstyle='round,pad=0.3',
                                                       facecolor='white',
                                                       edgecolor=stage_colors[stage_nb],
                                                       alpha=0.8),
                                             arrowprops=dict(arrowstyle='->',
                                                             connectionstyle='arc3,rad=0.2',
                                                             color=stage_colors[stage_nb],
                                                             alpha=0.7))
        
        ax3.set_xlabel('Époque')
        ax3.set_ylabel('Accélération LR (Δ²LR par époque²)')
        ax3.set_title('Accélération du Learning Rate - Points d\'Inflexion et Changements d\'Accélération')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Annotation explicative pour interpréter le graphique d'accélération avec points d'inflexion
        ax3.text(0.02, 0.98,
                 'Valeurs négatives = LR décélère (ralentissement qui s\'accélère)\n'
                 'Valeurs positives = LR accélère (accélération qui s\'intensifie)\n'
                 'Valeurs proches de 0 = Vitesse LR constante (accélération stable)\n'
                 'X = Points d\'inflexion (changements de dynamique du LR)\n'
                 'Les points d\'inflexion indiquent des changements de politique d\'optimisation',
                 transform=ax3.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(Path(CONFIG.OUTPUT_DIR) / "curriculum_progression.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    
    def _plot_stage_comparison(self, metrics: Dict[str, Any]):
        """Graphique de comparaison entre étapes."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        stages = [1, 2, 3]
        stage_names = ["Sans obstacles", "Un obstacle", "Obstacles multiples"]
        
        ax1.set_ylabel('Perte finale')
        ax1.set_title('Perte Finale par Étape')
        ax1.set_yscale('log')
        
        # Époques utilisées par étape
        epochs_used = [metrics['stage_metrics'][s]['epochs_trained'] for s in stages]
        epochs_planned = [CONFIG.STAGE_1_EPOCHS, CONFIG.STAGE_2_EPOCHS, CONFIG.STAGE_3_EPOCHS]
        
        x = np.arange(len(stages))
        width = 0.35
        
        ax2.bar(x - width / 2, epochs_planned, width, label='Prévues', alpha=0.7, color='lightblue')
        ax2.bar(x + width / 2, epochs_used, width, label='Utilisées', alpha=0.7, color='darkblue')
        
        ax2.set_xlabel('Étape')
        ax2.set_ylabel('Nombre d\'époques')
        ax2.set_title('Époques Prévues vs Utilisées')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stage_names, rotation=15)
        ax2.legend()
        
        # Temps de convergence
        convergence_times = []
        for stage_nb in stages:
            stage_losses = metrics['stage_metrics'][stage_nb]['loss_history']
            convergence_times.append(len(stage_losses))
        
        ax3.plot(stages, convergence_times, 'o-', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('Étape')
        ax3.set_ylabel('Époque de convergence')
        ax3.set_title('Vitesse de Convergence par Étape')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(CONFIG.OUTPUT_DIR) / "stage_comparison.png",
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    
    def _plot_performance_metrics(self, metrics: Dict[str, Any]):
        """Graphique des métriques de performance globales."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Résumé textuel des performances
        total_time = metrics['total_time_seconds']
        total_epochs = metrics['total_epochs_actual']
        final_loss = metrics['final_loss']
        
        summary_text = f"""
🎯 RÉSUMÉ ENTRAÎNEMENT MODULAIRE NCA

📊 STATISTIQUES GLOBALES:
   • Seed: {CONFIG.SEED}
   • Temps total: {total_time / 60:.1f} minutes ({total_time:.1f}s)
   • Époques totales: {total_epochs}
   • Perte finale: {final_loss:.6f}

🏆 PERFORMANCE PAR ÉTAPE:"""
        
        for stage_nb in [1, 2, 3]:
            stage_data = metrics['stage_metrics'][stage_nb]
            stage_name = {1: "Sans obstacles", 2: "Un obstacle", 3: "Obstacles multiples"}[stage_nb]
            
            summary_text += f"""
   • Étape {stage_nb} ({stage_name}):
     - Époques: {stage_data['epochs_trained']}
     - Perte finale: {stage_data['final_loss']:.6f}
     - Arrêt précoce: {'✅' if stage_data['early_stopped'] else '❌'}"""
        
        summary_text += f"""

📈 ARCHITECTURE:
   • Taille grille: {CONFIG.GRID_SIZE}x{CONFIG.GRID_SIZE}
   • Couches cachées: {CONFIG.HIDDEN_SIZE} neurones, {CONFIG.N_LAYERS} couches
   • Pas temporels NCA: {CONFIG.NCA_STEPS}
   • Taille batch: {CONFIG.BATCH_SIZE}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Résumé Performance Entraînement Modulaire NCA', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(Path(CONFIG.OUTPUT_DIR) / "performance_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
