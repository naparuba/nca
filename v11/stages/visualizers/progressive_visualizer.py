"""
Visualiseur progressif pour l'apprentissage modulaire avec intensités variables.
Migré depuis visualize_modular_progressive_obstacles_variable_intensity.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path


class ProgressiveVisualizer:
    """
    Système de visualisation avancé pour l'apprentissage modulaire avec intensités variables.
    Étend les fonctionnalités v7__ avec support des intensités variables.
    """
    
    def __init__(self, interactive: bool = False):
        self.interactive = interactive
        self.frame_data = {}  # Données par étape
        self.intensity_data = {}  # Données d'intensité par étape
        
    def visualize_stage_results(self, model, stage: int, simulator, cfg,
                              vis_seed: int = 123, source_intensity: Optional[float] = None) -> Dict[str, Any]:
        """
        Visualise les résultats d'une étape avec support intensité variable.
        
        Args:
            model: Modèle NCA entraîné
            stage: Numéro d'étape (1-4)
            simulator: Simulateur de diffusion
            cfg: Configuration
            vis_seed: Graine pour reproductibilité
            source_intensity: Intensité spécifique pour étape 4 (None = intensité standard)
            
        Returns:
            Dictionnaire avec données de visualisation étendues
        """
        print(f"\n🎨 Génération des visualisations pour l'étape {stage}...")
        
        # Génération de la séquence de test avec seed fixe
        torch.manual_seed(vis_seed)
        np.random.seed(vis_seed)
        
        # Génération adaptée selon l'étape
        if stage == 4 and source_intensity is not None:
            # Étape 4: utilise l'intensité spécifiée
            target_seq, source_mask, obstacle_mask, used_intensity = simulator.generate_stage_sequence(
                stage=4, n_steps=cfg.POSTVIS_STEPS, size=cfg.GRID_SIZE,
                seed=vis_seed, source_intensity=source_intensity)
        else:
            # Étapes 1-3: intensité standard
            target_seq, source_mask, obstacle_mask, used_intensity = simulator.generate_stage_sequence(
                stage=stage, n_steps=cfg.POSTVIS_STEPS, size=cfg.GRID_SIZE, seed=vis_seed)
            if used_intensity is None:
                used_intensity = cfg.DEFAULT_SOURCE_INTENSITY

        # Simulation NCA avec intensité appropriée
        nca_sequence = self._simulate_nca_with_intensity(model, target_seq[0],
                                                       source_mask, obstacle_mask,
                                                       used_intensity, cfg)
        
        # Données de visualisation étendues
        vis_data = {
            'stage': stage,
            'target_sequence': [t.detach().cpu().numpy() for t in target_seq],
            'nca_sequence': [t.detach().cpu().numpy() for t in nca_sequence],
            'source_mask': source_mask.detach().cpu().numpy(),
            'obstacle_mask': obstacle_mask.detach().cpu().numpy(),
            'source_intensity': used_intensity,  # NOUVEAU
            'vis_seed': vis_seed
        }
        
        # Création des visualisations avec intensité
        self._create_stage_animations_with_intensity(vis_data, cfg)
        self._create_stage_convergence_plot_with_intensity(vis_data, cfg)
        
        model.train()
        return vis_data
    
    def _simulate_nca_with_intensity(self, model, initial_grid, source_mask, obstacle_mask,
                                   source_intensity: float, cfg) -> List[torch.Tensor]:
        """
        Simule le NCA avec une intensité spécifique.
        """
        # Import dynamique pour éviter les dépendances circulaires
        try:
            from .. import OptimizedNCAUpdater, NCAUpdater
        except ImportError:
            # Fallback si import relatif échoue
            try:
                from __main__ import OptimizedNCAUpdater, NCAUpdater
            except ImportError:
                # Créer un updater basique si nécessaire
                class BasicNCAUpdater:
                    def __init__(self, model, device):
                        self.model = model
                        self.device = device
                    
                    def step(self, grid, source_mask, obstacle_mask, source_intensity=None):
                        # Version simplifiée pour la visualisation
                        with torch.no_grad():
                            new_grid = grid.clone()
                            if source_intensity is not None:
                                new_grid[source_mask] = source_intensity
                            else:
                                new_grid[source_mask] = cfg.SOURCE_INTENSITY
                        return new_grid
                
                OptimizedNCAUpdater = BasicNCAUpdater
        
        model.eval()
        updater = OptimizedNCAUpdater(model, cfg.DEVICE) if hasattr(cfg, 'USE_OPTIMIZATIONS') and cfg.USE_OPTIMIZATIONS else None
        
        if updater is None:
            # Fallback simple
            updater = type('SimpleUpdater', (), {
                'step': lambda self, grid, source_mask, obstacle_mask, source_intensity=None: self._simple_step(grid, source_mask, obstacle_mask, source_intensity, cfg)
            })()
            updater._simple_step = lambda grid, source_mask, obstacle_mask, source_intensity, cfg: self._simple_nca_step(grid, source_mask, obstacle_mask, source_intensity, cfg)
        
        # Simulation NCA
        nca_sequence = []
        grid_pred = torch.zeros_like(initial_grid)
        grid_pred[source_mask] = source_intensity
        nca_sequence.append(grid_pred.clone())
        
        with torch.no_grad():
            for _ in range(cfg.POSTVIS_STEPS):
                if hasattr(updater, 'step'):
                    grid_pred = updater.step(grid_pred, source_mask, obstacle_mask, source_intensity)
                else:
                    grid_pred = self._simple_nca_step(grid_pred, source_mask, obstacle_mask, source_intensity, cfg)
                nca_sequence.append(grid_pred.clone())
        
        return nca_sequence
    
    def _simple_nca_step(self, grid, source_mask, obstacle_mask, source_intensity, cfg):
        """Version simplifiée d'un pas NCA pour la visualisation."""
        new_grid = grid.clone()
        if source_intensity is not None:
            new_grid[source_mask] = source_intensity
        return new_grid
    
    def _create_stage_animations_with_intensity(self, vis_data: Dict[str, Any], cfg):
        """Crée les animations GIF pour une étape avec affichage d'intensité."""
        # Import local pour éviter les dépendances circulaires
        from .intensity_animator import IntensityAwareAnimator
        
        stage = vis_data['stage']
        intensity = vis_data['source_intensity']
        
        # Création du répertoire de sortie
        if stage == 4:
            stage_dir = Path(cfg.OUTPUT_DIR) / f"stage_{stage}"
            intensity_suffix = f"_I_{intensity:.3f}"
        else:
            stage_dir = Path(cfg.OUTPUT_DIR) / f"stage_{stage}"
            intensity_suffix = ""
            
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Animation comparative avec intensité
        animator = IntensityAwareAnimator()
        
        comparison_path = stage_dir / f"animation_comparaison_étape_{stage}{intensity_suffix}.gif"
        animator.create_comparison_with_intensity(
            vis_data['target_sequence'],
            vis_data['nca_sequence'],
            vis_data['obstacle_mask'],
            intensity,
            comparison_path
        )
        
        # Animation NCA seule avec intensité
        nca_path = stage_dir / f"animation_nca_étape_{stage}{intensity_suffix}.gif"
        animator.create_intensity_labeled_gif(
            vis_data['nca_sequence'],
            vis_data['obstacle_mask'],
            intensity,
            nca_path,
            f"Étape {stage} - NCA"
        )
        
        print(f"✅ Animations étape {stage} (I={intensity:.3f}) sauvegardées dans {stage_dir}")
    
    def _create_stage_convergence_plot_with_intensity(self, vis_data: Dict[str, Any], cfg):
        """Crée le graphique de convergence pour une étape avec intensité."""
        stage = vis_data['stage']
        intensity = vis_data['source_intensity']
        
        if stage == 4:
            stage_dir = Path(cfg.OUTPUT_DIR) / f"stage_{stage}"
            intensity_suffix = f"_I_{intensity:.3f}"
        else:
            stage_dir = Path(cfg.OUTPUT_DIR) / f"stage_{stage}"
            intensity_suffix = ""
            
        target_seq = vis_data['target_sequence']
        nca_seq = vis_data['nca_sequence']
        
        # Calcul de l'erreur temporelle
        errors = []
        for t in range(min(len(target_seq), len(nca_seq))):
            error = np.mean((target_seq[t] - nca_seq[t]) ** 2)
            errors.append(error)
        
        # Graphique avec intensité dans le titre
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(errors, 'b-', linewidth=2, label='Erreur MSE')
        
        # Seuil de convergence - utilise la nouvelle méthode au lieu de l'attribut dupliqué
        threshold = cfg.get_convergence_threshold(stage, getattr(cfg, 'stage_manager', None))
        ax.axhline(y=threshold, color='r', linestyle='--',
                  label=f'Seuil convergence étape {stage}')
        
        ax.set_xlabel('Pas de temps')
        ax.set_ylabel('Erreur MSE')
        ax.set_title(f'Convergence Étape {stage} (I={intensity:.3f}) - Seed {vis_data["vis_seed"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        convergence_path = stage_dir / f"convergence_étape_{stage}{intensity_suffix}.png"
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graphique de convergence étape {stage} (I={intensity:.3f}) sauvegardé: {convergence_path}")
    
    def create_intensity_comparison_grid(self, model, simulator, cfg,
                                       intensity_samples: Optional[List[float]] = None,
                                       vis_seed: int = 123):
        """
        Crée une grille comparative pour différentes intensités (Étape 4).
        """
        if intensity_samples is None:
            intensity_samples = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        print("🎨 Génération de la grille comparative d'intensités...")
        
        fig, axes = plt.subplots(2, len(intensity_samples), figsize=(4*len(intensity_samples), 8))
        
        if len(intensity_samples) == 1:
            axes = axes.reshape(-1, 1)
        
        # Initialisation des variables pour les colorbars
        im1 = None
        im2 = None
        
        for i, intensity in enumerate(intensity_samples):
            # Génération avec intensité spécifique
            vis_data = self.visualize_stage_results(model, stage=4, simulator=simulator, cfg=cfg,
                                                  vis_seed=vis_seed, source_intensity=intensity)
            
            # État initial
            im1 = axes[0, i].imshow(vis_data['nca_sequence'][0], cmap='hot', vmin=0, vmax=1)
            axes[0, i].contour(vis_data['obstacle_mask'], levels=[0.5], colors='cyan', linewidths=2)
            axes[0, i].set_title(f'Initial (I={intensity:.3f})', fontweight='bold')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # État final
            im2 = axes[1, i].imshow(vis_data['nca_sequence'][-1], cmap='hot', vmin=0, vmax=1)
            axes[1, i].contour(vis_data['obstacle_mask'], levels=[0.5], colors='cyan', linewidths=2)
            axes[1, i].set_title(f'Final (I={intensity:.3f})', fontweight='bold')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        # Ajout des colorbars seulement si on a des images
        if im1 is not None and im2 is not None:
            fig.colorbar(im1, ax=axes[0, :], shrink=0.6, aspect=20)
            fig.colorbar(im2, ax=axes[1, :], shrink=0.6, aspect=20)
        
        plt.suptitle('Comparaison d\'Intensités - Étape 4', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(Path(cfg.OUTPUT_DIR) / "intensity_comparison_grid.png",
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Grille comparative d'intensités générée")
    
    def visualize_intensity_curriculum(self, stage_4_metrics: Dict[str, Any], cfg):
        """
        Visualise le curriculum d'intensité de l'étape 4.
        """
        # Import local pour éviter les dépendances circulaires
        from .metrics_plotter import VariableIntensityMetricsPlotter
        
        print("🎨 Génération des graphiques de curriculum d'intensité...")
        
        output_dir = Path(cfg.OUTPUT_DIR)
        metrics_plotter = VariableIntensityMetricsPlotter()
        
        if 'intensity_history' in stage_4_metrics:
            metrics_plotter.plot_intensity_distribution(
                stage_4_metrics['intensity_history'], output_dir)
        
        if 'performance_by_intensity' in stage_4_metrics:
            metrics_plotter.plot_performance_by_intensity_range(
                stage_4_metrics['performance_by_intensity'], output_dir)
        
        if 'convergence_analysis' in stage_4_metrics:
            metrics_plotter.plot_convergence_analysis_by_intensity(
                stage_4_metrics['convergence_analysis'], output_dir)

    def create_curriculum_summary_extended(self, global_metrics: Dict[str, Any], cfg):
        """Crée un résumé visuel complet du curriculum d'apprentissage étendu (v8__)."""
        print("\n🎨 Génération du résumé visuel du curriculum étendu...")
        
        # Graphique de progression globale étendu (4 étapes)
        self._plot_curriculum_progression_extended(global_metrics, cfg)
        
        # Comparaison inter-étapes étendue
        self._plot_stage_comparison_extended(global_metrics, cfg)
        
        # Métriques de performance étendues
        self._plot_performance_metrics_extended(global_metrics, cfg)
        
        print("✅ Résumé visuel complet étendu généré")
    
    def _plot_curriculum_progression_extended(self, metrics: Dict[str, Any], cfg):
        """Graphique de la progression globale du curriculum étendu (4 étapes)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Historique des pertes avec codes couleur par étape (4 étapes)
        if 'global_history' in metrics:
            losses = metrics['global_history']['losses']
            stages = metrics['global_history']['stages']
            epochs = metrics['global_history']['epochs']
            
            stage_colors = {1: 'green', 2: 'orange', 3: 'red', 4: 'purple'}
            stage_names = {1: 'Sans obstacles', 2: 'Un obstacle', 3: 'Obstacles multiples', 4: 'Intensités variables'}
            
            for stage in [1, 2, 3, 4]:
                stage_indices = [i for i, s in enumerate(stages) if s == stage]
                if stage_indices:
                    stage_losses = [losses[i] for i in stage_indices]
                    stage_epochs = [epochs[i] for i in stage_indices]
                    
                    ax1.plot(stage_epochs, stage_losses,
                            color=stage_colors[stage],
                            label=f'Étape {stage} ({stage_names[stage]})',
                            linewidth=2)
            
            # Seuils de convergence - utilise la nouvelle méthode au lieu de l'attribut dupliqué
            for stage in [1, 2, 3, 4]:
                threshold = cfg.get_convergence_threshold(stage, getattr(cfg, 'stage_manager', None))
                if threshold != 0.05:  # Seulement si différent de la valeur par défaut
                    ax1.axhline(y=threshold, color=stage_colors[stage],
                               linestyle='--', alpha=0.7,
                               label=f'Seuil étape {stage}')
        
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Perte MSE')
        ax1.set_title('Progression du Curriculum d\'Apprentissage Étendu (4 Étapes)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Learning rate par étape
        if 'stage_histories' in metrics:
            stage_colors = {1: 'green', 2: 'orange', 3: 'red', 4: 'purple'}
            for stage in [1, 2, 3, 4]:
                if stage in metrics['stage_histories']:
                    stage_history = metrics['stage_histories'][stage]
                    if 'lr' in stage_history and stage_history['lr']:
                        stage_start = metrics.get('stage_start_epochs', {}).get(stage, 0)
                        stage_epochs_local = [stage_start + e for e in stage_history['epochs']]
                        ax2.plot(stage_epochs_local, stage_history['lr'],
                                color=stage_colors[stage],
                                label=f'LR Étape {stage}',
                                linewidth=2)
        
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Évolution du Learning Rate par Étape')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(Path(cfg.OUTPUT_DIR) / "curriculum_progression_extended.png",
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_stage_comparison_extended(self, metrics: Dict[str, Any], cfg):
        """Graphique de comparaison entre les 4 étapes."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        stages = [1, 2, 3, 4]
        stage_names = ["Sans obstacles", "Un obstacle", "Obstacles multiples", "Intensités variables"]
        stage_colors = ['green', 'orange', 'red', 'purple']
        
        if 'stage_metrics' in metrics:
            # Pertes finales par étape
            final_losses = []
            convergence_status = []
            epochs_used = []
            
            for s in stages:
                if s in metrics['stage_metrics']:
                    final_losses.append(metrics['stage_metrics'][s].get('final_loss', 0))
                    convergence_status.append(metrics['stage_metrics'][s].get('convergence_met', False))
                    epochs_used.append(metrics['stage_metrics'][s].get('epochs_trained', 0))
                else:
                    final_losses.append(0)
                    convergence_status.append(False)
                    epochs_used.append(0)
            
            # Graphique des pertes finales
            bars = ax1.bar(stage_names, final_losses, color=stage_colors, alpha=0.7)
            for i, (bar, converged) in enumerate(zip(bars, convergence_status)):
                if converged:
                    bar.set_edgecolor('darkgreen')
                    bar.set_linewidth(3)
            
            ax1.set_ylabel('Perte finale')
            ax1.set_title('Perte Finale par Étape')
            ax1.set_yscale('log')
            plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')
            
            # Époques utilisées par étape
            epochs_planned = [
                getattr(cfg, 'STAGE_1_EPOCHS', 0),
                getattr(cfg, 'STAGE_2_EPOCHS', 0),
                getattr(cfg, 'STAGE_3_EPOCHS', 0),
                getattr(cfg, 'STAGE_4_EPOCHS', 0)
            ]
            
            x = np.arange(len(stages))
            width = 0.35
            
            ax2.bar(x - width/2, epochs_planned, width, label='Prévues', alpha=0.7, color='lightblue')
            ax2.bar(x + width/2, epochs_used, width, label='Utilisées', alpha=0.7, color='darkblue')
            
            ax2.set_xlabel('Étape')
            ax2.set_ylabel('Nombre d\'époques')
            ax2.set_title('Époques Prévues vs Utilisées')
            ax2.set_xticks(x)
            ax2.set_xticklabels(stage_names, rotation=15, ha='right')
            ax2.legend()
            
            # Temps de convergence
            convergence_times = []
            for stage in stages:
                if stage in metrics['stage_metrics'] and 'loss_history' in metrics['stage_metrics'][stage]:
                    stage_losses = metrics['stage_metrics'][stage]['loss_history']
                    threshold = cfg.get_convergence_threshold(stage, getattr(cfg, 'stage_manager', None))
                    
                    convergence_epoch = None
                    for i, loss in enumerate(stage_losses):
                        if loss < threshold:
                            convergence_epoch = i
                            break
                    
                    convergence_times.append(convergence_epoch if convergence_epoch else len(stage_losses))
                else:
                    convergence_times.append(0)
            
            ax3.plot(stages, convergence_times, 'o-', linewidth=2, markersize=8, color='purple')
            ax3.set_xlabel('Étape')
            ax3.set_ylabel('Époque de convergence')
            ax3.set_title('Vitesse de Convergence par Étape')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(stages)
            
            # Efficacité (convergence / époques utilisées)
            efficiency = []
            for i in range(len(stages)):
                if epochs_used[i] > 0:
                    eff = (1.0 if convergence_status[i] else 0.5) / epochs_used[i]
                    efficiency.append(eff)
                else:
                    efficiency.append(0)
            
            ax4.bar(stage_names, efficiency, color=stage_colors, alpha=0.7)
            ax4.set_ylabel('Efficacité (convergence/époque)')
            ax4.set_title('Efficacité d\'Apprentissage par Étape')
            plt.setp(ax4.get_xticklabels(), rotation=15, ha='right')
        
        plt.tight_layout()
        plt.savefig(Path(cfg.OUTPUT_DIR) / "stage_comparison_extended.png",
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics_extended(self, metrics: Dict[str, Any], cfg):
        """Graphique des métriques de performance globales étendues."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Résumé textuel des performances étendu
        total_time = metrics.get('total_time_seconds', 0)
        total_epochs = metrics.get('total_epochs_actual', 0)
        all_converged = metrics.get('all_stages_converged', False)
        final_loss = metrics.get('final_loss', 0)
        
        summary_text = f"""
🎯 RÉSUMÉ ENTRAÎNEMENT MODULAIRE NCA v8__ (INTENSITÉS VARIABLES)

📊 STATISTIQUES GLOBALES:
   • Seed: {cfg.SEED}
   • Temps total: {total_time/60:.1f} minutes ({total_time:.1f}s)
   • Époques totales: {total_epochs}/{cfg.TOTAL_EPOCHS}
   • Toutes étapes convergées: {'✅ OUI' if all_converged else '❌ NON'}
   • Perte finale: {final_loss:.6f}

🏆 PERFORMANCE PAR ÉTAPE:"""
        
        stage_names = {1: "Sans obstacles", 2: "Un obstacle", 3: "Obstacles multiples", 4: "Intensités variables"}
        
        for stage in [1, 2, 3, 4]:
            if 'stage_metrics' in metrics and stage in metrics['stage_metrics']:
                stage_data = metrics['stage_metrics'][stage]
                stage_name = stage_names[stage]
                
                summary_text += f"""
   • Étape {stage} ({stage_name}):
     - Époques: {stage_data.get('epochs_trained', 0)} (convergée: {'✅' if stage_data.get('convergence_met', False) else '❌'})
     - Perte finale: {stage_data.get('final_loss', 0):.6f}
     - Arrêt précoce: {'✅' if stage_data.get('early_stopped', False) else '❌'}"""
        
        # Ajout des statistiques d'intensité pour l'étape 4
        if 'intensity_metrics' in metrics:
            intensity_stats = metrics['intensity_metrics'].get('statistics', {})
            summary_text += f"""

🔥 STATISTIQUES INTENSITÉS (ÉTAPE 4):
   • Intensités utilisées: {intensity_stats.get('count', 0)}
   • Intensité moyenne: {intensity_stats.get('mean', 0):.3f}
   • Écart-type: {intensity_stats.get('std', 0):.3f}
   • Plage: [{intensity_stats.get('min', 0):.3f}, {intensity_stats.get('max', 0):.3f}]"""
        
        summary_text += f"""

⚙️  CONFIGURATION:
   • Curriculum learning: {'✅' if getattr(cfg, 'ENABLE_CURRICULUM', False) else '❌'}
   • Intensités variables: {'✅' if getattr(cfg, 'VARIABLE_INTENSITY_TRAINING', False) else '❌'}
   • Cache optimisé: {'✅' if getattr(cfg, 'USE_SEQUENCE_CACHE', False) else '❌'}
   • Updater vectorisé: {'✅' if getattr(cfg, 'USE_VECTORIZED_PATCHES', False) else '❌'}

📈 ARCHITECTURE:
   • Taille grille: {cfg.GRID_SIZE}x{cfg.GRID_SIZE}
   • Couches cachées: {cfg.HIDDEN_SIZE} neurones, {cfg.N_LAYERS} couches
   • Pas temporels NCA: {cfg.NCA_STEPS}
   • Taille batch: {cfg.BATCH_SIZE}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Résumé Performance Entraînement Modulaire NCA v8__ (Intensités Variables)',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(Path(cfg.OUTPUT_DIR) / "performance_summary_extended.png",
                   dpi=150, bbox_inches='tight')
        plt.close()
