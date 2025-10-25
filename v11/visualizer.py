from pathlib import Path
from typing import Dict, Any, List, TYPE_CHECKING
import json

import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt

from config import CONFIG
from nca_model import NCA
from stage_manager import STAGE_MANAGER
from stages.base_stage import REALITY_LAYER
from torched import get_MSELoss

if TYPE_CHECKING:
    from stages.base_stage import BaseStage


class ProgressiveVisualizer:
    """
    Système de visualisation avancé pour l'apprentissage modulaire.
    Génère des animations et graphiques comparatifs par étape.
    """
    
    def __init__(self):
        self._loss_fn = get_MSELoss()
    
    
    def check_configuration_already_evaluated(self, stage_nb, n_layers, hidden_size, nb_epochs_trained):
        # type: (int, int, int, int) -> bool
        """
        Vérifie si une configuration spécifique a déjà été évaluée et sauvegardée.
        
        Cette méthode permet d'éviter de ré-entraîner une configuration déjà testée,
        ce qui économise du temps de calcul lors d'expérimentations multiples.
        
        Args:
            stage_nb: Numéro du stage à vérifier
            n_layers: Nombre de couches du modèle
            hidden_size: Taille de la couche cachée
            nb_epochs_trained: Nombre d'époques d'entraînement
        
        Returns:
            True si la configuration existe déjà dans le fichier JSON, False sinon
        """
        perf_file = Path(CONFIG.OUTPUT_DIR) / CONFIG.PERFORMANCE_FILE
        
        # Si le fichier n'existe pas, aucune configuration n'a été évaluée
        if not perf_file.exists():
            return False
        
        try:
            # Charger les données existantes
            with open(perf_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Construire les clés de navigation dans la structure JSON
            stage_key = f"stage_{stage_nb}"
            layers_key = str(n_layers)
            hidden_key = str(hidden_size)
            epochs_key = str(nb_epochs_trained)
            
            # Vérifier si toute la chaîne de clés existe
            if stage_key in data:
                if layers_key in data[stage_key]:
                    if hidden_key in data[stage_key][layers_key]:
                        if epochs_key in data[stage_key][layers_key][hidden_key]:
                            # Configuration trouvée
                            return True
            
            return False
        
        except (json.JSONDecodeError, KeyError):
            # En cas d'erreur de lecture, on considère que la config n'existe pas
            return False
    
    
    def evaluate_model_stage(self, model, stage):
        # type: (NCA, BaseStage) -> None
        
        stage_nb = stage.get_stage_nb()
        
        print(f"\n🔍 Évaluation du modèle à l'étape {stage_nb}...")
        
        # Génération de la séquence de test avec seed fixe
        torch.manual_seed(CONFIG.VISUALIZATION_SEED)
        np.random.seed(CONFIG.VISUALIZATION_SEED)
        
        total_loss = torch.tensor(0.0, device=CONFIG.DEVICE)
        
        losses = [] # type: List[float]
        
        # Nouvelles métriques par couche
        temp_correct_cells_pct = []  # type: List[float]  # % de cases correctes (vide/chaleur) pour température
        temp_heat_ratio = []  # type: List[float]  # Rapport de chaleur totale (sum NCA / sum reference)
        obstacle_correct_cells_pct = []  # type: List[float]  # % de cases correctes pour obstacles
        
        for nb_evaluation in range(CONFIG.NB_EPOCHS_FOR_EVALUATION):
            reality_worlds, nca_temporal_sequence = self.generate_and_run_one_sequence(model, stage)
            
            for temporal_step in range(1, len(nca_temporal_sequence)):
                grid_pred = nca_temporal_sequence[temporal_step]
                target = reality_worlds[temporal_step].get_as_tensor()
                step_loss = self._loss_fn(grid_pred, target)
                losses.append(step_loss.item())
                total_loss = total_loss + step_loss
                
                # === MÉTRIQUES TEMPÉRATURE ===
                
                # 1. Pourcentage de cases correctes (vide vs chaleur)
                # On considère qu'une case est "vide" si < 1% de SOURCE_INTENSITY, sinon elle a de la "chaleur"
                threshold = CONFIG.SOURCE_INTENSITY * 0.01
                target_temp = target[REALITY_LAYER.TEMPERATURE]
                pred_temp = grid_pred[REALITY_LAYER.TEMPERATURE]
                
                # Classification binaire : vide (0) ou chaleur (1)
                target_has_heat = (target_temp >= threshold).float()
                pred_has_heat = (pred_temp >= threshold).float()
                
                # Pourcentage de cases correctement classifiées
                correct_temp_cells = (target_has_heat == pred_has_heat).float().mean().item()
                temp_correct_cells_pct.append(correct_temp_cells * 100.0)  # En pourcentage
                
                # 2. Rapport de chaleur totale (somme des chaleurs)
                # Évite la division par zéro : si la référence est vide, on met un ratio à 0
                sum_target_heat = target_temp.sum().item()
                sum_pred_heat = pred_temp.sum().item()
                if sum_target_heat > 0:
                    heat_ratio = sum_pred_heat / sum_target_heat
                else:
                    heat_ratio = 0.0
                temp_heat_ratio.append(heat_ratio * 100.0)  # En pourcentage
                
                # === MÉTRIQUES OBSTACLES ===
                
                # Pourcentage de cases avec le bon état d'obstacle
                # Les obstacles sont soit 0 (pas d'obstacle) soit OBSTACLE_FULL_BLOCK_VALUE (obstacle)
                target_obstacles = target[REALITY_LAYER.OBSTACLE]
                pred_obstacles = grid_pred[REALITY_LAYER.OBSTACLE]
                
                # On considère qu'un obstacle est présent si >= 50% de OBSTACLE_FULL_BLOCK_VALUE
                obstacle_threshold = CONFIG.OBSTACLE_FULL_BLOCK_VALUE * 0.5
                target_has_obstacle = (target_obstacles >= obstacle_threshold).float()
                pred_has_obstacle = (pred_obstacles >= obstacle_threshold).float()
                
                # Pourcentage de cases correctement classifiées
                correct_obstacle_cells = (target_has_obstacle == pred_has_obstacle).float().mean().item()
                obstacle_correct_cells_pct.append(correct_obstacle_cells * 100.0)  # En pourcentage
        
        # Filtrage des outliers : on garde les 80% centraux (retire 10% top et 10% bottom)
        # Tri des losses pour identifier les percentiles
        sorted_losses = sorted(losses)
        n_total = len(sorted_losses)
        
        # Calcul des indices pour garder les 80% centraux
        # On retire 10% du bas et 10% du haut
        lower_cutoff_idx = int(n_total * 0.10)  # Index de début (10% bas)
        upper_cutoff_idx = int(n_total * 0.90)  # Index de fin (90%, donc on vire 10% haut)
        
        # Extraction des 80% centraux
        filtered_losses = sorted_losses[lower_cutoff_idx:upper_cutoff_idx]
        n_filtered = len(filtered_losses)
        
        # Calcul des statistiques sur les données filtrées
        avg_losses = sum(filtered_losses) / n_filtered if n_filtered > 0 else 0.0
        std_dev_losses = np.std(filtered_losses) if n_filtered > 0 else 0.0
        
        # === CALCUL DES STATISTIQUES POUR LES NOUVELLES MÉTRIQUES (SANS FILTRAGE DES OUTLIERS) ===
        
        # Température - % cases correctes
        avg_temp_correct_pct = np.mean(temp_correct_cells_pct)
        std_temp_correct_pct = np.std(temp_correct_cells_pct)
        
        # Température - Rapport chaleur totale
        avg_temp_heat_ratio = np.mean(temp_heat_ratio)
        std_temp_heat_ratio = np.std(temp_heat_ratio)
        
        # Obstacles - % cases correctes
        avg_obstacle_correct_pct = np.mean(obstacle_correct_cells_pct)
        std_obstacle_correct_pct = np.std(obstacle_correct_cells_pct)
        
        print(f"✅ Évaluation étape {stage_nb} terminée.")
        print(f"   Total loss (all): {total_loss.item():.6f}")
        print(f"   Données filtrées: {n_filtered}/{n_total} ({n_filtered/n_total*100:.1f}% conservés)")
        print(f"   Outliers retirés: {lower_cutoff_idx} (low) + {n_total - upper_cutoff_idx} (high)")
        print(f"   Avg Loss (80% centraux): {avg_losses:.6f}")
        print(f"   Std Dev (80% centraux): {std_dev_losses:.6f}")
        
        # Sauvegarde des performances dans le fichier JSON
        self._save_evaluation_performance(
            stage_nb=stage_nb,
            n_layers=CONFIG.N_LAYERS,
            hidden_size=CONFIG.HIDDEN_SIZE,
            nb_epochs_trained=CONFIG.NB_EPOCHS_BY_STAGE,
            total_loss=total_loss.item(),
            avg_loss=avg_losses,
            std_dev=std_dev_losses,
            nb_evaluations=CONFIG.NB_EPOCHS_FOR_EVALUATION,
            # Nouvelles métriques
            avg_temp_correct_pct=avg_temp_correct_pct,
            std_temp_correct_pct=std_temp_correct_pct,
            avg_temp_heat_ratio=avg_temp_heat_ratio,
            std_temp_heat_ratio=std_temp_heat_ratio,
            avg_obstacle_correct_pct=avg_obstacle_correct_pct,
            std_obstacle_correct_pct=std_obstacle_correct_pct
        )
    
    
    def _save_evaluation_performance(self, stage_nb, n_layers, hidden_size, nb_epochs_trained, total_loss, avg_loss, std_dev, nb_evaluations, avg_temp_correct_pct, std_temp_correct_pct, avg_temp_heat_ratio, std_temp_heat_ratio, avg_obstacle_correct_pct, std_obstacle_correct_pct):
        # type: (int, int, int, int, float, float, float, int, float, float, float, float, float, float) -> None
        """
        Sauvegarde les performances d'évaluation dans un fichier JSON structuré.
        
        Structure du JSON:
        {
            "stage_1": {
                "3": {  # n_layers
                    "128": {  # hidden_size
                        "50": {  # nb_epochs_trained
                            "total_loss": 0.123,
                            "avg_loss": 0.0045,
                            "std_dev": 0.001,
                            "nb_evaluations": 50,
                            "metrics_by_layer": {
                                "temperature": {
                                    "avg_correct_cells_pct": 95.5,
                                    "std_correct_cells_pct": 2.3,
                                    "avg_heat_ratio_pct": 98.2,
                                    "std_heat_ratio_pct": 1.5
                                },
                                "obstacles": {
                                    "avg_correct_cells_pct": 99.8,
                                    "std_correct_cells_pct": 0.2
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Si le fichier existe déjà, on charge les données existantes et on met à jour
        uniquement les valeurs concernées sans perdre les autres runs.
        """
        perf_file = Path(CONFIG.OUTPUT_DIR) / CONFIG.PERFORMANCE_FILE
        
        # Charger les données existantes si le fichier existe
        if perf_file.exists():
            with open(perf_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
        
        # Créer les clés de structure si elles n'existent pas
        stage_key = f"stage_{stage_nb}"
        layers_key = str(n_layers)
        hidden_key = str(hidden_size)
        epochs_key = str(nb_epochs_trained)
        
        if stage_key not in data:
            data[stage_key] = {}
        if layers_key not in data[stage_key]:
            data[stage_key][layers_key] = {}
        if hidden_key not in data[stage_key][layers_key]:
            data[stage_key][layers_key][hidden_key] = {}
        
        # Mettre à jour les performances pour cette configuration
        data[stage_key][layers_key][hidden_key][epochs_key] = {
            "total_loss": total_loss,
            "avg_loss": avg_loss,
            "std_dev": std_dev,
            "nb_evaluations": nb_evaluations,
            "metrics_by_layer": {
                "temperature": {
                    "avg_correct_cells_pct": avg_temp_correct_pct,
                    "std_correct_cells_pct": std_temp_correct_pct,
                    "avg_heat_ratio_pct": avg_temp_heat_ratio,
                    "std_heat_ratio_pct": std_temp_heat_ratio
                },
                "obstacles": {
                    "avg_correct_cells_pct": avg_obstacle_correct_pct,
                    "std_correct_cells_pct": std_obstacle_correct_pct
                }
            }
        }
        
        # Sauvegarder le fichier JSON avec indentation pour lisibilité
        perf_file.parent.mkdir(parents=True, exist_ok=True)
        with open(perf_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Performances sauvegardées dans {perf_file}")
    
    
    def generate_and_run_one_sequence(self, model, stage):
        # type: (NCA, BaseStage) -> (List, List)
        simulation_temporal_sequence = stage.generate_simulation_temporal_sequence(n_steps=CONFIG.POSTVIS_STEPS, size=CONFIG.GRID_SIZE)
        
        # Prédiction du modèle
        model.eval()
        
        # Simulation NCA avec torch.no_grad() pour éviter le gradient
        reality_worlds = simulation_temporal_sequence.get_reality_worlds()
        source_mask = simulation_temporal_sequence.get_source_mask()
        obstacle_mask = simulation_temporal_sequence.get_obstacle_mask()
        
        # Run model
        nca_temporal_sequence = []
        world_nca_prediction = torch.zeros_like(reality_worlds[0].get_as_tensor())  # start with the same start as reality
        # On accède à la couche température (REALITY_LAYER.TEMPERATURE = 0) avant d'appliquer le masque
        world_nca_prediction[REALITY_LAYER.TEMPERATURE][source_mask] = CONFIG.SOURCE_INTENSITY
        world_nca_prediction[REALITY_LAYER.OBSTACLE][obstacle_mask] = CONFIG.OBSTACLE_FULL_BLOCK_VALUE  # configure the obstacles
        nca_temporal_sequence.append(world_nca_prediction.clone())
        
        with torch.no_grad():  # Désactive le calcul de gradient pour les visualisations
            for _ in range(CONFIG.POSTVIS_STEPS):
                world_nca_prediction = model.run_step(world_nca_prediction, source_mask)  # , obstacle_mask)
                nca_temporal_sequence.append(world_nca_prediction.clone())
        
        return reality_worlds, nca_temporal_sequence
    
    
    # Visualise les résultats d'une étape spécifique
    def visualize_stage_results(self, model, stage):
        # type: (NCA, BaseStage) -> None
        
        stage_nb = stage.get_stage_nb()
        
        print(f"\n🎨 Génération des visualisations pour l'étape {stage_nb}...")
        
        # Génération de la séquence de test avec seed fixe
        torch.manual_seed(CONFIG.VISUALIZATION_SEED)
        np.random.seed(CONFIG.VISUALIZATION_SEED)
        
        reality_worlds, nca_temporal_sequence = self.generate_and_run_one_sequence(model, stage)
        
        # .detach() pour sécurité
        vis_data = {
            'stage_nb':              stage_nb,
            'reality_worlds':        [t.get_as_tensor().detach().cpu().numpy() for t in reality_worlds],
            'nca_temporal_sequence': [t.detach().cpu().numpy() for t in nca_temporal_sequence],
        }
        
        # Sauvegarde des animations
        self._create_stage_animations(vis_data)
        
        model.train()
        return
    
    
    def _create_stage_animations(self, vis_data):
        # type: (Dict[str, Any]) -> None
        
        stage_nb = vis_data['stage_nb']
        stage_dir = Path(CONFIG.OUTPUT_DIR) / f"stage_{stage_nb}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Animation comparative
        self._save_comparison_gif(
                vis_data['reality_worlds'],
                vis_data['nca_temporal_sequence'],
                stage_dir / f"animation_comparaison_étape_{stage_nb}.gif"
        )
        
        print(f"✅ Animations étape {stage_nb} sauvegardées dans {stage_dir}")
    
    
    @staticmethod
    def _save_comparison_gif(reality_worlds, nca_temporal_sequence, filepath):
        # type: (List[np.ndarray], List[np.ndarray], Path) -> None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Cible - On affiche uniquement la couche température (REALITY_LAYER.TEMPERATURE = 0)
            im1 = ax1.imshow(reality_worlds[frame][REALITY_LAYER.TEMPERATURE], cmap='hot', vmin=0, vmax=1)
            ax1.contour(reality_worlds[frame][REALITY_LAYER.OBSTACLE], levels=[0.5], colors='cyan', linewidths=2)
            ax1.set_title(f'Cible - t={frame}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # NCA - On affiche uniquement la couche température (REALITY_LAYER.TEMPERATURE = 0)
            im2 = ax2.imshow(nca_temporal_sequence[frame][REALITY_LAYER.TEMPERATURE], cmap='hot', vmin=0, vmax=1)
            ax2.contour(nca_temporal_sequence[frame][REALITY_LAYER.OBSTACLE], levels=[0.5], colors='cyan', linewidths=2)
            ax2.set_title(f'NCA - t={frame}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            return [im1, im2]
        
        
        n_frames = min(len(reality_worlds), len(nca_temporal_sequence))
        ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200, blit=False)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()
    
    
    def create_curriculum_summary(self):
        print("\n🎨 Génération du résumé visuel du curriculum...")
        
        # Graphique de progression globale
        self._plot_curriculum_progression()
        
        print("✅ Résumé visuel complet généré")
    
    
    def plot_performance_comparison(self):
        # type: () -> None
        """
        Génère un graphique de comparaison des performances depuis le fichier JSON.
        
        Affiche un bar plot avec:
        - X: Labels combinés "S{stage}_L{layers}_H{hidden}_E{epochs}"
        - Y: avg_loss avec error bars (std_dev)
        - Background coloré par stage
        - Annotations pour la meilleure configuration
        """
        print("\n📊 Génération du graphique de comparaison des performances...")
        
        perf_file = Path(CONFIG.OUTPUT_DIR) / CONFIG.PERFORMANCE_FILE
        
        if not perf_file.exists():
            print(f"⚠️ Fichier {perf_file} introuvable. Aucune performance à visualiser.")
            return
        
        # Charger les données
        with open(perf_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extraire et trier les configurations
        configurations = []
        
        for stage_key in sorted(data.keys(), key=lambda x: int(x.split('_')[1])):
            stage_nb = int(stage_key.split('_')[1])
            
            for n_layers in sorted(data[stage_key].keys(), key=int):
                for hidden_size in sorted(data[stage_key][n_layers].keys(), key=int):
                    for nb_epochs in sorted(data[stage_key][n_layers][hidden_size].keys(), key=int):
                        metrics = data[stage_key][n_layers][hidden_size][nb_epochs]
                        
                        configurations.append({
                            'stage': stage_nb,
                            'n_layers': int(n_layers),
                            'hidden_size': int(hidden_size),
                            'nb_epochs': int(nb_epochs),
                            'avg_loss': metrics['avg_loss'],
                            'std_dev': metrics['std_dev'],
                            'label': f"S{stage_nb}_L{n_layers}_H{hidden_size}_E{nb_epochs}"
                        })
        
        if not configurations:
            print("⚠️ Aucune configuration trouvée dans le fichier JSON.")
            return
        
        # Préparer les données pour le plot
        labels = [cfg['label'] for cfg in configurations]
        avg_losses = [cfg['avg_loss'] for cfg in configurations]
        std_devs = [cfg['std_dev'] for cfg in configurations]
        stages = [cfg['stage'] for cfg in configurations]
        
        # Trouver la meilleure configuration (plus petite avg_loss)
        best_idx = avg_losses.index(min(avg_losses))
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(max(16, len(configurations) * 0.8), 8))
        
        # Récupérer les couleurs des stages depuis le STAGE_MANAGER
        stage_colors = {}
        for stage in STAGE_MANAGER.get_stages():
            stage_colors[stage.get_stage_nb()] = stage.get_color()
        
        # Couleurs des barres selon le stage
        bar_colors = [stage_colors.get(stage, 'gray') for stage in stages]
        
        # Tracer les barres avec error bars
        x_positions = range(len(configurations))
        bars = ax.bar(x_positions, avg_losses, yerr=std_devs,
                     color=bar_colors, alpha=0.7, capsize=5,
                     edgecolor='black', linewidth=1.5)
        
        # Mettre en évidence la meilleure configuration
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        bars[best_idx].set_alpha(1.0)
        
        # Ajouter une étoile sur la meilleure configuration
        ax.text(best_idx, avg_losses[best_idx] + std_devs[best_idx], '⭐',
                ha='center', va='bottom', fontsize=20, color='gold')
        
        # Ajouter des zones de background colorées par stage
        current_stage = stages[0]
        stage_start = 0
        
        for i in range(1, len(stages) + 1):
            if i == len(stages) or stages[i] != current_stage:
                # Fin d'une zone de stage
                stage_end = i
                ax.axvspan(stage_start - 0.5, stage_end - 0.5,
                          alpha=0.15, color=stage_colors.get(current_stage, 'gray'),
                          zorder=0)
                
                # Ligne de séparation
                if i < len(stages):
                    ax.axvline(x=i - 0.5, color='black', linestyle='--',
                              linewidth=2, alpha=0.5)
                
                # Préparer pour le prochain stage
                if i < len(stages):
                    current_stage = stages[i]
                    stage_start = i
        
        # Ajouter une ligne de tendance (moyenne mobile simple sur 3 points)
        if len(avg_losses) >= 3:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(avg_losses, size=min(5, len(avg_losses)), mode='nearest')
            ax.plot(x_positions, smoothed, 'r--', linewidth=2,
                   alpha=0.6, label='Tendance (moyenne mobile)')
        
        # Configuration des axes
        ax.set_xlabel('Configuration (Stage_Layers_HiddenSize_Epochs)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Perte Moyenne (avg_loss)', fontsize=12, fontweight='bold')
        ax.set_title('Comparaison des Performances par Configuration\n(error bars = écart-type)',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Labels en X avec rotation
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
        # Grille horizontale
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Échelle logarithmique en Y si les valeurs varient beaucoup
        if max(avg_losses) / min(avg_losses) > 10:
            ax.set_yscale('log')
            ax.set_ylabel('Perte Moyenne (avg_loss) - échelle log', fontsize=12, fontweight='bold')
        
        # Légende des stages
        from matplotlib.patches import Patch
        legend_elements = []
        for stage in sorted(set(stages)):
            legend_elements.append(
                Patch(facecolor=stage_colors.get(stage, 'gray'),
                     alpha=0.7, edgecolor='black',
                     label=f'Stage {stage}')
            )
        
        if len(avg_losses) >= 3:
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], color='r', linestyle='--', linewidth=2,
                      alpha=0.6, label='Tendance')
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Annotations pour la meilleure config
        best_config = configurations[best_idx]
        textstr = f"🏆 Meilleure config:\n{best_config['label']}\nLoss: {best_config['avg_loss']:.6f} ± {best_config['std_dev']:.6f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gold', linewidth=2)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = Path(CONFIG.OUTPUT_DIR) / "evaluation_performances.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Graphique de performances sauvegardé dans {output_path}")
        print(f"🏆 Meilleure configuration: {best_config['label']} avec loss={best_config['avg_loss']:.6f}")
    
    
    @staticmethod
    def _plot_curriculum_progression():
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


_visualizer = None


def get_visualizer():
    global _visualizer
    if _visualizer is None:
        _visualizer = ProgressiveVisualizer()
    return _visualizer
