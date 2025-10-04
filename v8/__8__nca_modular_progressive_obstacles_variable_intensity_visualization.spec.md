# Spécification: Système de Visualisation pour NCA Modulaire avec Intensités Variables (Version 8__)

## Vue d'ensemble

Cette spécification définit le système de visualisation avancé pour la version 8__ du NCA modulaire avec intensités variables. Elle étend les fonctionnalités de visualisation de la v7__ en ajoutant le support des intensités variables et des métriques spécialisées pour l'étape 4.

## Architecture du Système de Visualisation

### Composants Principaux

1. **`ProgressiveVisualizer`** - Visualiseur principal étendu
2. **`IntensityAwareAnimator`** - Générateur d'animations avec affichage d'intensité
3. **`VariableIntensityMetricsPlotter`** - Graphiques spécialisés pour intensités variables
4. **`CurriculumSummaryGenerator`** - Générateur de résumés visuels complets

## Classes et Fonctionnalités

### 1. `ProgressiveVisualizer` (Version 8__ Étendue)

```python
class ProgressiveVisualizer:
    """
    Système de visualisation avancé pour l'apprentissage modulaire avec intensités variables.
    Étend les fonctionnalités v7__ avec support des intensités variables.
    """
    
    def __init__(self, interactive: bool = False):
        self.interactive = interactive
        self.frame_data = {}  # Données par étape
        self.intensity_data = {}  # NOUVEAU: Données d'intensité par étape
        
    # === MÉTHODES HÉRITÉES DE v7__ (ADAPTÉES) ===
    
    def visualize_stage_results(self, model: ImprovedNCA, stage: int, 
                              vis_seed: int = 123, source_intensity: Optional[float] = None) -> Dict[str, Any]:
        """
        Visualise les résultats d'une étape avec support intensité variable.
        
        Args:
            model: Modèle NCA entraîné
            stage: Numéro d'étape (1-4)
            vis_seed: Graine pour reproductibilité
            source_intensity: Intensité spécifique pour étape 4 (None = intensité standard)
            
        Returns:
            Dictionnaire avec données de visualisation étendues
        """
        # Génération adaptée selon l'étape
        if stage == 4 and source_intensity is not None:
            # Étape 4: utilise l'intensité spécifiée
            target_seq, source_mask, obstacle_mask = simulator.generate_stage_sequence(
                stage=4, source_intensity=source_intensity, seed=vis_seed)
        else:
            # Étapes 1-3: intensité standard
            target_seq, source_mask, obstacle_mask = simulator.generate_stage_sequence(
                stage=stage, seed=vis_seed)
            source_intensity = cfg.DEFAULT_SOURCE_INTENSITY
            
        # Simulation NCA avec intensité appropriée
        nca_sequence = self._simulate_nca_with_intensity(model, target_seq[0], 
                                                       source_mask, obstacle_mask, 
                                                       source_intensity)
        
        # Données de visualisation étendues
        vis_data = {
            'stage': stage,
            'target_sequence': [t.detach().cpu().numpy() for t in target_seq],
            'nca_sequence': [t.detach().cpu().numpy() for t in nca_sequence],
            'source_mask': source_mask.detach().cpu().numpy(),
            'obstacle_mask': obstacle_mask.detach().cpu().numpy(),
            'source_intensity': source_intensity,  # NOUVEAU
            'vis_seed': vis_seed
        }
        
        # Création des visualisations avec intensité
        self._create_stage_animations_with_intensity(vis_data)
        self._create_stage_convergence_plot_with_intensity(vis_data)
        
        return vis_data
    
    # === NOUVELLES MÉTHODES VERSION 8__ ===
    
    def visualize_intensity_curriculum(self, stage_4_metrics: Dict[str, Any]):
        """
        NOUVEAU: Visualise le curriculum d'intensité de l'étape 4.
        
        Args:
            stage_4_metrics: Métriques de l'étape 4 avec historique des intensités
        """
        self._plot_intensity_distribution(stage_4_metrics)
        self._plot_intensity_progression(stage_4_metrics)
        self._plot_performance_by_intensity(stage_4_metrics)
        
    def create_intensity_comparison_grid(self, model: ImprovedNCA, 
                                       intensity_samples: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]):
        """
        NOUVEAU: Crée une grille comparative pour différentes intensités.
        
        Args:
            model: Modèle entraîné
            intensity_samples: Liste des intensités à comparer
        """
        fig, axes = plt.subplots(2, len(intensity_samples), figsize=(20, 8))
        
        for i, intensity in enumerate(intensity_samples):
            # Génération avec intensité spécifique
            vis_data = self.visualize_stage_results(model, stage=4, 
                                                  source_intensity=intensity)
            
            # État initial
            axes[0, i].imshow(vis_data['nca_sequence'][0], cmap='hot', vmin=0, vmax=1)
            axes[0, i].contour(vis_data['obstacle_mask'], levels=[0.5], colors='cyan', linewidths=2)
            axes[0, i].set_title(f'Initial (I={intensity:.3f})')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # État final
            axes[1, i].imshow(vis_data['nca_sequence'][-1], cmap='hot', vmin=0, vmax=1)
            axes[1, i].contour(vis_data['obstacle_mask'], levels=[0.5], colors='cyan', linewidths=2)
            axes[1, i].set_title(f'Final (I={intensity:.3f})')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        plt.tight_layout()
        plt.savefig(Path(cfg.OUTPUT_DIR) / "intensity_comparison_grid.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
```

### 2. `IntensityAwareAnimator` (Nouveau)

```python
class IntensityAwareAnimator:
    """
    Générateur d'animations avec affichage d'intensité dans les titres.
    Spécialisé pour la version 8__ avec intensités variables.
    """
    
    def create_intensity_labeled_gif(self, sequence: List[np.ndarray], 
                                   obstacle_mask: np.ndarray,
                                   source_intensity: float,
                                   filepath: Path, 
                                   base_title: str):
        """
        Crée un GIF avec l'intensité affichée dans le titre.
        
        Args:
            sequence: Séquence temporelle à animer
            obstacle_mask: Masque des obstacles
            source_intensity: Intensité de la source
            filepath: Chemin de sauvegarde
            base_title: Titre de base (avant intensité)
        """
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def animate(frame):
            ax.clear()
            im = ax.imshow(sequence[frame], cmap='hot', vmin=0, vmax=1)
            ax.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            
            # NOUVEAU: Titre avec intensité
            title = f'{base_title} (I={source_intensity:.3f}) - t={frame}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Colorbar pour référence
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            return [im]
        
        ani = animation.FuncAnimation(fig, animate, frames=len(sequence), 
                                    interval=200, blit=False)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()
    
    def create_comparison_with_intensity(self, target_seq: List[np.ndarray], 
                                       nca_seq: List[np.ndarray],
                                       obstacle_mask: np.ndarray, 
                                       source_intensity: float,
                                       filepath: Path):
        """
        Crée une animation de comparaison avec affichage d'intensité.
        """
        import matplotlib.animation as animation
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Cible
            im1 = ax1.imshow(target_seq[frame], cmap='hot', vmin=0, vmax=1)
            ax1.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax1.set_title(f'Cible (I={source_intensity:.3f}) - t={frame}')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # NCA
            im2 = ax2.imshow(nca_seq[frame], cmap='hot', vmin=0, vmax=1)
            ax2.contour(obstacle_mask, levels=[0.5], colors='cyan', linewidths=2)
            ax2.set_title(f'NCA (I={source_intensity:.3f}) - t={frame}')
            ax2.set_xticks([])
            ax2.set_yticks([])
            
            return [im1, im2]
        
        n_frames = min(len(target_seq), len(nca_seq))
        ani = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                    interval=200, blit=False)
        ani.save(filepath, writer='pillow', fps=5)
        plt.close()
```

### 3. `VariableIntensityMetricsPlotter` (Nouveau)

```python
class VariableIntensityMetricsPlotter:
    """
    Générateur de graphiques spécialisés pour les métriques d'intensité variable.
    """
    
    def plot_intensity_distribution(self, intensity_history: List[float], 
                                  output_dir: Path):
        """
        Graphique de distribution des intensités utilisées pendant l'entraînement.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogramme
        ax1.hist(intensity_history, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Intensité de Source')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution des Intensités (Étape 4)')
        ax1.grid(True, alpha=0.3)
        
        # Évolution temporelle
        ax2.plot(intensity_history, 'o-', alpha=0.6, markersize=2)
        ax2.set_xlabel('Simulation #')
        ax2.set_ylabel('Intensité')
        ax2.set_title('Évolution des Intensités au Cours de l\'Entraînement')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "intensity_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_performance_by_intensity_range(self, metrics_by_intensity: Dict[str, List[float]],
                                          output_dir: Path):
        """
        Performance du modèle selon les plages d'intensité.
        """
        # Regroupement par plages d'intensité
        ranges = {
            'Très faible (0.0-0.2)': [],
            'Faible (0.2-0.4)': [],
            'Moyenne (0.4-0.6)': [],
            'Forte (0.6-0.8)': [],
            'Très forte (0.8-1.0)': []
        }
        
        for intensity, loss in zip(metrics_by_intensity['intensities'], 
                                 metrics_by_intensity['losses']):
            if intensity <= 0.2:
                ranges['Très faible (0.0-0.2)'].append(loss)
            elif intensity <= 0.4:
                ranges['Faible (0.2-0.4)'].append(loss)
            elif intensity <= 0.6:
                ranges['Moyenne (0.4-0.6)'].append(loss)
            elif intensity <= 0.8:
                ranges['Forte (0.6-0.8)'].append(loss)
            else:
                ranges['Très forte (0.8-1.0)'].append(loss)
        
        # Graphique en boîtes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data_to_plot = [losses for losses in ranges.values() if losses]
        labels = [label for label, losses in ranges.items() if losses]
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Coloration des boîtes
        colors = ['lightcoral', 'lightsalmon', 'lightblue', 'lightgreen', 'lightsteelblue']
        for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Perte MSE')
        ax.set_title('Performance par Plage d\'Intensité (Étape 4)')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "performance_by_intensity_range.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_convergence_analysis_by_intensity(self, convergence_data: Dict[str, Any],
                                             output_dir: Path):
        """
        Analyse de convergence selon l'intensité.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Temps de convergence vs intensité
        intensities = convergence_data['intensities']
        convergence_times = convergence_data['convergence_times']
        
        scatter = ax1.scatter(intensities, convergence_times, alpha=0.6, c=intensities, 
                            cmap='viridis', s=50)
        ax1.set_xlabel('Intensité de Source')
        ax1.set_ylabel('Temps de Convergence (époques)')
        ax1.set_title('Temps de Convergence vs Intensité')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1)
        
        # Stabilité vs intensité
        stability_scores = convergence_data['stability_scores']
        
        scatter2 = ax2.scatter(intensities, stability_scores, alpha=0.6, c=intensities,
                             cmap='plasma', s=50)
        ax2.set_xlabel('Intensité de Source')
        ax2.set_ylabel('Score de Stabilité')
        ax2.set_title('Stabilité vs Intensité')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(output_dir / "convergence_analysis_by_intensity.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
```

### 4. Système de Visualisation Intégré

```python
def create_complete_visualization_suite(model: ImprovedNCA, 
                                      global_metrics: Dict[str, Any],
                                      output_dir: Path):
    """
    Crée la suite complète de visualisations pour la version 8__.
    
    Comprend:
    - Visualisations par étape (1-4) avec intensités
    - Comparaisons d'intensités pour l'étape 4
    - Graphiques de curriculum étendu
    - Métriques spécialisées intensités variables
    """
    
    # Initialisation des visualiseurs
    visualizer = ProgressiveVisualizer()
    animator = IntensityAwareAnimator()
    metrics_plotter = VariableIntensityMetricsPlotter()
    
    # 1. Visualisations par étape (étendues)
    print("🎨 Génération des visualisations par étape...")
    for stage in [1, 2, 3]:
        # Étapes 1-3: intensité standard
        stage_vis = visualizer.visualize_stage_results(model, stage)
        
    # Étape 4: plusieurs intensités
    stage_4_intensities = [0.0, 0.3, 0.7, 1.0]
    for intensity in stage_4_intensities:
        stage_vis = visualizer.visualize_stage_results(model, stage=4, 
                                                     source_intensity=intensity)
    
    # 2. Grille comparative d'intensités
    print("🎨 Génération de la grille comparative d'intensités...")
    visualizer.create_intensity_comparison_grid(model)
    
    # 3. Curriculum d'intensité (nouveau)
    print("🎨 Génération des graphiques de curriculum d'intensité...")
    if 'stage_4_metrics' in global_metrics:
        visualizer.visualize_intensity_curriculum(global_metrics['stage_4_metrics'])
    
    # 4. Métriques spécialisées intensités
    print("🎨 Génération des métriques d'intensité...")
    if 'intensity_metrics' in global_metrics:
        intensity_metrics = global_metrics['intensity_metrics']
        metrics_plotter.plot_intensity_distribution(intensity_metrics['history'], output_dir)
        metrics_plotter.plot_performance_by_intensity_range(intensity_metrics, output_dir)
        metrics_plotter.plot_convergence_analysis_by_intensity(intensity_metrics, output_dir)
    
    # 5. Résumé visuel étendu (hérité v7__ + ajouts v8__)
    print("🎨 Génération du résumé visuel complet...")
    visualizer.create_curriculum_summary_extended(global_metrics)
    
    print("✅ Suite complète de visualisations générée!")
```

## Fonctionnalités Héritées de v7__ (Adaptées)

### Visualisations Conservées et Étendues

1. **Animations par étape** ✅
   - Héritées: animations comparatives cible vs NCA
   - Extension: affichage intensité dans les titres

2. **Graphiques de convergence** ✅
   - Hérités: analyse de convergence par étape
   - Extension: seuils adaptatifs pour étape 4

3. **Progression du curriculum** ✅
   - Hérité: graphique des pertes par étape
   - Extension: 4ème étape avec intensités variables

4. **Comparaison inter-étapes** ✅
   - Hérité: barres comparatives de performance
   - Extension: métriques spéciales étape 4

5. **Résumé de performance** ✅
   - Hérité: rapport textuel complet
   - Extension: statistiques d'intensité

## Nouvelles Fonctionnalités Version 8__

### Spécialisées Intensités Variables

1. **Distribution des intensités** 🆕
   - Histogramme des intensités utilisées
   - Évolution temporelle des intensités

2. **Performance par plage d'intensité** 🆕
   - Boxplots de performance selon intensité
   - Analyse de robustesse

3. **Grille comparative d'intensités** 🆕
   - Comparaison visuelle côte à côte
   - États initial et final pour chaque intensité

4. **Convergence selon intensité** 🆕
   - Temps de convergence vs intensité
   - Scores de stabilité vs intensité

## Structure des Fichiers de Sortie

```
__8__nca_outputs_modular_progressive_obstacles_variable_intensity_seed_XXX/
├── stage_1/
│   ├── animation_comparaison_étape_1.gif
│   ├── animation_nca_étape_1.gif
│   └── convergence_étape_1.png
├── stage_2/
│   ├── animation_comparaison_étape_2.gif
│   ├── animation_nca_étape_2.gif
│   └── convergence_étape_2.png
├── stage_3/
│   ├── animation_comparaison_étape_3.gif
│   ├── animation_nca_étape_3.gif
│   └── convergence_étape_3.png
├── stage_4/
│   ├── animation_comparaison_étape_4_I_0.000.gif
│   ├── animation_comparaison_étape_4_I_0.300.gif
│   ├── animation_comparaison_étape_4_I_0.700.gif
│   ├── animation_comparaison_étape_4_I_1.000.gif
│   └── convergence_étape_4_multi_intensities.png
├── intensity_comparison_grid.png                    # 🆕
├── intensity_distribution.png                       # 🆕
├── performance_by_intensity_range.png              # 🆕
├── convergence_analysis_by_intensity.png           # 🆕
├── curriculum_progression_extended.png             # Étendu
├── stage_comparison_with_stage_4.png              # Étendu
├── performance_summary_extended.png               # Étendu
└── complete_metrics_with_intensities.json         # Étendu
```

## Intégration avec le Système Principal

Le système de visualisation s'intègre avec le `ModularTrainer` de la version 8__ :

```python
# Dans ModularTrainer.train_full_curriculum()
def train_full_curriculum(self) -> Dict[str, Any]:
    # ...entraînement...
    
    # Collecte des métriques d'intensité pour étape 4
    if hasattr(self, 'intensity_manager'):
        global_metrics['intensity_metrics'] = {
            'history': self.intensity_manager.intensity_history,
            'statistics': self.intensity_manager.get_intensity_statistics(),
            # ... autres métriques spécialisées
        }
    
    # Visualisations complètes
    create_complete_visualization_suite(self.model, global_metrics, Path(cfg.OUTPUT_DIR))
    
    return global_metrics
```

## Compatibilité et Migration

- **100% compatible** avec les visualisations v7__ existantes
- **Extension transparente** pour la 4ème étape
- **Rétrocompatibilité** assurée pour les étapes 1-3
- **Nouveaux paramètres optionnels** pour intensités variables

Cette spécification assure une continuité parfaite avec v7__ tout en ajoutant les fonctionnalités spécialisées pour les intensités variables de v8__.
