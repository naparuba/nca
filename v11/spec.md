# Spécification: NCA Modulaire v11 - Architecture Découplée

## Vue d'ensemble

**Version 11** représente une refactorisation majeure vers une architecture découplée et modulaire pour l'apprentissage progressif des Neural Cellular Automata (NCA). Le système implémente un curriculum d'apprentissage en 3 étapes avec une séparation claire des responsabilités.

## Architecture Modulaire - État Actuel

### Philosophie de Conception

L'architecture v11 suit les principes suivants:
- **Séparation des responsabilités**: Chaque module a une responsabilité unique et bien définie
- **Découplage**: Les modules communiquent via des interfaces claires
- **Extensibilité**: Ajout facile de nouvelles étapes ou fonctionnalités
- **Singleton pattern**: Utilisation de singletons pour les gestionnaires globaux (STAGE_MANAGER, simulator)

### Structure Actuelle des Modules

```
v11/
├── config.py           → ModularConfig (paramètres centralisés)
├── torching.py         → DEVICE (détection CUDA/CPU)
├── main.py             → Point d'entrée principal
├── nca_model.py        → ImprovedNCA (réseau neuronal)
├── updater.py          → OptimizedNCAUpdater (application du modèle)
├── simulator.py        → HeatDiffusionSimulator (simulation physique)
├── sequences.py        → OptimizedSequenceCache (cache par étape)
├── stage_manager.py    → StageManager (orchestration étapes)
├── trainer.py          → ModularTrainer (logique d'entraînement)
├── visualizer.py       → ProgressiveVisualizer (génération graphiques)
└── stages/
    ├── base_stage.py           → BaseStage (classe abstraite)
    ├── stage_1_no_obstacle.py  → Stage1NoObstacle
    ├── stage_2_one_obstacle.py → Stage2OneObstacle
    └── stage_3_few_obstacles.py→ Stage3FewObstacles
```

## Graphe d'Appels - Flux Principal

### 1. Initialisation (main.py)
```
main() 
  ├── torch.manual_seed(CONFIG.SEED)
  ├── ImprovedNCA() → nca_model.py
  ├── ModularTrainer(model) → trainer.py
  ├── trainer.train_full_curriculum()
  ├── ProgressiveVisualizer() → visualizer.py
  └── visualizer.visualize_stage_results()
```

### 2. Entraînement Modulaire (trainer.py)
```
ModularTrainer.train_full_curriculum()
  └── for stage in STAGE_MANAGER.get_stages():
      └── _train_stage(stage, CONFIG.NB_EPOCHS_BY_STAGE)
          ├── sequence_cache.initialize_stage_cache(stage)
          │   └── simulator.generate_stage_sequence(stage, ...)
          │       └── stage.generate_environment(size, source_pos)
          ├── for epoch in range(max_epochs):
          │   ├── _adjust_learning_rate(stage_nb, epoch_in_stage)
          │   ├── sequence_cache.get_stage_batch(stage, 1)
          │   └── _train_step(target_seq, source_mask, obstacle_mask)
          │       └── updater.step(grid_pred, source_mask, obstacle_mask)
          │           └── model(valid_patches)
          └── stage.save_stage_checkpoint(model_state, optimizer_state)
```

### 3. Génération des Stages (stages/)
```
STAGE_MANAGER.get_stages()
  ├── Stage1NoObstacle.generate_environment() → torch.zeros()
  ├── Stage2OneObstacle.generate_environment() → Un obstacle
  └── Stage3FewObstacles.generate_environment() → Obstacles multiples
      └── _validate_connectivity() → Flood-fill algorithm
```

### 4. Simulation Physique (simulator.py)
```
HeatDiffusionSimulator.generate_stage_sequence()
  ├── stage.generate_environment(size, source_pos)
  ├── Initialisation: grid[i0, j0] = SOURCE_INTENSITY
  └── for _ in range(n_steps):
      └── step(grid, source_mask, obstacle_mask)
          └── F.conv2d(x, kernel, padding=1) # Diffusion 3x3
```

### 5. Cache et Optimisations (sequences.py)
```
OptimizedSequenceCache
  ├── initialize_stage_cache(stage)
  │   └── for i in range(STAGE_CACHE_SIZE):
  │       └── simulator.generate_stage_sequence()
  ├── get_stage_batch(stage, batch_size)
  └── clear_stage_cache(stage_nb) # Libération mémoire
```

### 6. Visualisation (visualizer.py)
```
ProgressiveVisualizer
  ├── visualize_stage_results(model, stage)
  │   ├── simulator.generate_stage_sequence()
  │   ├── OptimizedNCAUpdater.step() # Prédiction NCA
  │   ├── _create_stage_animations()
  │   └── _create_stage_convergence_plot()
  └── create_curriculum_summary()
      ├── _plot_curriculum_progression()
      └── _plot_stage_comparison()
```

## Détail des Modules

### 1. Configuration (config.py)
**Responsabilité**: Paramètres centralisés du système
```python
class ModularConfig:
    SEED = 3333
    NB_EPOCHS_BY_STAGE = 30
    TOTAL_EPOCHS = 90  # 3 * 30
    GRID_SIZE = 16
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 4
    STAGE_CACHE_SIZE = 250
```

### 2. Modèle NCA (nca_model.py)
**Responsabilité**: Architecture du réseau neuronal
```python
class ImprovedNCA(nn.Module):
    - input_size: 11 (patch 3x3 + source + obstacle)
    - Architecture: Linear + BatchNorm + ReLU + Dropout
    - Sortie: delta * 0.1 (scaling pour stabilité)
```

### 3. Updater (updater.py)
**Responsabilité**: Application optimisée du modèle NCA
```python
class OptimizedNCAUpdater:
    - Extraction vectorisée des patches 3x3
    - Application seulement sur positions valides
    - Contraintes: obstacles = 0, sources = constantes
```

### 4. Simulateur (simulator.py)
**Responsabilité**: Simulation physique de référence
```python
class HeatDiffusionSimulator:
    - Noyau de convolution 3x3 pour diffusion
    - Génération de séquences par stage
    - Gestion automatique des contraintes
```

### 5. Gestionnaire de Stages (stage_manager.py)
**Responsabilité**: Orchestration des étapes d'apprentissage
```python
class StageManager:
    - Liste des stages: [Stage1, Stage2, Stage3]
    - Attribution automatique des numéros de stage
    - Interface unifiée: get_stages(), get_stage(nb)
```

### 6. Stages Individuels (stages/)
**Responsabilité**: Définition des environnements par étape

#### BaseStage (base_stage.py)
```python
class BaseStage(ABC):
    - Métriques: epochs_trained, loss_history, stage_lrs
    - Sauvegarde: checkpoints + métriques JSON
    - Interface: generate_environment() [abstraite]
```

#### Stage1NoObstacle
```python
- Environnement: torch.zeros() (pas d'obstacles)
- Objectif: Apprentissage de la diffusion de base
```

#### Stage2OneObstacle
```python
- Environnement: Un obstacle de taille aléatoire
- Contraintes: Évitement de la source, placement valide
```

#### Stage3FewObstacles
```python
- Environnement: 2-4 obstacles multiples
- Validation: Algorithme de connectivité (flood-fill)
- Contrainte: 50% minimum de connectivité
```

### 7. Cache de Séquences (sequences.py)
**Responsabilité**: Optimisation mémoire et performance
```python
class OptimizedSequenceCache:
    - Cache séparé par stage (250 séquences/stage)
    - Libération automatique des stages précédents
    - Mélange périodique pour diversité
```

### 8. Entraîneur (trainer.py)
**Responsabilité**: Logique d'entraînement modulaire
```python
class ModularTrainer:
    - Learning rate adaptatif: 1.0 → 0.6 linéaire par stage
    - Décroissance cosine intra-stage
    - Gradient clipping (max_norm=1.0)
    - Sauvegarde automatique des checkpoints
```

### 9. Visualiseur (visualizer.py)
**Responsabilité**: Génération des graphiques et animations
```python
class ProgressiveVisualizer:
    - Animations GIF de comparaison par stage
    - Graphiques de convergence
    - Résumé global du curriculum
    - Métriques de performance
```

## Flux de Données

### 1. Séquence d'Entraînement
```
Stage → Environment → Simulator → Target Sequence
  ↓
Cache → Batch → Trainer → Model → Prediction
  ↓
Loss → Optimizer → Model Update
```

### 2. Gestion Mémoire
```
Stage N: Initialize Cache (250 sequences)
  ↓
Stage N: Training Loop
  ↓
Stage N+1: Clear Cache N, Initialize Cache N+1
```

### 3. Learning Rate Dynamique
```
base_lr = 1e-3
stage_multiplier = 1.0 - ((stage_nb-1) / (n_stages-1)) * 0.4
cosine_factor = 0.5 * (1 + cos(π * epoch / epochs_per_stage))
final_lr = base_lr * stage_multiplier * (0.1 + 0.9 * cosine_factor)
```

## Points d'Extension

### 1. Ajout de Nouveaux Stages
```python
# 1. Créer stage_4_complex_obstacles.py
class Stage4ComplexObstacles(BaseStage):
    def generate_environment(self, size, source_pos):
        # Implémentation spécifique
        
# 2. Ajouter dans stage_manager.py
self._stages = [Stage1(), Stage2(), Stage3(), Stage4()]
```

### 2. Nouveaux Types de Visualisation
```python
# Dans visualizer.py
def create_3d_visualization(self, vis_data):
    # Visualisation 3D des gradients
    
def create_flow_field_analysis(self, vis_data):
    # Analyse des champs de flux
```

### 3. Optimisations Supplémentaires
```python
# Cache intelligent avec LRU
# Parallélisation des stages
# Mixed precision training
# Distributed training
```

## Métriques et Monitoring

### 1. Métriques par Stage
- **Loss history**: Évolution de la perte MSE
- **Learning rates**: Historique des LR adaptatifs
- **Epochs trained**: Nombre d'époques utilisées
- **Convergence**: Temps de convergence par stage

### 2. Métriques Globales
- **Temps total**: Durée d'entraînement complète
- **Mémoire**: Utilisation GPU/CPU par stage
- **Stabilité**: Variance des performances

### 3. Visualisations Générées
- **Animations**: GIFs de comparaison cible/prédiction
- **Convergence**: Graphiques d'erreur temporelle
- **Curriculum**: Progression globale multi-stage
- **Performance**: Comparaisons inter-stages

## État Actuel vs Prochaines Refactorisations

### ✅ Réalisé en v11
- Découplage complet des modules
- Architecture orientée objet claire
- Système de stages extensible
- Cache optimisé par étape
- Visualisations avancées
- Learning rate adaptatif dynamique

### 🎯 Prochaines Améliorations Possibles
- **Scheduler avancé**: Transitions adaptatives entre stages
- **Métriques temps réel**: Dashboard de monitoring
- **Parallélisation**: Entraînement multi-GPU
- **Validation**: Tests unitaires pour chaque module
- **Documentation**: API docs complète
- **Configuration**: YAML/JSON externe pour les paramètres

## Contraintes Techniques

### 1. Mémoire
- Cache limité à 250 séquences/stage
- Libération automatique des stages précédents
- Utilisation de `.detach().cpu()` pour les visualisations

### 2. Performance
- Extraction vectorisée des patches (unfold)
- Application seulement sur positions valides
- Gradient clipping pour stabilité numérique

### 3. Reproductibilité
- Seeds fixes: CONFIG.SEED (entraînement), VISUALIZATION_SEED (visualisation)
- Deterministic operations sur GPU si disponible

Cette architecture v11 représente un système robuste et extensible pour l'apprentissage modulaire progressif des NCA, avec une séparation claire des responsabilités et des interfaces bien définies pour les futures extensions.
