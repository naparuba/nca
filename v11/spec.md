# Spécification: NCA Modulaire avec Obstacles Progressifs (Version 11)

## Vue d'ensemble

Cette version implémente un système d'apprentissage modulaire en plusieurs étapes pour les Neural Cellular Automata (NCA), avec une progression des obstacles de complexité croissante. Le système utilise un seul modèle qui apprend progressivement à gérer des scénarios de plus en plus complexes.

**Version 11** introduit un découplage important des responsabilités avec une architecture orientée objet claire et une séparation des préoccupations.

## Architecture Modulaire - Découplage des Composants

### Philosophie de Conception

L'architecture v11 suit les principes suivants:
- **Séparation des responsabilités**: Chaque module a une responsabilité unique et bien définie
- **Découplage**: Les modules communiquent via des interfaces claires
- **Extensibilité**: Ajout facile de nouvelles étapes ou fonctionnalités
- **Singleton pattern**: Utilisation de singletons pour les gestionnaires globaux (STAGE_MANAGER, simulator)
- **Inversion de dépendances**: Les modules de haut niveau ne dépendent pas des détails d'implémentation

### Hiérarchie des Modules

```
v11/
├── Configuration & Constantes
│   ├── config.py           → ModularConfig (paramètres centralisés)
│   └── torching.py         → DEVICE (détection CUDA/CPU)
│
├── Modèle & Mise à Jour
│   ├── nca_model.py        → ImprovedNCA (réseau neuronal)
│   └── updater.py          → OptimizedNCAUpdater (application du modèle)
│
├── Simulation & Environnement
│   ├── simulator.py        → DiffusionSimulator (simulation physique)
│   ├── obstacles.py        → ProgressiveObstacleManager (gestion obstacles)
│   └── stages/             → Définitions des étapes
│       ├── base_stage.py           → BaseStage (classe abstraite)
│       ├── stage_1_no_obstacle.py  → Stage1NoObstacle
│       ├── stage_2_one_obstacle.py → Stage2OneObstacle
│       └── stage_3_few_obstacles.py→ Stage3FewObstacles
│
├── Gestion du Curriculum
│   ├── stage_manager.py    → StageManager (orchestration étapes)
│   ├── scheduler.py        → CurriculumScheduler (progression adaptative)
│   └── sequences.py        → OptimizedSequenceCache (cache par étape)
│
├── Entraînement
│   ├── trainer.py          → ModularTrainer (logique d'entraînement)
│   └── train.py            → main() (point d'entrée)
│
└── Visualisation
    └── visualizer.py       → ProgressiveVisualizer (génération graphiques)
```

## Graphe d'Appels Détaillé

### 1. Flux d'Exécution Principal

```
train.py::main()
    │
    ├──> ImprovedNCA() [nca_model.py]
    │       └──> CONFIG [config.py]
    │
    ├──> ModularTrainer(model) [trainer.py]
    │       ├──> OptimizedNCAUpdater(model) [updater.py]
    │       ├──> CurriculumScheduler() [scheduler.py]
    │       └──> OptimizedSequenceCache() [sequences.py]
    │               └──> get_simulator() [simulator.py]
    │
    ├──> trainer.train_full_curriculum()
    │       │
    │       ├──> trainer.train_stage(1, epochs)
    │       ├──> trainer.train_stage(2, epochs)
    │       └──> trainer.train_stage(3, epochs)
    │               │
    │               ├──> sequence_cache.initialize_stage_cache(stage_nb)
    │               │       └──> simulator.generate_stage_sequence(stage_nb, ...)
    │               │               └──> obstacle_manager.generate_stage_environment(stage_nb, ...)
    │               │                       └──> STAGE_MANAGER.get_stage(stage_nb)
    │               │                               └──> BaseStage.generate_environment(...)
    │               │
    │               ├──> FOR each epoch:
    │               │   ├──> curriculum.adjust_learning_rate(optimizer, stage_nb, epoch)
    │               │   ├──> sequence_cache.get_stage_batch(stage_nb, batch_size)
    │               │   └──> trainer.train_step(...)
    │               │           └──> updater.step(grid, source_mask, obstacle_mask)
    │               │                   └──> model(patches)
    │               │
    │               ├──> curriculum.should_advance_stage(stage_nb, losses)
    │               └──> STAGE_MANAGER.get_stage(stage_nb).save_stage_checkpoint(...)
    │
    ├──> ProgressiveVisualizer() [visualizer.py]
    │       ├──> visualizer.visualize_stage_results(model, stage_nb)
    │       │       ├──> get_simulator().generate_stage_sequence(...)
    │       │       ├──> OptimizedNCAUpdater(model).step(...)
    │       │       ├──> _create_stage_animations(...)
    │       │       └──> _create_stage_convergence_plot(...)
    │       │
    │       └──> visualizer.create_curriculum_summary(metrics)
    │               ├──> _plot_curriculum_progression(...)
    │               ├──> _plot_stage_comparison(...)
    │               └──> _plot_performance_metrics(...)
    │
    └──> trainer.save_final_model(metrics)
```

### 2. Dépendances entre Modules

```
┌─────────────────────────────────────────────────────────────┐
│                         train.py                             │
│                     (Point d'entrée)                         │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──> config.py (CONFIG global)
             ├──> torching.py (DEVICE global)
             │
             v
┌────────────────────────────────────────────────────────────┐
│                      nca_model.py                           │
│                    ImprovedNCA(nn.Module)                   │
│  Dépendances: config.py, torch                              │
└────────────┬───────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────┐
│                       updater.py                            │
│                  OptimizedNCAUpdater                        │
│  Dépendances: nca_model.py, torch                           │
└────────────┬───────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────┐
│                      trainer.py                             │
│                   ModularTrainer                            │
│  Dépendances: nca_model, updater, scheduler,                │
│               sequences, stage_manager, config, torching    │
└────────────┬───────────────────────────────────────────────┘
             │
             ├──────────────────┬─────────────────┬───────────┐
             v                  v                 v           v
    ┌─────────────┐    ┌──────────────┐  ┌──────────┐  ┌─────────────┐
    │scheduler.py │    │sequences.py  │  │simulator.│  │visualizer.py│
    │Curriculum   │    │Optimized     │  │Diffusion │  │Progressive  │
    │Scheduler    │    │SequenceCache │  │Simulator │  │Visualizer   │
    └─────────────┘    └──────┬───────┘  └────┬─────┘  └──────┬──────┘
                              │               │               │
                              v               v               │
                       ┌──────────────────────────┐           │
                       │    simulator.py          │           │
                       │  get_simulator()         │<──────────┘
                       │  (singleton)             │
                       └──────────┬───────────────┘
                                  │
                                  v
                       ┌──────────────────────────┐
                       │    obstacles.py          │
                       │ ProgressiveObstacle      │
                       │ Manager                  │
                       └──────────┬───────────────┘
                                  │
                                  v
                       ┌──────────────────────────┐
                       │  stage_manager.py        │
                       │  STAGE_MANAGER           │
                       │  (singleton)             │
                       └──────────┬───────────────┘
                                  │
                                  v
                       ┌──────────────────────────┐
                       │    stages/               │
                       │  ├─ base_stage.py        │
                       │  ├─ stage_1_*.py         │
                       │  ├─ stage_2_*.py         │
                       │  └─ stage_3_*.py         │
                       └──────────────────────────┘
```

### 3. Flux de Données - Génération d'Environnement

```
sequence_cache.initialize_stage_cache(stage_nb)
    │
    └──> simulator.generate_stage_sequence(stage_nb, n_steps, size)
            │
            ├──> Génération position source aléatoire (i0, j0)
            │
            └──> obstacle_manager.generate_stage_environment(stage_nb, size, (i0, j0))
                    │
                    └──> STAGE_MANAGER.get_stage(stage_nb)
                            │
                            └──> BaseStage.generate_environment(size, source_pos)
                                    │
                                    ├─ [Stage 1] → torch.zeros(...) # Pas d'obstacles
                                    │
                                    ├─ [Stage 2] → Génère 1 obstacle
                                    │                └─ Vérification non-chevauchement source
                                    │
                                    └─ [Stage 3] → Génère 2-4 obstacles
                                                     ├─ Vérification non-chevauchement source
                                                     ├─ Vérification non-chevauchement entre obstacles
                                                     └─ _validate_connectivity() # Flood-fill
                                                         └─ Si échec → fallback Stage 2
```

### 4. Flux de Données - Entraînement

```
trainer.train_step(target_seq, source_mask, obstacle_mask, stage_nb)
    │
    ├──> Initialisation grille: grid_pred[source_mask] = INTENSITY
    │
    └──> FOR t_step in range(NCA_STEPS):
            │
            ├──> target = target_sequence[t_step + 1]
            │
            ├──> grid_pred = updater.step(grid_pred, source_mask, obstacle_mask)
            │       │
            │       ├──> Padding de la grille
            │       ├──> F.unfold() → extraction patches 3x3 vectorisée
            │       ├──> Concaténation [patches(9), source(1), obstacle(1)] → [H*W, 11]
            │       ├──> Filtrage positions valides (non-obstacles)
            │       ├──> model(valid_patches) → deltas
            │       ├──> Application deltas: new_grid += deltas
            │       ├──> Clamp [0, 1]
            │       └──> Contraintes: obstacles=0, sources=constantes
            │
            ├──> loss = MSE(grid_pred, target)
            │
            └──> total_loss += loss
            
         Après boucle temporelle:
         ├──> avg_loss = total_loss / NCA_STEPS
         ├──> avg_loss.backward()
         ├──> gradient_clipping(max_norm=1.0)
         └──> optimizer.step()
```

### 5. Flux de Décision - Curriculum Learning

```
trainer.train_stage(stage_nb, max_epochs)
    │
    └──> FOR epoch in range(max_epochs):
            │
            ├──> curriculum.adjust_learning_rate(optimizer, stage_nb, epoch)
            │       │
            │       ├──> base_lr * stage_multiplier[stage_nb]
            │       │      └─ Stage 1: 1.0, Stage 2: 0.8, Stage 3: 0.6
            │       │
            │       └──> Cosine decay: 0.5 * (1 + cos(π * epoch / max_epochs))
            │              └─ Final LR = stage_lr * (0.1 + 0.9 * cos_factor)
            │
            ├──> Entraînement batch...
            │
            └──> curriculum.should_advance_stage(stage_nb, recent_losses)
                    │
                    ├──> Calcul amélioration: losses[-2] - losses[-1]
                    │
                    ├──> Si amélioration < STAGNATION_THRESHOLD:
                    │       └─ no_improvement_count++
                    │    Sinon:
                    │       └─ no_improvement_count = 0
                    │
                    └──> Return: no_improvement_count >= STAGNATION_PATIENCE
                           └─ Si True → Early stopping de l'étape
```

## Classes et Composants Principaux

### Module: Configuration (`config.py`)

#### `ModularConfig`
**Responsabilité**: Configuration centralisée de tous les paramètres du système

**Attributs principaux**:
```python
# Seeds
SEED: int = 3333
VISUALIZATION_SEED: int = 3333

# Entraînement
NB_EPOCHS_BY_STAGE: int = 100
TOTAL_EPOCHS: int = 300  # 3 stages × 100
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 4

# Curriculum
STAGNATION_THRESHOLD: float = 0.000001
STAGNATION_PATIENCE: int = 20  # NB_EPOCHS_BY_STAGE // 5

# Architecture
GRID_SIZE: int = 16
HIDDEN_SIZE: int = 128
N_LAYERS: int = 3
NCA_STEPS: int = 20

# Obstacles
MIN_OBSTACLE_SIZE: int = 2
MAX_OBSTACLE_SIZE: int = 4

# Visualisation
PREVIS_STEPS: int = 30
POSTVIS_STEPS: int = 50
OUTPUT_DIR: str = "outputs"
```

**Utilisation**: Instance globale `CONFIG` importée partout

---

### Module: Modèle Neural (`nca_model.py`)

#### `ImprovedNCA(nn.Module)`
**Responsabilité**: Réseau neuronal calculant les deltas de mise à jour

**Architecture**:
- Input: 11 features (9 patch 3×3 + 1 source + 1 obstacle)
- N couches cachées avec BatchNorm + ReLU + Dropout(0.1)
- Output: 1 delta scalé par `delta_scale=0.1`

**Méthodes**:
- `__init__(input_size: int)`: Construction architecture
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass

**Dépendances**: `config.py`

---

### Module: Mise à Jour (`updater.py`)

#### `OptimizedNCAUpdater`
**Responsabilité**: Application optimisée du modèle NCA sur une grille

**Optimisations**:
- Extraction vectorisée des patches via `F.unfold()` (pas de boucles)
- Application uniquement sur cellules valides (non-obstacles)
- Contraintes strictes (obstacles=0, sources=constantes)

**Méthodes**:
- `__init__(model: ImprovedNCA)`
- `step(grid, source_mask, obstacle_mask) -> torch.Tensor`: Un pas de mise à jour

**Flux**:
1. Padding réplicatif de la grille
2. Extraction patches 3×3 pour toutes positions
3. Concaténation features additionnelles (source, obstacle)
4. Filtrage positions valides
5. Application modèle → deltas
6. Mise à jour grille + clamp [0,1]
7. Application contraintes

**Dépendances**: `nca_model.py`

---

### Module: Simulation (`simulator.py`)

#### `DiffusionSimulator`
**Responsabilité**: Simulation physique de diffusion de chaleur

**Pattern**: Singleton via `get_simulator()`

**Méthodes**:
- `step(grid, source_mask, obstacle_mask) -> torch.Tensor`: Diffusion classique (convolution)
- `generate_stage_sequence(stage_nb, n_steps, size) -> (sequence, source_mask, obstacle_mask)`: 
  - Génère position source aléatoire
  - Délègue génération obstacles à `ProgressiveObstacleManager`
  - Simule n_steps de diffusion
  - Retourne séquence temporelle complète

**Noyau de diffusion**: Moyenne 3×3 via convolution

**Dépendances**: `obstacles.py`, `config.py`, `torching.py`

---

### Module: Gestion Obstacles (`obstacles.py`)

#### `ProgressiveObstacleManager`
**Responsabilité**: Orchestration de la génération d'obstacles selon l'étape

**Méthodes**:
- `generate_stage_environment(stage_nb, size, source_pos) -> torch.Tensor`:
  - Délègue à `STAGE_MANAGER.get_stage(stage_nb)`
  - Retourne masque d'obstacles
  
- `get_difficulty_metrics(stage_nb, obstacle_mask) -> Dict`:
  - Calcul ratio d'obstacles
  - Score de complexité

**Pattern**: Façade vers le système de stages

**Dépendances**: `stage_manager.py`

---

### Module: Gestion des Étapes (`stage_manager.py`)

#### `StageManager`
**Responsabilité**: Registre et accès aux définitions d'étapes

**Pattern**: Singleton via `STAGE_MANAGER`

**Méthodes**:
- `get_stages() -> List[BaseStage]`: Liste toutes les étapes
- `get_stage(stage_nb: int) -> BaseStage`: Accès à une étape spécifique

**Initialisation**: Instancie les 3 stages et leur assigne leur numéro

**Dépendances**: `stages/*.py`

---

### Module: Stages (`stages/`)

#### `BaseStage` (Classe Abstraite)
**Responsabilité**: Interface commune pour toutes les étapes

**Attributs**:
- `NAME: str`: Identifiant technique
- `DISPLAY_NAME: str`: Nom d'affichage
- `_stage_nb: int`: Numéro d'étape (assigné par StageManager)

**Méthodes abstraites**:
- `generate_environment(size, source_pos) -> torch.Tensor`: Génération masque obstacles

**Méthodes concrètes**:
- `save_stage_checkpoint(metrics, model_state, optimizer_state)`: Sauvegarde checkpoint

**Dépendances**: `config.py`

---

#### `Stage1NoObstacle`
**Responsabilité**: Étape 1 - Apprentissage de base sans obstacles

**Implémentation**:
```python
def generate_environment(size, source_pos):
    return torch.zeros((size, size), dtype=torch.bool)  # Grille vide
```

---

#### `Stage2OneObstacle`
**Responsabilité**: Étape 2 - Introduction d'un obstacle unique

**Algorithme**:
1. Taille aléatoire obstacle: [MIN_OBSTACLE_SIZE, MAX_OBSTACLE_SIZE]
2. 100 tentatives de placement
3. Vérification non-chevauchement avec source
4. Placement du premier obstacle valide

---

#### `Stage3FewObstacles`
**Responsabilité**: Étape 3 - Gestion obstacles multiples (2-4)

**Algorithme**:
1. Nombre aléatoire d'obstacles: [2, 4]
2. Pour chaque obstacle:
   - Taille aléatoire
   - 50 tentatives de placement
   - Vérification non-chevauchement source
   - Vérification non-chevauchement obstacles existants
3. Validation connectivité via flood-fill
4. Si échec connectivité → fallback vers Stage2

**Méthodes spéciales**:
- `_validate_connectivity(obstacle_mask, source_pos) -> bool`: 
  - Flood-fill depuis source
  - Vérifie ≥50% grille libre accessible
  
- `_generate_stage_2_environment(...)`: Fallback vers 1 obstacle

---

### Module: Cache de Séquences (`sequences.py`)

#### `OptimizedSequenceCache`
**Responsabilité**: Gestion des caches de séquences d'entraînement par étape

**Architecture**:
- Cache séparé par étape: `stage_caches[stage_nb]`
- Tailles adaptatives: Stage 1→150, Stage 2→200, Stage 3→250
- Index circulaire pour itération

**Méthodes**:
- `initialize_stage_cache(stage_nb)`: 
  - Génère cache_size séquences via simulator
  - Stocke {target_seq, source_mask, obstacle_mask, stage_nb}
  
- `get_stage_batch(stage_nb, batch_size) -> List[Dict]`:
  - Retourne batch avec index circulaire
  
- `shuffle_stage_cache(stage_nb)`: Mélange aléatoire du cache
  
- `clear_stage_cache(stage_nb)`: Libération mémoire

**Optimisation**: Libération cache étape précédente pour économiser RAM

**Dépendances**: `simulator.py`, `config.py`

---

### Module: Planificateur Curriculum (`scheduler.py`)

#### `CurriculumScheduler`
**Responsabilité**: Gestion de la progression adaptative entre étapes

**État interne**:
- `stage_metrics_history[stage_nb]`: Historique métriques
- `no_improvement_counts[stage_nb]`: Compteur stagnation

**Méthodes**:

##### `should_advance_stage(current_stage, recent_losses) -> bool`
**Logique**:
1. Calcul amélioration: `losses[-2] - losses[-1]`
2. Si amélioration < `STAGNATION_THRESHOLD`: compteur++
3. Sinon: compteur = 0
4. Return: compteur ≥ `STAGNATION_PATIENCE`

**Critère**: Early stopping par stagnation

##### `adjust_learning_rate(optimizer, stage_nb, epoch_in_stage)`
**Stratégie multi-niveaux**:
1. **Réduction par étape**: 
   - Stage 1: 1.0×
   - Stage 2: 0.8×
   - Stage 3: 0.6×
   
2. **Cosine decay intra-étape**:
   ```
   cos_factor = 0.5 * (1 + cos(π * epoch / max_epochs))
   final_lr = stage_lr * (0.1 + 0.9 * cos_factor)
   ```
   - Minimum: 10% du LR de base
   - Décroissance douce

**Dépendances**: `config.py`

---

### Module: Entraîneur (`trainer.py`)

#### `ModularTrainer`
**Responsabilité**: Orchestration complète de l'entraînement modulaire

**Composants internes**:
- `model: ImprovedNCA`
- `updater: OptimizedNCAUpdater`
- `optimizer: AdamW` (weight_decay=1e-4)
- `curriculum: CurriculumScheduler`
- `sequence_cache: OptimizedSequenceCache`
- `loss_fn: MSELoss`

**État**:
- `current_stage: int`
- `stage_histories[stage_nb]`: {losses, epochs, lr}
- `global_history`: {losses, stages, epochs}
- `total_epochs_trained: int`

**Méthodes principales**:

##### `train_step(target_seq, source_mask, obstacle_mask, stage_nb) -> float`
**Algorithme**:
1. Initialisation grille avec source
2. Boucle temporelle (NCA_STEPS):
   - Mise à jour via `updater.step()`
   - Calcul loss vs target
   - Accumulation
3. Moyenne temporelle
4. Backpropagation + gradient clipping (max_norm=1.0)
5. Optimizer step

##### `train_stage(stage_nb, max_epochs) -> Dict[str, Any]`
**Workflow**:
1. Initialisation cache séquences
2. Boucle d'époques:
   - Ajustement LR (curriculum)
   - Mélange cache périodique (tous les 20 époques)
   - Entraînement par batch
   - Vérification avancement (curriculum.should_advance_stage)
   - Early stopping si stagnation
3. Sauvegarde checkpoint via `STAGE_MANAGER.get_stage().save_stage_checkpoint()`
4. Libération cache étape précédente

**Retour**: Métriques {stage_nb, epochs_trained, final_loss, early_stopped, loss_history}

##### `train_full_curriculum() -> Dict[str, Any]`
**Orchestration**:
1. Entraînement séquentiel stages 1→2→3
2. Agrégation métriques globales
3. Sauvegarde modèle final + métriques JSON

**Retour**: Métriques complètes incluant stage_histories, global_history

##### `save_final_model(global_metrics)`
**Sauvegarde**:
- `final_model.pth`: {model_state_dict, optimizer_state_dict, global_metrics, config}
- `complete_metrics.json`: Métriques en JSON

**Dépendances**: `nca_model.py`, `updater.py`, `scheduler.py`, `sequences.py`, `stage_manager.py`, `config.py`, `torching.py`

---

### Module: Point d'Entrée (`train.py`)

#### `main()`
**Responsabilité**: Orchestration complète du pipeline

**Workflow**:
1. **Initialisation**:
   - Seeds (torch, numpy)
   - Création répertoire sortie
   - Instanciation `ImprovedNCA`
   - Instanciation `ModularTrainer`

2. **Entraînement**:
   - `trainer.train_full_curriculum()`

3. **Visualisation**:
   - `ProgressiveVisualizer()`
   - Pour chaque stage: `visualize_stage_results(model, stage_nb)`
   - `create_curriculum_summary(metrics)`

4. **Rapport**:
   - Affichage résumé console
   - Détails par étape

**Gestion erreurs**: Try/except avec traceback complet

**Dépendances**: Tous les modules

---

### Module: Visualisation (`visualizer.py`)

#### `ProgressiveVisualizer`
**Responsabilité**: Génération de toutes les visualisations et rapports

**Méthodes principales**:

##### `visualize_stage_results(model, stage_nb)`
**Workflow**:
1. Génération séquence test (seed fixe VISUALIZATION_SEED)
2. Prédiction modèle (mode eval, no_grad)
3. Création animations:
   - GIF comparaison (cible vs NCA)
   - GIF NCA seul
4. Graphique convergence (erreur MSE temporelle)
5. Sauvegarde dans `stage_{stage_nb}/`

##### `create_curriculum_summary(global_metrics)`
**Génère 3 graphiques**:

1. **`_plot_curriculum_progression()`**:
   - Subplot 1: Losses par étape (couleurs différentes, log scale)
   - Subplot 2: Learning rate par étape (log scale)
   - Subplot 3: **Accélération LR** (dérivée seconde):
     - Détection points d'inflexion (changements de signe)
     - Zones rouge (décélération) / vert (accélération)
     - Annotations des inflexions majeures
     - Explication pédagogique

2. **`_plot_stage_comparison()`**:
   - Perte finale par étape
   - Époques prévues vs utilisées (barres)
   - Vitesse de convergence (line plot)

3. **`_plot_performance_metrics()`**:
   - Résumé textuel complet:
     - Statistiques globales (temps, époques, loss)
     - Performance par étape (détaillée)
     - Architecture (hyperparamètres)

**Méthodes utilitaires**:
- `_create_stage_animations(vis_data)`: Génération GIFs
- `_create_stage_convergence_plot(vis_data)`: Graphique erreur temporelle
- `_save_comparison_gif(...)`: GIF côte à côte
- `_save_single_gif(...)`: GIF simple

**Dépendances**: `config.py`, `updater.py`, `nca_model.py`, `simulator.py`, `matplotlib`

---

## Structure de Fichiers et Outputs

### Répertoire de Sortie: `outputs/`

```
outputs/
├── stage_1/
│   ├── model_checkpoint.pth              # Checkpoint modèle + optimizer + metrics
│   ├── metrics.json                      # Métriques étape 1 (JSON)
│   ├── animation_comparaison_étape_1.gif # GIF cible vs NCA
│   ├── animation_nca_étape_1.gif         # GIF NCA seul
│   └── convergence_étape_1.png           # Erreur MSE temporelle
│
├── stage_2/
│   ├── model_checkpoint.pth
│   ├── metrics.json
│   ├── animation_comparaison_étape_2.gif
│   ├── animation_nca_étape_2.gif
│   └── convergence_étape_2.png
│
├── stage_3/
│   ├── model_checkpoint.pth
│   ├── metrics.json
│   ├── animation_comparaison_étape_3.gif
│   ├── animation_nca_étape_3.gif
│   └── convergence_étape_3.png
│
├── final_model.pth                       # Modèle final complet
├── complete_metrics.json                 # Toutes les métriques consolidées
│
├── curriculum_progression.png            # 3 subplots: losses, LR, accélération
├── stage_comparison.png                  # Comparaison inter-étapes
└── performance_summary.png               # Résumé textuel complet
```

### Format `complete_metrics.json`

```json
{
  "total_epochs_planned": 300,
  "total_epochs_actual": 245,
  "total_time_seconds": 1234.56,
  "total_time_formatted": "20.6 min",
  "final_loss": 0.001234,
  
  "stage_metrics": {
    "1": {
      "stage_nb": 1,
      "epochs_trained": 80,
      "final_loss": 0.002,
      "early_stopped": true,
      "loss_history": [0.1, 0.05, ...]
    },
    "2": { ... },
    "3": { ... }
  },
  
  "stage_histories": {
    "1": {
      "losses": [...],
      "epochs": [...],
      "lr": [...]
    },
    "2": { ... },
    "3": { ... }
  },
  
  "global_history": {
    "losses": [...],      # Toutes les losses concaténées
    "stages": [...],      # Numéro stage pour chaque loss
    "epochs": [...]       # Époque globale
  },
  
  "stage_start_epochs": {
    "1": 0,
    "2": 80,
    "3": 165
  }
}
```

## Patterns et Bonnes Pratiques

### 1. Singleton Pattern
**Utilisé pour**: 
- `STAGE_MANAGER` (stage_manager.py)
- `get_simulator()` (simulator.py)
- `CONFIG` (config.py)

**Justification**: Un seul gestionnaire global évite les incohérences

### 2. Strategy Pattern
**Utilisé pour**: Stages (BaseStage + implémentations)

**Justification**: Chaque étape a son algorithme de génération d'obstacles

### 3. Facade Pattern
**Utilisé pour**: `ProgressiveObstacleManager`

**Justification**: Interface simple cachant complexité du système de stages

### 4. Template Method Pattern
**Utilisé pour**: `BaseStage.save_stage_checkpoint()` (méthode concrète dans classe abstraite)

**Justification**: Comportement commun avec points d'extension

### 5. Dependency Injection
**Utilisé pour**: `ModularTrainer(model)`, `OptimizedNCAUpdater(model)`

**Justification**: Flexibilité et testabilité

## Points d'Extension pour Futures Refactorisations

### 1. Nouvelles Étapes
**Comment ajouter une Stage 4**:
1. Créer `stages/stage_4_*.py` héritant de `BaseStage`
2. Implémenter `generate_environment()`
3. Ajouter dans `StageManager.__init__()`
4. Mettre à jour `CONFIG.STAGE_4_EPOCHS`

**Impact**: Minimal, architecture extensible

### 2. Nouveaux Critères de Curriculum
**Actuellement**: Stagnation uniquement

**Extensions possibles**:
- Convergence absolue (seuil de loss)
- Variance des losses
- Métriques de qualité (corrélation avec cible)

**Où modifier**: `CurriculumScheduler.should_advance_stage()`

### 3. Autres Architectures NCA
**Actuellement**: MLP dense

**Extensions possibles**:
- CNN pour features spatiales
- Attention mechanism
- Graph Neural Networks

**Où modifier**: `ImprovedNCA` (remplacer `nn.Sequential`)

**Interface stable**: `forward(x: torch.Tensor) -> torch.Tensor`

### 4. Modes de Visualisation
**Extensions possibles**:
- Heatmaps interactives
- Animations 3D
- Dashboards temps réel (TensorBoard)

**Où modifier**: `ProgressiveVisualizer` (ajouter méthodes)

### 5. Persistence et Reprise
**Actuellement**: Sauvegarde finale uniquement

**Extensions possibles**:
- Checkpoints intermédiaires
- Reprise après interruption
- Versioning des modèles

**Où modifier**: `ModularTrainer` (ajouter logique checkpointing)

## Conclusion

### Points Forts de l'Architecture v11
1. ✅ **Séparation responsabilités claire**
2. ✅ **Extensibilité** (ajout étapes facile)
3. ✅ **Patterns robustes** (Singleton, Strategy, Facade)
4. ✅ **Optimisations vectorielles** (F.unfold)
5. ✅ **Curriculum adaptatif** (early stopping intelligent)
6. ✅ **Visualisations complètes** (multiples graphiques)
7. ✅ **Reproductibilité** (seeds, déterminisme)

### Points d'Amélioration
1. ⚠️ **Visualizer trop monolithique** (450 lignes)
2. ⚠️ **Logging non structuré** (print au lieu de logging)
3. ⚠️ **Absence de tests** (0% couverture)
4. ⚠️ **Validation inputs manquante** (risque erreurs runtime)
5. ⚠️ **Documentation utilisateur** (manque README/quickstart)

### Prêt pour Refactorisation v12
L'architecture actuelle est **saine et bien structurée**. Les prochaines refactorisations peuvent se concentrer sur:
1. Découpage `visualizer.py`
2. Ajout système de logging
3. Tests automatisés (priorité haute)
4. Générateur lazy (optimisation mémoire)
5. Documentation utilisateur

**Aucune refactorisation majeure de l'architecture n'est nécessaire**. Les modifications seront incrémentales et ciblées.
