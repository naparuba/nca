# Description du Code - NCA Modulaire avec Intensités Variables (Version 8__)

## Vue d'ensemble du système

Ce document décrit l'architecture et l'API des classes du système **Neural Cellular Automaton (NCA) Modulaire avec Intensités Variables** (Version 8__). Le système implémente un apprentissage progressif en 4 étapes avec la nouveauté majeure de l'**étape 4** qui introduit les **intensités de source variables**.

### Innovation principale (Version 8__)
- **Étape 4 avec intensités variables** : Le système apprend avec des intensités de source différentes entre les simulations (mais fixes pendant chaque simulation)
- **Curriculum progressif d'intensité** : De [0.5, 1.0] vers [0.0, 1.0] au fil de l'entraînement
- **Support complet des intensités** dans tous les composants (simulateur, updaters, visualisations)

## 📋 Classes principales et API

### 1. `ModularConfig` - Configuration étendue

**Rôle** : Configuration centralisée pour l'apprentissage modulaire avec intensités variables.

**API publique** :
```python
def __init__(self, seed: int = 123)
```

**Propriétés principales** :
- `STAGE_1_RATIO = 0.3` : 30% des époques pour l'étape 1 (sans obstacles)
- `STAGE_2_RATIO = 0.3` : 30% des époques pour l'étape 2 (un obstacle)  
- `STAGE_3_RATIO = 0.2` : 20% des époques pour l'étape 3 (obstacles multiples)
- `STAGE_4_RATIO = 0.2` : 20% des époques pour l'étape 4 (intensités variables) ⭐ **NOUVEAU**

**Configuration intensités variables (NOUVEAU)** :
```python
VARIABLE_INTENSITY_TRAINING = True
MIN_SOURCE_INTENSITY = 0.0    # Intensité minimale (éteint)
MAX_SOURCE_INTENSITY = 1.0    # Intensité maximale (standard)
STAGE_4_SOURCE_CONFIG = {
    'intensity_distribution': 'uniform',
    'sample_per_simulation': True,
    'fixed_during_simulation': True,
    'intensity_range_expansion': True,
    'initial_range': [0.5, 1.0],  # Plage initiale
    'final_range': [0.0, 1.0]     # Plage finale
}
```

**Seuils de convergence par étape** :
```python
CONVERGENCE_THRESHOLDS = {
    1: 0.0002,  # Très strict pour base solide
    2: 0.0002,  # Très strict pour adaptation obstacles
    3: 0.0002,  # Strict pour complexité
    4: 0.0002   # Strict pour intensités variables ⭐ NOUVEAU
}
```

---

### 2. `ProgressiveObstacleManager` - Gestionnaire d'obstacles progressifs

**Rôle** : Génère des environnements d'obstacles adaptés à chaque étape d'apprentissage.

**API publique** :
```python
def __init__(self, device: str = cfg.DEVICE)

def generate_stage_environment(self, stage: int, size: int, source_pos: Tuple[int, int], 
                              seed: Optional[int] = None) -> torch.Tensor
```

**Méthodes spécialisées par étape** :
- `_generate_stage_1_environment()` : Grille vide (apprentissage de base)
- `_generate_stage_2_environment()` : Un seul obstacle (contournement)
- `_generate_stage_3_environment()` : 2-4 obstacles (complexité)
- `_generate_stage_4_environment()` : 1-2 obstacles (intensités variables) ⭐ **NOUVEAU**

**Validation intelligente** :
```python
def _validate_connectivity(self, obstacle_mask: torch.Tensor, source_pos: Tuple[int, int]) -> bool
def get_difficulty_metrics(self, stage: int, obstacle_mask: torch.Tensor) -> Dict[str, float]
```

**Logique progressive** :
- **Étape 1** : Aucun obstacle pour apprendre la diffusion pure
- **Étape 2** : Un obstacle avec validation de non-chevauchement source
- **Étape 3** : Obstacles multiples avec validation de connectivité (flood-fill)
- **Étape 4** : Configuration simplifiée pour focus sur intensités variables

---

### 3. `DiffusionSimulator` - Simulateur adapté intensités variables

**Rôle** : Simulateur de diffusion thermique avec support des intensités variables (Version 8__).

**API publique** :
```python
def __init__(self, device: str = cfg.DEVICE)

def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor,
         source_intensity: Optional[float] = None) -> torch.Tensor

def generate_stage_sequence(self, stage: int, n_steps: int, size: int,
                           seed: Optional[int] = None, 
                           source_intensity: Optional[float] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[float]]
```

**Innovation Version 8__** :
- **Paramètre `source_intensity`** : Support intensité variable pour étape 4
- **Retour étendu** : Tuple avec 4 éléments incluant l'intensité utilisée
- **Logique adaptative** :
  - **Étapes 1-3** : `source_intensity=None` → intensité standard (1.0)
  - **Étape 4** : `source_intensity` spécifiée → intensité variable

**Algorithme de diffusion** :
```python
# Convolution 3x3 pour diffusion
new_grid = F.conv2d(grid, kernel_3x3, padding=1)
# Contraintes obstacles
new_grid[obstacle_mask] = 0.0
# ⭐ NOUVEAU : Support intensité variable
if source_intensity is not None:
    new_grid[source_mask] = source_intensity  # Étape 4
else:
    new_grid[source_mask] = grid[source_mask]  # Étapes 1-3
```

---

### 4. `ImprovedNCA` - Modèle Neural Cellular Automaton

**Rôle** : Réseau de neurones profond pour apprendre les règles de diffusion.

**API publique** :
```python
def __init__(self, input_size: int = 11, hidden_size: int = 128, n_layers: int = 3)
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Architecture** :
- **Entrée** : 11 features (patch 3x3 + source + obstacle)
- **Corps** : 3 couches cachées avec BatchNorm + ReLU + Dropout
- **Sortie** : 1 valeur (delta) avec Tanh + scaling (0.1)

**Caractéristiques** :
- **Stabilité** : BatchNorm et scaling des deltas pour éviter l'instabilité
- **Régularisation** : Dropout 0.1 pour éviter le surapprentissage
- **Capacité** : ~128k paramètres pour apprendre les 4 étapes

---

### 5. `OptimizedNCAUpdater` & `NCAUpdater` - Application du modèle

**Rôle** : Appliquent le modèle NCA sur une grille avec optimisations.

#### `OptimizedNCAUpdater` (Version optimisée)
```python
def __init__(self, model: ImprovedNCA, device: str = cfg.DEVICE)
def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor,
         source_intensity: Optional[float] = None) -> torch.Tensor
```

**Innovation Version 8__** :
- **Paramètre `source_intensity`** : Support intensité variable
- **Extraction vectorisée** : `F.unfold()` pour patches 3x3 optimisée
- **Application sélective** : Seulement sur cellules non-obstacles

#### `NCAUpdater` (Version standard)
```python
def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor,
         source_intensity: Optional[float] = None) -> torch.Tensor
```

**Logique commune** :
```python
# ⭐ NOUVEAU : Support intensité variable dans les deux updaters
if source_intensity is not None:
    new_grid[source_mask] = source_intensity  # Étape 4
else:
    new_grid[source_mask] = grid[source_mask]  # Étapes 1-3
```

---

### 6. `CurriculumScheduler` - Planificateur de progression

**Rôle** : Gère la progression automatique entre les étapes avec logique renforcée.

**API publique** :
```python
def __init__(self, convergence_thresholds: Dict[int, float], patience: int = 10)

def should_advance_stage(self, current_stage: int, recent_losses: List[float]) -> bool
def adjust_learning_rate(self, optimizer, stage: int, epoch_in_stage: int)
def get_stage_loss_weights(self, stage: int) -> Dict[str, float]
```

**Logique de convergence renforcée** :
- **Critère étendu** : 10 époques minimum (était 5)
- **Convergence ET stabilité** : Moyenne < seuil ET variance < 0.001
- **Patience doublée** : 20 époques de stagnation avant avancement forcé
- **Learning rate adaptatif** : Réduction progressive par étape avec cosine decay

**Poids de perte par étape** :
```python
weights = {
    1: {'mse': 1.0, 'convergence': 2.0, 'stability': 1.0},
    2: {'mse': 1.0, 'convergence': 1.5, 'stability': 1.5, 'adaptation': 1.0},
    3: {'mse': 1.0, 'convergence': 1.0, 'stability': 2.0, 'robustness': 1.5},
    4: {'mse': 1.0, 'convergence': 1.2, 'stability': 2.5, 'robustness': 2.0}  # ⭐ NOUVEAU
}
```

---

### 7. `OptimizedSequenceCache` - Cache par étape

**Rôle** : Cache spécialisé par étape pour optimiser l'entraînement.

**API publique** :
```python
def __init__(self, simulator: DiffusionSimulator, device: str = cfg.DEVICE)

def initialize_stage_cache(self, stage: int)
def get_stage_batch(self, stage: int, batch_size: int)
def shuffle_stage_cache(self, stage: int)
def clear_stage_cache(self, stage: int)
```

**Stratégie de cache** :
- **Étape 1** : 150 séquences (base simple)
- **Étape 2** : 200 séquences (variété obstacles)
- **Étape 3** : 250 séquences (complexité élevée)
- **Étape 4** : **PAS DE CACHE** ⭐ (intensités variables incompatibles)

**Gestion mémoire** :
- **Libération progressive** : Cache étape N-1 libéré quand étape N commence
- **Mélange périodique** : Évite la mémorisation de l'ordre

---

### 8. `SimulationIntensityManager` - Gestionnaire d'intensités ⭐ **NOUVEAU**

**Rôle** : Gère le curriculum d'intensités progressif pour l'étape 4.

**API publique** :
```python
def __init__(self, device: str = cfg.DEVICE)

def sample_simulation_intensity(self, epoch_progress: float) -> float
def get_progressive_range(self, epoch_progress: float) -> Tuple[float, float]
def validate_intensity(self, intensity: float) -> float
def get_intensity_statistics(self) -> Dict[str, float]
def clear_history(self)
```

**Curriculum progressif** :
```python
# Progression : 0.0 = début étape 4, 1.0 = fin étape 4
epoch_progress = epoch / max_epochs

# Interpolation linéaire entre plages
# 0% : [0.5, 1.0] → Sources moyennes à fortes
# 50% : [0.25, 1.0] → Ajout sources faibles  
# 100% : [0.0, 1.0] → Toute la plage (y compris éteint)
```

**Validation intelligente** :
- **Clipping** : Assure [0.0, 1.0]
- **Évite quasi-zéro** : Si 0 < intensity < 0.001 → intensity = 0.001
- **Statistiques** : Historique pour analyse et debugging

---

### 9. `ModularTrainer` - Entraîneur principal

**Rôle** : Orchestre l'entraînement modulaire progressif en 4 étapes.

**API publique** :
```python
def __init__(self, model: ImprovedNCA, device: str = cfg.DEVICE)

def train_step(self, target_sequence: List[torch.Tensor], source_mask: torch.Tensor,
               obstacle_mask: torch.Tensor, stage: int, 
               source_intensity: Optional[float] = None) -> float

def train_stage(self, stage: int, max_epochs: int) -> Dict[str, Any]
def train_stage_4(self, max_epochs: int) -> Dict[str, Any]  # ⭐ NOUVEAU
def train_full_curriculum(self) -> Dict[str, Any]

def save_stage_checkpoint(self, stage: int, metrics: Dict[str, Any])
def save_final_model(self, global_metrics: Dict[str, Any])
```

**Innovation Version 8__ - Méthode spécialisée `train_stage_4()`** :
```python
for epoch_in_stage in range(max_epochs):
    epoch_progress = epoch_in_stage / max(max_epochs - 1, 1)
    
    for batch_idx in range(cfg.BATCH_SIZE):
        # ⭐ Échantillonne intensité pour cette simulation
        current_intensity = self.intensity_manager.sample_simulation_intensity(epoch_progress)
        
        # Génère séquence avec intensité fixe
        target_seq, source_mask, obstacle_mask, used_intensity = simulator.generate_stage_sequence(
            stage=4, source_intensity=current_intensity
        )
        
        # Entraîne avec cette intensité fixe
        loss = self.train_step(target_seq, source_mask, obstacle_mask, stage, current_intensity)
```

**Métriques étendues pour étape 4** :
```python
stage_metrics = {
    'stage': 4,
    'epochs_trained': epoch_in_stage + 1,
    'final_loss': final_loss,
    'convergence_met': convergence_met,
    'early_stopped': early_stop,
    'loss_history': stage_losses,
    'intensity_stats_history': intensity_stats_history,  # ⭐ NOUVEAU
    'global_intensity_stats': global_intensity_stats     # ⭐ NOUVEAU
}
```

**Gestion des historiques** :
- **Par étape** : `stage_histories[1-4]` avec pertes, époques, learning rates
- **Global** : `global_history` avec progression continue
- **Checkpoints** : Sauvegarde modèle + métriques à chaque étape

---

### 10. `ProgressiveVisualizer` - Système de visualisation

**Rôle** : Génère visualisations et animations adaptées à chaque étape.

**API publique** :
```python
def __init__(self, interactive: bool = interactive_mode)

def visualize_stage_results(self, model: ImprovedNCA, stage: int, 
                           vis_seed: int = 3333) -> Dict[str, Any]
def create_curriculum_summary(self, global_metrics: Dict[str, Any])

# Méthodes privées pour génération
def _create_stage_animations(self, vis_data: Dict[str, Any])
def _create_stage_convergence_plot(self, vis_data: Dict[str, Any])
def _save_comparison_gif(self, target_seq, nca_seq, obstacle_mask, filepath, title, source_intensity)
def _save_single_gif(self, sequence, obstacle_mask, filepath, title, source_intensity)
```

**Innovation Version 8__ - Affichage d'intensité** :
```python
# ⭐ Tous les GIFs affichent l'intensité dans le titre
ax.set_title(f'{title} - t={frame} (I={source_intensity:.3f})')

# Pour étape 4 : intensité variable échantillonnée
# Pour étapes 1-3 : intensité standard (I=1.000)
```

**Types de visualisations** :
- **Animations comparatives** : Cible vs NCA côte à côte
- **Animations NCA** : Prédiction seule avec obstacles
- **Graphiques de convergence** : Erreur MSE vs temps avec seuils
- **Résumé curriculum** : Progression globale avec codes couleur par étape

---

## 🔄 Flux d'exécution principal

### 1. Initialisation
```python
cfg = ModularConfig(seed=123)
model = ImprovedNCA().to(device)
trainer = ModularTrainer(model, device)
```

### 2. Entraînement modulaire (4 étapes)
```python
global_metrics = trainer.train_full_curriculum()
# Étape 1: Sans obstacles (150 époques)
# Étape 2: Un obstacle (150 époques) 
# Étape 3: Obstacles multiples (100 époques)
# Étape 4: Intensités variables (100 époques) ⭐ NOUVEAU
```

### 3. Visualisations progressives
```python
visualizer = ProgressiveVisualizer()
for stage in [1, 2, 3, 4]:
    visualizer.visualize_stage_results(model, stage)
visualizer.create_curriculum_summary(global_metrics)
```

## 🎯 Points clés pour évolution future

### Architecture modulaire
- **Classes découplées** : Chaque classe a une responsabilité claire
- **Interfaces stables** : APIs compatibles pour extensions futures
- **Configuration centralisée** : Facilite l'ajout de nouveaux paramètres

### Support intensités variables
- **Propagation complète** : Tous les composants supportent `source_intensity`
- **Rétrocompatibilité** : `source_intensity=None` → comportement v7__
- **Extensibilité** : Base pour futures extensions (sources multiples, variations temporelles)

### Optimisations et robustesse
- **Cache intelligent** : Optimise étapes 1-3, désactivé pour étape 4
- **Validation rigoureuse** : Connectivité, intensités, convergence
- **Gestion erreurs** : Fallbacks et nettoyage mémoire

### Métriques et debugging
- **Historiques détaillés** : Par étape et global
- **Statistiques intensité** : Analyse complète du curriculum d'intensité
- **Visualisations riches** : GIFs avec intensité, graphiques multi-étapes

Ce système constitue une base solide pour l'évolution vers des NCA plus complexes avec gestion avancée des sources variables.
