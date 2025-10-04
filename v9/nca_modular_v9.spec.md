# Spécification NCA Modulaire v9 - Architecture Découplée avec Stages Extensibles

## Vue d'ensemble

Le NCA Modulaire v9 représente une refonte architecturale majeure du système v8, introduisant une **architecture découplée et extensible** basée sur des stages modulaires autonomes. Cette version privilégie la séparation des responsabilités, l'extensibilité et la maintenabilité du code.

### Innovation Principale : Architecture Modulaire Découplée

- **Stages autonomes** : Chaque stage gère sa propre configuration, logique d'entraînement et validation
- **Registre de stages** : Système d'enregistrement permettant l'ajout facile de nouveaux stages
- **Interface standardisée** : Tous les stages implémentent l'interface `BaseStage`
- **Séparation visualisation/génération** : Architecture claire entre génération de données et visualisation

## Spécifications Fonctionnelles

### 1. Architecture à 4 Stages Modulaires

#### Stage 1 : Apprentissage de Base (Sans Obstacles)
**Objectif** : Établir les bases de la diffusion thermique pure
- **Environnement** : Grille vide avec source centrale d'intensité fixe 1.0
- **Durée** : 30% du temps total (150 époques sur 500)
- **Convergence** : Seuil strict de 0.0002 pour une base solide
- **Learning Rate** : Taux standard (multiplicateur 1.0)
- **Obstacles** : Aucun (min=0, max=0)

#### Stage 2 : Introduction d'Obstacles Uniques
**Objectif** : Apprendre le contournement d'obstacles simples
- **Environnement** : Un obstacle rectangulaire, source d'intensité 1.0
- **Durée** : 30% du temps total (150 époques sur 500)
- **Convergence** : Seuil strict de 0.0002
- **Learning Rate** : Réduit (multiplicateur 0.8)
- **Obstacles** : Un seul (min=1, max=1)

#### Stage 3 : Gestion d'Obstacles Complexes
**Objectif** : Maîtriser les configurations multi-obstacles
- **Environnement** : 2-4 obstacles variés, source d'intensité 1.0
- **Durée** : 20% du temps total (100 époques sur 500)
- **Convergence** : Seuil plus tolérant de 0.015
- **Learning Rate** : Fortement réduit (multiplicateur 0.6)
- **Obstacles** : Multiples (min=2, max=4)

#### Stage 4 : Intensités Variables (Innovation)
**Objectif** : Adaptation aux intensités de source variables
- **Environnement** : 1-2 obstacles, **intensité variable entre simulations**
- **Durée** : 20% du temps total (100 époques sur 500)
- **Innovation** : 
  - Intensité **fixe pendant chaque simulation** (0.0 à 1.0)
  - Intensité **différente à chaque nouvelle simulation**
  - Curriculum progressif : [0.5,1.0] → [0.0,1.0]
- **Convergence** : Seuil adapté de 0.001
- **Learning Rate** : Très réduit (multiplicateur 0.4)
- **Obstacles** : Limités (min=1, max=2)

### 2. Curriculum d'Apprentissage Progressif

#### Étape 1-3 : Complexité Croissante des Obstacles
- **Progression** : Sans obstacles → Un obstacle → Obstacles multiples
- **Intensité** : Fixe à 1.0 pour toutes les étapes
- **Apprentissage** : Complexité spatiale croissante

#### Étape 4 : Curriculum d'Intensité Variable
```
Phase 1 (0-25% des époques)  : [0.5, 1.0] - Sources moyennes à fortes
Phase 2 (25-50% des époques) : [0.3, 1.0] - Ajout des sources faibles  
Phase 3 (50-75% des époques) : [0.1, 1.0] - Sources très faibles
Phase 4 (75-100% des époques): [0.0, 1.0] - Plage complète (source éteinte incluse)
```

### 3. Cas d'Usage et Applications

#### Applications Industrielles
- **Systèmes de chauffage** : Équipements de puissances variables (0% à 100%)
- **Contrôle thermique** : Adaptation aux conditions opérationnelles variées
- **Maintenance prédictive** : Fonctionnement avec équipements dégradés
- **Optimisation énergétique** : Gestion efficace des ressources

#### Avantages de l'Approche Modulaire
- **Extensibilité** : Ajout facile de nouveaux stages sans impact sur l'existant
- **Maintenabilité** : Code découplé et responsabilités claires
- **Testabilité** : Chaque stage peut être testé indépendamment
- **Réutilisabilité** : Stages réutilisables dans d'autres contextes

## Spécifications Techniques

### 1. Architecture Logicielle Modulaire

#### Interface BaseStage
```python
class BaseStage(ABC):
    """Interface commune pour tous les stages d'entraînement"""
    
    @abstractmethod
    def generate_environment(self, size: int, source_pos: Tuple[int, int], 
                           seed: Optional[int] = None) -> torch.Tensor
    
    @abstractmethod  
    def should_converge(self, losses: List[float], patience: int = 10) -> bool
    
    @abstractmethod
    def get_training_metrics(self) -> Dict[str, Any]
```

#### Configuration par Stage
```python
@dataclass
class StageConfig:
    stage_id: int
    name: str
    description: str
    epochs_ratio: float = 0.25
    convergence_threshold: float = 0.0002
    learning_rate_multiplier: float = 1.0
    min_obstacles: int = 0
    max_obstacles: int = 0
```

#### Registre de Stages Extensible
```python
class StageRegistry:
    """
    Système d'enregistrement des stages pour extensibilité maximale.
    Permet l'ajout de nouveaux stages sans modification du code existant.
    """
    
    def register_stage(self, stage_id: int, stage_class: Type[BaseStage])
    def get_stage_class(self, stage_id: int) -> Type[BaseStage]
    def list_available_stages(self) -> Dict[int, str]
```

### 2. Gestionnaire de Stages Modulaire

#### ModularStageManager
```python
class ModularStageManager:
    """
    Orchestrateur principal de l'entraînement modulaire.
    Gère l'exécution séquentielle des stages de manière découplée.
    """
    
    def __init__(self, global_config, device: str):
        self.registry = StageRegistry()
        self.global_config = global_config
        self.device = device
        self.training_metrics = {}
    
    def train_sequence(self, model, simulator, sequence: Optional[List[int]] = None) -> Dict[str, Any]
    def train_single_stage(self, stage: BaseStage, model, simulator, max_epochs: int) -> Dict[str, Any]
    def get_global_metrics(self) -> Dict[str, Any]
```

### 3. Simulateur de Diffusion Modulaire

#### Interface Unifiée avec Support Intensité Variable
```python
class ModularDiffusionSimulator:
    """
    Simulateur adapté à l'architecture modulaire avec support intensités variables.
    """
    
    def step(self, grid: torch.Tensor, source_mask: torch.Tensor,
             obstacle_mask: torch.Tensor, source_intensity: Optional[float] = None) -> torch.Tensor
             
    def generate_sequence_with_stage(self, stage: BaseStage, n_steps: int, size: int,
                                   source_intensity: Optional[float] = None,
                                   seed: Optional[int] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, float]
```

### 4. Architecture de Visualisation Découplée

#### Séparation Claire Génération/Visualisation
```
v9/
├── nca_modular_v9.py              # Logique principale et génération
├── stages/                        # Architecture modulaire
│   ├── base_stage.py             # Interface commune
│   ├── stage1.py à stage4.py     # Implémentations spécialisées
│   ├── stage_manager.py          # Orchestrateur
│   └── visualizers/              # Système de visualisation découplé
│       ├── progressive_visualizer.py    # Visualiseur principal
│       ├── intensity_animator.py        # Animations avec intensités
│       ├── metrics_plotter.py           # Graphiques spécialisés
│       └── visualization_suite.py       # Suite complète
```

#### Composants de Visualisation
```python
# Visualiseur progressif principal
class ProgressiveVisualizer:
    def visualize_stage_results(model, stage, simulator, cfg, source_intensity=None)
    def create_intensity_comparison_grid(model, simulator, cfg)
    def create_curriculum_summary_extended(global_metrics, cfg)

# Animateur spécialisé intensités
class IntensityAwareAnimator:
    def create_intensity_labeled_gif(sequence, obstacle_mask, source_intensity, filepath)
    def create_comparison_with_intensity(target_seq, nca_seq, obstacle_mask, source_intensity)

# Graphiques de métriques spécialisées
class VariableIntensityMetricsPlotter:
    def plot_intensity_distribution(intensity_history, output_dir)
    def plot_performance_by_intensity_range(metrics_by_intensity, output_dir)
    def plot_convergence_analysis_by_intensity(convergence_data, output_dir)
```

### 5. Configuration Globale Simplifiée

#### GlobalConfig Épurée
```python
class GlobalConfig:
    """
    Configuration globale simplifiée - les détails sont gérés par les stages.
    """
    # Paramètres matériels
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 123
    
    # Paramètres de grille
    GRID_SIZE: int = 16
    SOURCE_INTENSITY: float = 1.0
    
    # Paramètres d'entraînement globaux
    TOTAL_EPOCHS: int = 500
    NCA_STEPS: int = 20
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 4
    
    # Architecture du modèle
    HIDDEN_SIZE: int = 128
    N_LAYERS: int = 3
    
    # Paramètres de visualisation
    POSTVIS_STEPS: int = 50
    OUTPUT_DIR: str = "nca_outputs_modular_v9"
    
    # Optimisations
    USE_OPTIMIZATIONS: bool = True
    USE_VECTORIZED_PATCHES: bool = True
```

### 6. Architecture de l'Intelligence Artificielle (Neural Cellular Automaton)

#### Modèle NCA - Spécifications Techniques Détaillées

**Type d'IA** : Neural Cellular Automaton (Automate Cellulaire Neuronal)
- **Paradigme** : Apprentissage local distribué avec règles cellulaires apprises
- **Architecture** : Réseau de neurones dense (fully connected) multi-couches
- **Application** : Prédiction de deltas locaux pour simulation de diffusion thermique

#### Structure des Canaux d'Entrée (11 Canaux Total)

**Canaux de Contexte Spatial (9 canaux) :**
```python
# Extraction des patches 3x3 via F.unfold()
grid_padded = F.pad(grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
patches = F.unfold(grid_padded, kernel_size=3, stride=1)  # Shape: [1, 9, H*W]

# Organisation des 9 canaux spatiaux :
Canal 0: Voisin Nord-Ouest     [i-1, j-1]
Canal 1: Voisin Nord           [i-1, j  ]  
Canal 2: Voisin Nord-Est       [i-1, j+1]
Canal 3: Voisin Ouest          [i  , j-1]
Canal 4: Cellule Centrale      [i  , j  ]  ← VALEUR ACTUELLE
Canal 5: Voisin Est            [i  , j+1]
Canal 6: Voisin Sud-Ouest      [i+1, j-1]
Canal 7: Voisin Sud            [i+1, j  ]
Canal 8: Voisin Sud-Est        [i+1, j+1]
```

**Canaux de Contraintes (2 canaux) :**
```python
# Ajout des contraintes d'environnement
Canal 9 : source_mask.flatten().float()    # 1.0 si source, 0.0 sinon
Canal 10: obstacle_mask.flatten().float()  # 1.0 si obstacle, 0.0 sinon

# Concaténation finale
full_patches = torch.cat([patches, source_flat, obstacle_flat], dim=1)  # [N, 11]
```

#### Architecture Neuronale Détaillée

**Classe `ImprovedNCA` - Spécifications Complètes :**

```python
class ImprovedNCA(nn.Module):
    def __init__(self, input_size: int = 11, hidden_size: int = 128, n_layers: int = 3):
        # COUCHE D'ENTRÉE : 11 → 128 neurones
        # COUCHES CACHÉES : 128 → 128 neurones (répété n_layers-1 fois)  
        # COUCHE DE SORTIE : 128 → 1 neurone (delta de température)
```

**Architecture Détaillée par Couche :**

1. **Couche d'Entrée** :
   - **Entrée** : 11 canaux (9 spatial + 2 contraintes)
   - **Sortie** : 128 neurones
   - **Transformation** : `Linear(11, 128)`
   - **Normalisation** : `BatchNorm1d(128)`
   - **Activation** : `ReLU()`
   - **Régularisation** : `Dropout(0.1)` - 10% de neurones désactivés

2. **Couches Cachées** (répétées n_layers-1 fois, défaut = 2 couches) :
   - **Entrée** : 128 neurones
   - **Sortie** : 128 neurones  
   - **Transformation** : `Linear(128, 128)`
   - **Normalisation** : `BatchNorm1d(128)`
   - **Activation** : `ReLU()`
   - **Régularisation** : `Dropout(0.1)`

3. **Couche de Sortie** :
   - **Entrée** : 128 neurones
   - **Sortie** : 1 neurone (delta de température)
   - **Transformation** : `Linear(128, 1)`
   - **Activation finale** : `Tanh()` → sortie dans [-1, 1]
   - **Mise à l'échelle** : `delta * 0.1` → sortie finale dans [-0.1, 0.1]

#### Paramètres du Modèle

**Nombre Total de Paramètres :**
```python
# Calcul détaillé des paramètres
Couche 1 (11→128) :  (11 × 128) + 128 bias = 1,536 paramètres
                     + 128 × 2 (BatchNorm)  = 256 paramètres
Couche 2 (128→128):  (128 × 128) + 128 bias = 16,512 paramètres  
                     + 128 × 2 (BatchNorm)  = 256 paramètres
Couche 3 (128→128):  (128 × 128) + 128 bias = 16,512 paramètres
                     + 128 × 2 (BatchNorm)  = 256 paramètres
Couche 4 (128→1)  :  (128 × 1) + 1 bias    = 129 paramètres

TOTAL ≈ 35,457 paramètres entraînables
```

**Configuration par Défaut :**
- **input_size** = 11 canaux
- **hidden_size** = 128 neurones par couche cachée
- **n_layers** = 3 couches (1 entrée + 2 cachées + 1 sortie)
- **delta_scale** = 0.1 (facteur de mise à l'échelle)

#### Processus de Traitement

**Pipeline de Traitement Complet :**

1. **Extraction de Contexte** :
   ```python
   # Pour chaque cellule [i,j], extraction du voisinage 3x3
   # Padding réplicatif pour les bords
   # Vectorisation via F.unfold() pour efficacité GPU
   ```

2. **Ajout de Contraintes** :
   ```python
   # Concaténation des masques source/obstacle
   # Information contextuelle sur le type de cellule
   ```

3. **Prédiction de Delta** :
   ```python
   # Application du réseau neuronal
   # Sortie : variation de température [-0.1, +0.1]
   ```

4. **Mise à Jour avec Contraintes** :
   ```python
   new_grid[i,j] = old_grid[i,j] + delta[i,j]
   new_grid = torch.clamp(new_grid, 0.0, 1.0)  # Bornes physiques
   new_grid[obstacle_mask] = 0.0               # Obstacles froids
   new_grid[source_mask] = source_intensity    # Source fixe
   ```

#### Compromis et Choix Architecturaux

**Compromis Spatialité vs Efficacité :**
- **Contexte 3x3** : Balance entre information locale suffisante et complexité
- **Alternative écartée** : Contexte 5x5 (25 canaux) → trop complexe
- **Alternative écartée** : Contexte 1x1 (1 canal) → information insuffisante

**Compromis Profondeur vs Overfitting :**
- **3 couches** : Sufficient pour apprendre des règles complexes
- **128 neurones** : Balance entre capacité et généralisation
- **Dropout 0.1** : Régularisation légère sans pertes de performance

**Compromis Delta vs Stabilité :**
- **delta_scale = 0.1** : Évite les oscillations tout en permettant l'évolution
- **Activation Tanh** : Bornes naturelles pour la stabilité numérique
- **Clamp [0,1]** : Respect des contraintes physiques (température normalisée)

#### Support Intensité Variable (Innovation Stage 4)

**Mécanisme d'Adaptation :**
```python
# Le modèle apprend à prédire des deltas appropriés
# indépendamment de l'intensité de source
# Adaptation via les canaux de contrainte (canal 9: source_mask)
# L'intensité variable est appliquée APRÈS la prédiction

if source_intensity is not None:
    new_grid[source_mask] = source_intensity  # Override post-prédiction
else:
    new_grid[source_mask] = grid[source_mask]  # Conservation standard
```

**Apprentissage Multi-Intensité :**
- **Stage 1-3** : Apprentissage avec intensité fixe 1.0
- **Stage 4** : Généralisation aux intensités [0.0, 1.0]
- **Mécanisme** : Le modèle apprend que les deltas doivent être cohérents avec l'intensité locale
- **Robustesse** : Gestion du cas limite intensité = 0.0 (source éteinte)

#### Optimisations Computationnelles

**Vectorisation Avancée :**
```python
# Traitement parallèle de toutes les cellules
# Extraction batch des patches 3x3 via unfold()
# Application vectorisée du réseau neuronal
# Pas de boucles Python → Performance GPU optimale
```

**Gestion Mémoire :**
```python
# Masquage des cellules obstacles pour économiser le calcul
valid_mask = ~obstacle_mask.flatten()
if valid_mask.any():
    valid_patches = full_patches[valid_mask]  # Seulement les cellules actives
    deltas = self.model(valid_patches)        # Calcul réduit
```

**Complexité Algorithmique :**
- **Spatiale** : O(H × W) pour une grille H×W
- **Temporelle** : O(T × H × W) pour T pas de temps
- **Paramètres** : O(1) - indépendant de la taille de grille
- **Parallélisation** : Parfaite pour GPU (opérations vectorisées)

## Validation et Tests

### 1. Critères de Validation par Stage

#### Stage 1 : Diffusion Pure
- **Convergence uniforme** : Distribution homogène dans toute la grille
- **Stabilité temporelle** : Pas de fluctuations après convergence
- **Reproductibilité** : Résultats identiques avec même seed

#### Stage 2 : Obstacle Unique
- **Contournement efficace** : Diffusion autour de l'obstacle
- **Conservation d'énergie** : Bilan thermique cohérent
- **Adaptation rapide** : Convergence malgré la complexité ajoutée

#### Stage 3 : Obstacles Multiples
- **Gestion de complexité** : Performance sur configurations variées
- **Connectivité préservée** : Validation des chemins de diffusion
- **Robustesse** : Fonctionnement sur diverses géométries

#### Stage 4 : Intensités Variables
- **Adaptation universelle** : Performance uniforme sur toute la plage [0.0, 1.0]
- **Cohérence physique** : Comportement proportionnel à l'intensité
- **Cas limites** : Gestion correcte de l'intensité 0.0 (source éteinte)

### 2. Tests Systémiques

#### Tests d'Intégration
- **Pipeline complet** : Entraînement des 4 stages consécutifs
- **Métriques cohérentes** : Validation des calculs de performance
- **Visualisations correctes** : Génération sans erreur de tous les graphiques

#### Tests de Régression
- **Compatibilité ascendante** : Interface stable pour les utilisateurs
- **Performance maintenue** : Pas de dégradation vs versions précédentes
- **Extensibilité validée** : Ajout de stages sans impact sur l'existant

## Configuration Recommandée de Production

### Paramètres d'Entraînement Optimaux
```python
# Configuration globale recommandée
TOTAL_EPOCHS = 500
GRID_SIZE = 16
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
N_LAYERS = 3

# Répartition optimale par stage
STAGE_1_RATIO = 0.30  # 150 époques - Base solide
STAGE_2_RATIO = 0.30  # 150 époques - Obstacles simples
STAGE_3_RATIO = 0.20  # 100 époques - Obstacles complexes  
STAGE_4_RATIO = 0.20  # 100 époques - Intensités variables

# Seuils de convergence validés
CONVERGENCE_THRESHOLDS = {
    1: 0.0002,  # Très strict pour la base
    2: 0.0002,  # Strict pour la robustesse
    3: 0.015,   # Tolérant pour la complexité
    4: 0.001    # Adapté aux intensités variables
}
```

### Optimisations de Performance
```python
# Activations recommandées
USE_OPTIMIZATIONS = True
USE_VECTORIZED_PATCHES = True
USE_SEQUENCE_CACHE = True  # Sauf Stage 4

# Configuration matérielle
DEVICE = "cuda"  # GPU requis pour performance optimale
TORCH_BACKENDS_CUDNN_BENCHMARK = True
```

### Architecture Évolutive

#### Plugins et Extensions
- **Système de plugins** : Chargement dynamique de nouveaux stages
- **API standardisée** : Interface stable pour développeurs tiers
- **Marketplace** : Écosystème de stages spécialisés

#### Intégration Continue
- **Tests automatisés** : Validation continue de l'extensibilité
- **Documentation dynamique** : Génération automatique des spécifications
- **Benchmarks** : Suivi de performance au fil des évolutions

## Conclusion

Le NCA Modulaire v9 représente une évolution architecturale majeure privilégiant l'extensibilité, la maintenabilité et la séparation des responsabilités. Cette architecture modulaire découplée permet :

1. **Ajout trivial de nouveaux stages** sans impact sur l'existant
2. **Maintenance simplifiée** grâce au découplage des composants  
3. **Testabilité maximale** avec des unités indépendantes
4. **Visualisation séparée** pour une architecture claire
5. **Performance optimisée** par des optimisations ciblées

Cette base solide ouvre la voie à des extensions futures ambitieuses tout en préservant la stabilité et la performance du système existant.
