# Spécifications Complètes - NCA Modulaire v9
## Architecture Découplée avec Stages Extensibles

*Version consolidée - 5 octobre 2025*

---

## Vue d'Ensemble du Système

Le NCA Modulaire v9 représente une **refonte architecturale majeure** introduisant une architecture découplée et extensible basée sur des stages modulaires autonomes. Cette version privilégie la séparation des responsabilités, l'extensibilité et la maintenabilité du code.

### Innovation Principale : Architecture Modulaire Découplée

- **Stages autonomes** : Chaque stage gère sa propre configuration, logique d'entraînement et validation
- **Système d'enregistrement** : Ajout facile de nouveaux stages sans modification du code existant
- **Interface standardisée** : Tous les stages implémentent l'interface `BaseStage`
- **Séparation visualisation/génération** : Architecture claire entre génération de données et visualisation

### Concepts Métier Principaux

#### Neural Cellular Automaton (NCA)
- **Paradigme** : Apprentissage local distribué avec règles cellulaires apprises
- **Cellules** : Chaque pixel de la grille 16x16 (256 cellules total)
- **Voisinage** : Patch 3x3 centré sur chaque cellule
- **Règle d'évolution** : Réseau neuronal prédit le delta de température
- **Contraintes physiques** : Respect des conditions aux limites (source, obstacles)

#### Diffusion de Chaleur
- **Source de chaleur** : Point unique d'intensité fixe ou variable (0.0-1.0)
- **Obstacles** : Zones imperméables (température fixée à 0)
- **Diffusion** : Propagation par convolution avec noyau moyenneur 3x3
- **Équilibre thermique** : État stable après convergence

---

## Spécifications Fonctionnelles

### 1. Architecture à 4 Stages Progressifs

#### Stage 1 : Apprentissage de Base (Sans Obstacles)
**Objectif** : Établir les bases de la diffusion thermique pure

- **Environnement** : Grille vide avec source centrale d'intensité fixe 1.0
- **Durée** : 30% du temps total (150 époques sur 500)
- **Convergence** : Seuil strict de 0.0002 pour une base solide
- **Learning Rate** : Taux standard (multiplicateur 1.0)
- **Obstacles** : Aucun (min=0, max=0)
- **Critères** : Convergence uniforme et stabilité temporelle

#### Stage 2 : Introduction d'Obstacles Simples
**Objectif** : Apprendre le contournement d'obstacles uniques

- **Environnement** : Un obstacle rectangulaire, source d'intensité 1.0
- **Durée** : 30% du temps total (150 époques sur 500)
- **Convergence** : Seuil strict de 0.0002
- **Learning Rate** : Réduit (multiplicateur 0.8)
- **Obstacles** : Un seul (min=1, max=1)
- **Critères** : Contournement efficace et conservation d'énergie

#### Stage 3 : Gestion d'Obstacles Complexes
**Objectif** : Maîtriser les configurations multi-obstacles

- **Environnement** : 2-4 obstacles variés, source d'intensité 1.0
- **Durée** : 20% du temps total (100 époques sur 500)
- **Convergence** : Seuil plus tolérant de 0.001
- **Learning Rate** : Fortement réduit (multiplicateur 0.6)
- **Obstacles** : Multiples (min=2, max=4)
- **Critères** : Gestion de complexité et robustesse

#### Stage 4 : Intensités Variables (Innovation Majeure)
**Objectif** : Adaptation aux intensités de source variables

- **Environnement** : 1-2 obstacles, **intensité variable entre simulations**
- **Durée** : 20% du temps total (100 époques sur 500)
- **Innovation** : 
  - Intensité **fixe pendant chaque simulation** (0.0 à 1.0)
  - Intensité **différente à chaque nouvelle simulation**
  - Curriculum progressif : [0.5,1.0] → [0.0,1.0]
- **Convergence** : Seuil adapté de 0.0015
- **Learning Rate** : Très réduit (multiplicateur 0.4)
- **Critères** : Adaptation universelle sur toute la plage d'intensités

### 2. Curriculum d'Apprentissage Progressif

#### Phases 1-3 : Complexité Spatiale Croissante
- **Progression** : Sans obstacles → Un obstacle → Obstacles multiples
- **Intensité** : Fixe à 1.0 pour toutes les phases
- **Focus** : Apprentissage de la complexité spatiale

#### Phase 4 : Curriculum d'Intensité Variable
```
Phase 1 (0-25% des époques)  : [0.5, 1.0] - Sources moyennes à fortes
Phase 2 (25-50% des époques) : [0.3, 1.0] - Ajout des sources faibles  
Phase 3 (50-75% des époques) : [0.1, 1.0] - Sources très faibles
Phase 4 (75-100% des époques): [0.0, 1.0] - Plage complète (source éteinte incluse)
```

### 3. Cas d'Utilisation Principaux

#### CU-1 : Entraînement Complet du Curriculum
**Acteur** : Chercheur/Développeur  
**Objectif** : Entraîner le NCA sur les 4 stages progressifs

**Scénario nominal :**
1. Lancement avec paramètres (seed, epochs, learning rate)
2. Exécution séquentielle des stages 1→2→3→4
3. Validation de convergence à chaque stage
4. Sauvegarde des checkpoints intermédiaires
5. Génération automatique des visualisations

**Critères de succès :**
- Convergence de tous les stages (loss < seuil)
- Temps d'exécution < 2h sur GPU moderne
- Génération complète des visualisations

#### CU-2 : Visualisation et Analyse
**Acteur** : Chercheur/Analyste  
**Objectif** : Analyser les performances par stage

**Scénario nominal :**
1. Génération d'animations GIF comparatives (cible vs NCA)
2. Création de graphiques de convergence temporelle
3. Visualisations spécialisées par stage (intensités variables pour Stage 4)
4. Métriques quantitatives (MSE, stabilité, temps de convergence)

#### CU-3 : Expérimentation et Extension
**Acteur** : Chercheur expérientiel  
**Objectif** : Tester des configurations personnalisées

**Scénario nominal :**
1. Modification des paramètres globaux
2. Ajout de nouveaux stages via le système extensible
3. Comparaison avec les baselines existantes
4. Validation des extensions

---

## Spécifications Techniques Détaillées

### 1. Architecture de l'Intelligence Artificielle

#### Modèle Neural Cellular Automaton (NCA)

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

**Paramètres du Modèle :**
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

#### Processus de Traitement et Compromis

**Pipeline de Traitement Complet :**
1. **Extraction de Contexte** : Voisinage 3x3 avec padding réplicatif
2. **Ajout de Contraintes** : Masques source/obstacle pour guidance
3. **Prédiction de Delta** : Application du réseau neuronal
4. **Mise à Jour avec Contraintes** : Respect des conditions physiques

**Compromis Architecturaux Critiques :**

- **Contexte 3x3 vs Efficacité** : Balance entre information locale et complexité
- **Profondeur 3 couches vs Overfitting** : Capacité d'apprentissage sans surapprentissage
- **Delta scale 0.1 vs Stabilité** : Évite les oscillations tout en permettant l'évolution

### 2. Architecture Logicielle Modulaire

#### Structure de Répertoires Réorganisée
```
v9/stages/
├── visualizers/                    # Classes partagées uniquement
│   ├── intensity_animator.py       # Animation intensités variables
│   ├── metrics_plotter.py          # Graphiques métriques
│   ├── progressive_visualizer.py   # Visualisations progressives
│   └── visualization_suite.py      # Suite complète
├── stage1/
│   ├── __init__.py                 # Export Stage1, Stage1Config, Stage1Visualizer
│   ├── train.py                    # Stage1 + Stage1Config
│   └── visualizer.py               # Stage1Visualizer
├── stage2/
│   ├── __init__.py                 
│   ├── train.py                    # Stage2 + Stage2Config
│   └── visualizer.py               # Stage2Visualizer
├── stage3/
│   ├── __init__.py                 
│   ├── train.py                    # Stage3 + Stage3Config
│   └── visualizer.py               # Stage3Visualizer
└── stage4/
    ├── __init__.py                 
    ├── train.py                    # Stage4 + Stage4Config + IntensityManager
    └── visualizer.py               # Stage4Visualizer
```

#### Interface BaseStage Standardisée
```python
class BaseStage(ABC):
    """Interface commune pour tous les stages d'entraînement"""
    
    @abstractmethod
    def generate_environment(self, size: int, source_pos: Tuple[int, int], 
                           seed: Optional[int] = None) -> torch.Tensor
    
    @abstractmethod  
    def validate_convergence(self, recent_losses: List[float], 
                           epoch_in_stage: int) -> bool
    
    @abstractmethod
    def get_loss_weights(self) -> Dict[str, float]
    
    @abstractmethod
    def prepare_training_data(self, global_config: Any) -> Dict[str, Any]
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

#### Gestionnaire de Stages Modulaire
```python
class ModularStageManager:
    """
    Orchestrateur principal de l'entraînement modulaire.
    Gère l'exécution séquentielle des stages de manière découplée.
    """
    
    def __init__(self, global_config, device: str):
        self.global_config = global_config
        self.device = device
        self.active_stages = {}
        self.stage_sequence = [1, 2, 3, 4]
    
    def execute_full_curriculum(self, train_callback) -> Dict[str, Any]
    def get_stage_metrics(self) -> Dict[str, Any]
```

### 3. Configuration des Stages

#### Tableau de Configuration Détaillé

| Stage | Objectif | Obstacles | Intensité | Seuil Convergence | Ratio Époques | LR Mult. |
|-------|----------|-----------|-----------|-------------------|---------------|----------|
| 1 | Diffusion pure | 0 | 1.0 fixe | 0.0002 | 30% | 1.0 |
| 2 | Obstacles simples | 1 | 1.0 fixe | 0.0002 | 30% | 0.8 |
| 3 | Obstacles complexes | 2-4 | 1.0 fixe | 0.001 | 20% | 0.6 |
| 4 | Intensités variables | 1-2 | 0.0-1.0 | 0.0015 | 20% | 0.4 |

#### Mécanismes de Convergence
- **Early Stopping** : Validation sur fenêtres de 10-20 époques selon le stage
- **Critères de stabilité** : Variance des pertes < seuils adaptatifs
- **Learning Rate Scheduling** : Décroissance cosine personnalisée par stage

### 4. Simulation Physique Modulaire

#### ModularDiffusionSimulator
- **Kernel de diffusion** : Moyenneur 3x3 normalisé (conv2d PyTorch)
- **Conditions aux limites** :
  - Source : Température maintenue à l'intensité spécifiée
  - Obstacles : Température forcée à 0
  - Bords : Padding réplicatif
- **Intégration temporelle** : Euler explicite, 20 pas par défaut
- **Support intensité variable** : Gestion dynamique pour Stage 4

#### Génération d'Environnements Intelligente
- **Placement aléatoire** : Source dans région centrale (évite les bords)
- **Obstacles rectangulaires** : Taille 2-4 pixels, positionnement optimisé
- **Validation automatique** : Vérification de connectivité et cohérence
- **Algorithmes de fallback** : Environnements simplifiés si génération échoue

### 5. Optimisations Techniques

#### Performance GPU
- **Vectorisation complète** : Traitement par batch de patches via unfold()
- **GPU natif** : Toutes les opérations sur device CUDA
- **Gradient clipping** : Norm max 1.0 pour stabilité numérique
- **Masquage intelligent** : Calcul seulement sur cellules valides

#### Gestion Mémoire
- **Cache adaptatif** : Réutilisation des séquences selon le stage
- **Batch size optimisé** : 4 par défaut, ajustable selon GPU
- **Libération explicite** : Nettoyage mémoire entre stages

### 6. Système de Visualisation Découplé

#### Types de Visualisations
1. **Animations GIF** : Comparaison temporelle cible vs NCA
2. **Convergence plots** : Évolution de l'erreur MSE par stage
3. **Métriques spécialisées** : Histoires de perte, learning rates
4. **Visualisations Stage 4** : Comparaisons multi-intensités, analyses de robustesse

#### Composants Spécialisés
- **ProgressiveVisualizer** : Visualiseur principal multi-stages
- **IntensityAwareAnimator** : Animations avec gestion d'intensités variables
- **VariableIntensityMetricsPlotter** : Graphiques spécialisés pour Stage 4

---

## Contraintes et Limitations

### Contraintes Techniques
- **Taille de grille fixe** : 16x16 (256 cellules)
- **Précision numérique** : Float32 PyTorch standard
- **Mémoire GPU minimale** : 2GB pour entraînement complet
- **Temps d'exécution** : ~1-2h sur GPU moderne (RTX 3070+)

### Limitations Identifiées
- **Scalabilité spatiale** : Architecture non adaptée aux grandes grilles
- **Physique simplifiée** : Diffusion idéalisée sans effets complexes
- **Généralisation limitée** : Spécialisé pour obstacles rectangulaires
- **Curriculum fixe** : Séquence de stages prédéfinie (extensible via plugins)

---

## Métriques et Validation

### Métriques Principales par Stage
- **MSE temporelle** : Erreur quadratique moyenne à chaque pas de temps
- **Convergence rate** : Vitesse d'atteinte du seuil spécifique au stage
- **Stabilité** : Variance des pertes sur fenêtre glissante adaptative
- **Métriques spécialisées** : Adaptation aux intensités (Stage 4), robustesse (Stage 3)

### Critères de Succès Globaux
- **Convergence complète** : Tous les stages atteignent leur seuil respectif
- **Qualité visuelle** : Animations fluides et physiquement cohérentes
- **Performance temporelle** : Entraînement < 2h sur hardware standard
- **Reproductibilité** : Résultats identiques avec même seed

### Tests de Validation
- **Tests d'intégration** : Pipeline complet des 4 stages
- **Tests de régression** : Performance maintenue vs versions précédentes
- **Tests d'extensibilité** : Ajout de nouveaux stages sans impact

---

## Configuration de Production Recommandée

### Paramètres Optimaux
```python
# Configuration globale validée
TOTAL_EPOCHS = 500
GRID_SIZE = 16
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
N_LAYERS = 3

# Répartition optimale par stage
STAGE_RATIOS = {1: 0.30, 2: 0.30, 3: 0.20, 4: 0.20}

# Seuils de convergence validés
CONVERGENCE_THRESHOLDS = {
    1: 0.0002,  # Très strict pour la base
    2: 0.0002,  # Strict pour la robustesse
    3: 0.001,   # Adapté à la complexité
    4: 0.0015   # Tolérant aux intensités variables
}
```

### Optimisations Matérielles
```python
# Configuration GPU recommandée
DEVICE = "cuda"
USE_OPTIMIZATIONS = True
USE_VECTORIZED_PATCHES = True
TORCH_BACKENDS_CUDNN_BENCHMARK = True
```

---

## Extensions Futures et Évolutivité

### Architecture Extensible
- **Système de plugins** : Chargement dynamique de nouveaux stages
- **API standardisée** : Interface stable pour développeurs tiers
- **Registre de stages** : Découverte automatique de nouvelles implémentations

### Améliorations Techniques Futures
1. **Architecture multi-échelle** : Convolutions à différentes résolutions
2. **Attention spatiale** : Mécanismes d'attention pour dépendances longues
3. **Mémoire externe** : Système de mémoire pour historique temporel
4. **Optimisation automatique** : Recherche d'hyperparamètres par RL

### Extensions Fonctionnelles
1. **Nouvelles physiques** : Électrostatique, fluides, chimie
2. **Obstacles dynamiques** : Environnements changeants temporellement
3. **Multi-sources** : Systèmes avec plusieurs sources simultanées
4. **Apprentissage interactif** : Interface utilisateur temps réel

### Scalabilité
1. **Grilles variables** : Support de tailles arbitraires
2. **Parallélisation multi-GPU** : Entraînement distribué
3. **Optimisations mémoire** : Checkpointing, mixed precision
4. **Déploiement** : Export ONNX, optimisation inference

---

## Applications Industrielles

### Domaines d'Application
- **Systèmes de chauffage** : Équipements de puissances variables (0% à 100%)
- **Contrôle thermique** : Adaptation aux conditions opérationnelles variées
- **Maintenance prédictive** : Fonctionnement avec équipements dégradés
- **Optimisation énergétique** : Gestion efficace des ressources

### Avantages Concurrentiels
- **Modularité** : Architecture extensible et maintenable
- **Robustesse** : Apprentissage progressif pour stabilité maximale
- **Innovation** : Support des intensités variables (unique dans le domaine)
- **Performance** : Optimisations GPU pour déploiement industriel

---

## Conclusion

Le NCA Modulaire v9 représente une **évolution architecturale majeure** privilégiant l'extensibilité, la maintenabilité et la performance. Cette architecture modulaire découplée permet :

1. **Extensibilité maximale** : Ajout trivial de nouveaux stages sans impact
2. **Maintenance simplifiée** : Découplage complet des responsabilités
3. **Performance optimisée** : Optimisations GPU ciblées et vectorisation avancée
4. **Innovation technique** : Support pionnier des intensités variables
5. **Validation rigoureuse** : Tests complets d'intégration et de régression

Cette base solide et évolutive ouvre la voie à des extensions futures ambitieuses tout en préservant la stabilité et la performance du système existant.

---

*Spécifications consolidées - NCA Modulaire v9*  
*Architecture Découplée avec Stages Extensibles*  
*Version finale - 5 octobre 2025*
