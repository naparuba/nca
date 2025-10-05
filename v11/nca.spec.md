# Spécifications Complètes - NCA Modulaire v11
## Architecture Découplée avec Stages Extensibles et Atténuation Temporelle

*Version mise à jour - 5 octobre 2025*

---

## Vue d'Ensemble du Système

Le NCA Modulaire v11 représente une **évolution de l'architecture modulaire v10** basée sur des stages autonomes. Cette version conserve les principes fondamentaux de séparation des responsabilités, d'extensibilité et de maintenabilité du code.

### Innovation Principale : Architecture Modulaire Découplée

- **Stages autonomes** : Chaque stage gère sa propre configuration, logique d'entraînement et validation
- **Système d'enregistrement** : Ajout facile de nouveaux stages sans modification du code existant
- **Interface standardisée** : Tous les stages implémentent l'interface `BaseStage`
- **Séparation visualisation/génération** : Architecture claire entre génération de données et visualisation

### Fonctionnalité Principale : Atténuation Temporelle des Sources

Cette version v11 maintient la fonctionnalité clé introduite en v10 : **l'atténuation temporelle des sources**. Le système apprend à gérer des sources dont l'intensité diminue progressivement au cours du temps, simulant des phénomènes physiques comme le refroidissement ou l'épuisement d'une ressource.

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

### 1. Architecture à 5 Stages Progressifs

#### Stage 1 : Apprentissage de Base (Sans Obstacles)
**Objectif** : Établir les bases de la diffusion thermique pure

- **Environnement** : Grille vide avec source centrale d'intensité fixe 1.0
- **Durée** : 20% du temps total (200 époques sur 1000)
- **Convergence** : Seuil strict de 0.0002 pour une base solide
- **Learning Rate** : Taux standard (multiplicateur 1.0)
- **Obstacles** : Aucun (min=0, max=0)
- **Critères** : Convergence uniforme et stabilité temporelle

#### Stage 2 : Introduction d'Obstacles Simples
**Objectif** : Apprendre le contournement d'obstacles uniques

- **Environnement** : Un obstacle rectangulaire, source d'intensité 1.0
- **Durée** : 20% du temps total (200 époques sur 1000)
- **Convergence** : Seuil strict de 0.0002
- **Learning Rate** : Réduit (multiplicateur 0.8)
- **Obstacles** : Un seul (min=1, max=1)
- **Critères** : Contournement efficace et conservation d'énergie

#### Stage 3 : Gestion d'Obstacles Complexes
**Objectif** : Maîtriser les configurations multi-obstacles

- **Environnement** : 2-4 obstacles variés, source d'intensité 1.0
- **Durée** : 20% du temps total (200 époques sur 1000)
- **Convergence** : Seuil plus tolérant de 0.001
- **Learning Rate** : Fortement réduit (multiplicateur 0.6)
- **Obstacles** : Multiples (min=2, max=4)
- **Critères** : Gestion de complexité et robustesse

#### Stage 4 : Intensités Variables (Innovation Majeure)
**Objectif** : Adaptation aux intensités de source variables

- **Environnement** : 1-2 obstacles, **intensité variable entre simulations**
- **Durée** : 20% du temps total (200 époques sur 1000)
- **Innovation** : 
  - Intensité **fixe pendant chaque simulation** (0.0 à 1.0)
  - Intensité **différente à chaque nouvelle simulation**
  - Curriculum progressif : [0.5,1.0] → [0.0,1.0]
- **Convergence** : Seuil adapté de 0.0015
- **Learning Rate** : Très réduit (multiplicateur 0.4)
- **Critères** : Adaptation universelle sur toute la plage d'intensités

#### Stage 5 : Atténuation Temporelle des Sources
**Objectif** : Maîtriser la diffusion avec sources d'intensité décroissante dans le temps

- **Environnement** : 1-2 obstacles, **intensité décroissante pendant la simulation**
- **Durée** : 20% du temps total (200 époques sur 1000)
- **Innovation** : 
  - Source commençant à une intensité variable (0.3 à 1.0)
  - **Atténuation linéaire dans le temps** au cours de chaque simulation
  - Taux de décroissance variable entre simulations (0.002 à 0.015 par pas)
  - Curriculum progressif d'intensités et taux d'atténuation
- **Convergence** : Seuil adapté de 0.00001
- **Learning Rate** : Très réduit (multiplicateur 0.3)
- **Critères** : Adaptation au refroidissement progressif, stabilité avec source faiblissante
- **Pertes spécialisées** : MSE standard + perte sur cellules sources + perte de cohérence temporelle


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

#### Phase 5 : Curriculum d'Atténuation Temporelle
```
Phase 1 (0-25% des époques)  : Intensités [0.5, 1.0], Atténuation minimale (0.002)
Phase 2 (25-50% des époques) : Intensités [0.4, 1.0], Atténuation progressive
Phase 3 (50-75% des époques) : Intensités [0.3, 1.0], Atténuation modérée
Phase 4 (75-100% des époques): Intensités [0.3, 1.0], Atténuation complète (jusqu'à 0.015)
```

### 3. Cas d'Utilisation Principaux

#### CU-1 : Entraînement Complet du Curriculum
**Acteur** : Chercheur/Développeur  
**Objectif** : Entraîner le NCA sur les 4 stages progressifs

**Scénario nominal :**
1. Lancement avec paramètres (seed, epochs, learning rate)
2. Exécution séquentielle des stages 1→2→3→4→5
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

#### Structure de Répertoires
```
v11/
├── nca_time_atenuation_v11.py     # Fichier principal avec architecture v11
├── stages/                        # Architecture modulaire avancée
│   ├── visualizers/               # Classes partagées uniquement
│   │   ├── intensity_animator.py  # Animation intensités variables
│   │   ├── metrics_plotter.py     # Graphiques métriques
│   │   └── ...
│   ├── stage1/
│   │   ├── __init__.py
│   │   ├── train.py               # Stage1 + Stage1Config
│   │   └── visualizer.py          # Stage1Visualizer
│   ├── ...
│   └── stage5/
│       ├── __init__.py
│       ├── train.py               # Stage5 + Stage5Config + TemporalAttenuationManager
│       └── visualizer.py          # Stage5Visualizer
├── nca.spec.md                    # Documentation complète
└── nca_outputs_modular_progressive_obstacles_variable_intensity_seed_123/
    └── ...                        # Résultats d'entraînement et visualisations
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

#### TemporalAttenuationManager pour Stage 5
```python
class TemporalAttenuationManager:
    """
    Gestionnaire spécialisé pour l'atténuation temporelle des sources du Stage 5.
    Gère la diminution progressive de l'intensité de la source pendant la simulation.
    """
    
    def sample_initial_intensity(self, epoch_progress: float) -> float:
        """Échantillonne une intensité initiale selon progression d'entraînement"""
    
    def sample_attenuation_rate(self, epoch_progress: float) -> float:
        """Échantillonne un taux d'atténuation selon progression d'entraînement"""
    
    def generate_temporal_sequence(self, initial_intensity: float, attenuation_rate: float,
                                  n_steps: int) -> List[float]:
        """Génère une séquence d'intensités décroissantes dans le temps"""
    
    def get_source_intensity_at_step(self, sequence_id: int, step: int) -> float:
        """Récupère l'intensité de la source à un pas de temps spécifique"""
```

#### Configuration du Stage 5
```python
class Stage5Config(StageConfig):
    """Configuration spécialisée pour le Stage 5."""
    
    def __init__(self):
        super().__init__(
            stage_id=5,
            name="Atténuation Temporelle des Sources",
            description="Apprentissage avec sources d'intensité décroissante dans le temps",
            epochs_ratio=0.2,
            convergence_threshold=0.00001,  # Seuil adapté à la complexité
            learning_rate_multiplier=0.3,   # LR très réduit pour ce stage
            min_obstacles=1,
            max_obstacles=2
        )
        
        # Configuration spéciale pour l'atténuation temporelle
        self.min_source_intensity = 0.3
        self.max_source_intensity = 1.0
        self.initial_intensity_range = [0.5, 1.0]  # Plage initiale restreinte
        self.final_intensity_range = [0.3, 1.0]    # Plage finale élargie
        
        # Configuration de l'atténuation temporelle
        self.min_attenuation_rate = 0.002  # Atténuation minimale par pas de temps
        self.max_attenuation_rate = 0.015  # Atténuation maximale par pas de temps
```

#### ModularNCAUpdater avec Support Temporel
```python
class ModularNCAUpdater:
    """Updater NCA adapté à l'architecture modulaire avec support temporel."""

    def __init__(self, model: ImprovedNCA, device: str, use_temporal_feature: bool = False):
        # Support optionnel d'une feature temporelle (pour expérimentation)
        self.use_temporal_feature = use_temporal_feature
        
    def step(self, grid: torch.Tensor, source_mask: torch.Tensor,
             obstacle_mask: torch.Tensor, source_intensity: Optional[float] = None,
             time_step: Optional[int] = None, max_steps: Optional[int] = None) -> torch.Tensor:
        """
        Application du NCA avec support intensité variable et information temporelle.
        """
```

#### ModularDiffusionSimulator avec Support d'Atténuation
```python
class ModularDiffusionSimulator:
    def generate_sequence_with_stage(self, stage: BaseStage, n_steps: int, size: int,
                                   source_intensity: Optional[float] = None,
                                   seed: Optional[int] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, float]:
        # Permettre au stage d'initialiser la séquence (pour atténuation, etc.)
        stage.initialize_sequence(n_steps, 0.5)  # 0.5 = milieu de progression

        # Simulation temporelle avec intensité variable dans le temps
        for step in range(n_steps):
            # Obtenir l'intensité de la source pour ce pas de temps
            current_intensity = stage.get_source_intensity_at_step(step, used_intensity)
            
            # Application du pas de simulation avec l'intensité du moment
            grid = self.step(grid, source_mask, obstacle_mask, current_intensity)
```

### 3. Configuration des Stages

#### Tableau de Configuration Détaillé

| Stage | Objectif | Obstacles | Intensité | Seuil Convergence | Ratio Époques | LR Mult. |
|-------|----------|-----------|-----------|-------------------|---------------|----------|
| 1 | Diffusion pure | 0 | 1.0 fixe | 0.0002 | 20% | 1.0 |
| 2 | Obstacles simples | 1 | 1.0 fixe | 0.0002 | 20% | 0.8 |
| 3 | Obstacles complexes | 2-4 | 1.0 fixe | 0.001 | 20% | 0.6 |
| 4 | Intensités variables | 1-2 | 0.0-1.0 | 0.0015 | 20% | 0.4 |
| 5 | Atténuation temporelle | 1-2 | 0.3-1.0 décroissant | 0.00001 | 20% | 0.3 |

#### Fonction de Perte Spécialisée pour Stage 5
- **MSE standard** : Sur toute la grille (poids standard)
- **Perte sur cellules sources** : Focus sur l'apprentissage de l'atténuation (poids élevé)
- **Perte de cohérence temporelle** : Assure une évolution temporelle cohérente (poids moyen)

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
- **Métriques spécialisées** :
  - Stage 4: Adaptation aux intensités variables
  - Stage 5: Précision de l'atténuation temporelle et stabilité du refroidissement

### Critères de Succès Globaux
- **Convergence complète** : Tous les stages atteignent leur seuil respectif
- **Qualité visuelle** : Animations fluides et physiquement cohérentes
- **Performance temporelle** : Entraînement < 2h sur hardware standard
- **Reproductibilité** : Résultats identiques avec même seed

### Tests de Validation
- **Tests d'intégration** : Pipeline complet des 5 stages
- **Tests de régression** : Performance maintenue vs versions précédentes
- **Tests d'extensibilité** : Ajout de nouveaux stages sans impact
