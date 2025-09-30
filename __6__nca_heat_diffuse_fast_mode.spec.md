# =============================================================================
# SPÉCIFICATION TECHNIQUE : Système NCA Diffusion de Chaleur avec Obstacles
# =============================================================================
# Fichier de spécification pour accélérer l'analyse IA du système 6__*py
# Généré automatiquement par analyse en profondeur du code source
# Date: 2024-12-29

## ARCHITECTURE GÉNÉRALE DU SYSTÈME

### Vue d'ensemble
Le système implémente un Neural Cellular Automaton (NCA) optimisé pour apprendre la diffusion de chaleur en présence d'obstacles. Il s'agit d'une version avancée avec optimisations de performance et support complet des obstacles.

### Fichiers principaux
- `6__nca_heat_diffuse_fast_mode.py` : Script d'entraînement principal
- `6__visualize_heat_diffuse_fast_mode.py` : Script de visualisation avancée

### Paradigme architectural
- **Séparation des responsabilités** : Simulateur physique / Modèle neuronal / Entraîneur / Visualiseur
- **Configuration centralisée** : Classe `Config` pour tous les hyperparamètres
- **Mode dual** : Interactif (temps réel) vs Non-interactif (sauvegarde)
- **Support multi-seed** : Seeds séparées pour entraînement et visualisation

## COMPOSANTS PRINCIPAUX

### 1. Configuration (Classe Config)
**Rôle** : Centralise tous les paramètres du système pour faciliter les expérimentations

**Paramètres clés** :
```python
DEVICE: str                    # "cuda" ou "cpu" auto-détecté
SEED: int                      # Seed principal (défaut: 123)
GRID_SIZE: int                 # Taille grille NxN (défaut: 16)
SOURCE_INTENSITY: float        # Intensité sources chaleur (défaut: 1.0)

# Entraînement
N_EPOCHS: int                  # Nombre époques (défaut: 100)
NCA_STEPS: int                 # Horizon temporel multi-step (défaut: 20)
LEARNING_RATE: float           # Taux apprentissage (défaut: 1e-3)
BATCH_SIZE: int                # Taille batch (défaut: 4)

# Architecture modèle
HIDDEN_SIZE: int               # Taille couches cachées (défaut: 128)
N_LAYERS: int                  # Nombre couches (défaut: 3)

# Obstacles
MIN_OBSTACLES: int             # Nombre min obstacles (défaut: 1)
MAX_OBSTACLES: int             # Nombre max obstacles (défaut: 3)
MIN_OBSTACLE_SIZE: int         # Taille min obstacle (défaut: 2)
MAX_OBSTACLE_SIZE: int         # Taille max obstacle (défaut: 4)

# Optimisations NOUVELLES
USE_OPTIMIZATIONS: bool        # Active optimisations (défaut: True)
USE_SEQUENCE_CACHE: bool       # Cache séquences (défaut: True)
USE_VECTORIZED_PATCHES: bool   # Extraction vectorisée (défaut: True)
CACHE_SIZE: int                # Taille cache (défaut: 200)
```

**Arguments ligne de commande supportés** :
- `--seed` : Seed d'entraînement
- `--vis-seed` : Seed de visualisation (indépendante)
- `--epochs` : Nombre d'époques
- `--grid-size` : Taille de la grille
- `--batch-size` : Taille des batches

### 2. Simulateur Physique (Classe DiffusionSimulator)
**Rôle** : Génère la vérité terrain (simulation physique) que le NCA doit apprendre

**Algorithme de diffusion** :
- Kernel de convolution 3x3 uniforme (moyenne des 9 voisins)
- Simule l'équation de diffusion de chaleur discrétisée
- Conditions de Dirichlet pour les sources (température fixe)
- Obstacles bloquent complètement la diffusion

**Génération d'obstacles** :
- Placement aléatoire avec évitement de la source
- Tailles variables (carrés de 2x2 à 4x4)
- Nombre variable (1 à 3 obstacles par grille)
- 50 tentatives max pour placement valide

**Méthodes clés** :
```python
generate_obstacles(size, source_pos, seed) -> torch.Tensor
step(grid, source_mask, obstacle_mask) -> torch.Tensor
generate_sequence(n_steps, size, seed) -> (List[Tensor], Tensor, Tensor)
```

### 3. Modèle NCA (Classe ImprovedNCA)
**Rôle** : Réseau neuronal qui apprend les règles de mise à jour locales

**Architecture avancée** :
- Input: Patch 3x3 (9 valeurs) + info source (1) + info obstacle (1) = **11 features**
- Couches cachées : `n_layers` × (`Linear` → `BatchNorm1d` → `ReLU` → `Dropout(0.1)`)
- Output: Delta dans [-0.1, 0.1] via `Tanh` + scaling
- Stabilité renforcée par normalisation et dropout

**Améliorations vs version classique** :
- Support natif des obstacles dans l'input
- BatchNorm pour stabilité d'entraînement
- Dropout pour régularisation
- Architecture flexible (paramétrable)
- Delta limité pour éviter explosions

### 4. Systèmes de Mise à Jour (NCAUpdater vs OptimizedNCAUpdater)

#### NCAUpdater (Version standard)
- Extraction patches par boucles Python
- Traitement séquentiel des positions
- Simple mais lent

#### OptimizedNCAUpdater (Version optimisée - NOUVEAU)
- **Extraction vectorisée** via `F.unfold()`
- Traitement parallèle sur GPU
- Performance drastiquement améliorée
- Zero-copy operations quand possible

**Différence de performance** : ~10-50x plus rapide selon la taille de grille

### 5. Système d'Entraînement Multi-Modal

#### Cache de Séquences Optimisé (OptimizedSequenceCache - NOUVEAU)
**Innovation majeure** : Pré-génère et stocke toutes les séquences d'entraînement

**Avantages** :
- Élimination du overhead de génération pendant l'entraînement
- Données stockées directement sur GPU
- Mélange périodique pour variété
- Accès cyclique O(1)

#### NCATrainer (Version hybride)
**Modes de fonctionnement** :
- **Standard** : Génération à la volée (comme versions précédentes)
- **Optimisé** : Utilise cache + updater vectorisé

**Fonctionnalités avancées** :
- Gradient clipping automatique (max_norm=1.0)
- Scheduler cosine pour learning rate
- Weight decay (1e-4) pour régularisation
- Perte multi-step (moyenne sur horizon temporel)
- Historique détaillé (loss + learning rate)

**Algorithme d'entraînement** :
```
Pour chaque époque:
  Pour chaque batch:
    - Initialise grille vide + sources
    - Pour NCA_STEPS étapes:
      * Applique NCA une fois
      * Calcule loss MSE vs target
      * Accumule loss
    - Backprop sur loss moyenne
    - Clip gradients
    - Update paramètres
```

### 6. Système de Visualisation Avancé (NCAVisualizer)

**Modes de fonctionnement** :
- **Interactif** : Animation temps réel (Qt5Agg/TkAgg)
- **Sauvegarde** : Génération fichiers pour post-traitement

**Données sauvegardées par frame** :
```python
{
    'step': int,
    'nca_grid': np.ndarray,      # État NCA
    'target_grid': np.ndarray,   # État cible
    'obstacle_mask': np.ndarray, # Masque obstacles
    'source_mask': np.ndarray,   # Masque sources
    'title': str
}
```

**Métriques générées** :
- Évolution perte (échelle log)
- Évolution learning rate (échelle log)
- Sauvegarde modèle avec métadonnées complètes

## SCRIPT DE VISUALISATION AVANCÉE

### Fonctionnalités du visualiseur (6__visualize_heat_diffuse_fast_mode.py)

**Support multi-seed** :
- Détection automatique répertoires par pattern
- Support ancien format + nouveau format avec seed
- Arguments : `--seed`, `--output-dir`, `--fps`

**Analyses générées** :
1. **GIF animé 4 panneaux** :
   - NCA en temps réel
   - Simulation cible (statique état final)
   - Différence absolue |NCA - Cible|
   - Obstacles & sources (composite RGB)

2. **Comparaison statique multi-étapes** :
   - Étapes clés : [0, 10, 20, final]
   - Métriques par étape (MSE, erreur max)
   - Support obstacles

3. **Analyse de convergence** :
   - MSE au cours du temps (échelle log)
   - Erreur maximale au cours du temps
   - Couverture d'obstacles (si applicable)

**Gestion des obstacles dans la visualisation** :
- Détection automatique présence obstacles
- Codage couleur : Gris (obstacles), Rouge (sources), Blanc cassé (libre)
- Métriques spécialisées (nombre obstacles, couverture)

## OPTIMISATIONS DE PERFORMANCE

### 1. Cache de Séquences Pré-calculées
- **Impact** : Élimine 80%+ du temps de génération
- **Mémoire** : ~200 séquences stockées sur GPU
- **Stratégie** : Génération une fois, réutilisation cyclique

### 2. Extraction Vectorisée des Patches
- **Technique** : `F.unfold()` remplace boucles Python
- **Impact** : 10-50x accélération selon taille grille
- **GPU-natif** : Opérations entièrement parallélisées

### 3. Détection Automatique Backend Matplotlib
- **Fallback intelligent** : Qt5Agg → TkAgg → Agg (sauvegarde)
- **Compatibilité** : Windows, Linux, environnements headless

### 4. Architecture Modulaire
- **Updaters interchangeables** : Standard vs Optimisé
- **Configuration centralisée** : Expérimentations faciles
- **Séparation concerns** : Physique / Neural / Viz

## FORMATS DE DONNÉES

### Structure répertoires de sortie
```
__6__nca_outputs_heat_diffuse_fast_mode_seed_{SEED}/
├── animation_avant_entraînement.npy
├── animation_après_entraînement.npy
├── gif_animation_avant_entraînement.gif
├── gif_animation_après_entraînement.gif
├── static_animation_avant_entraînement.png
├── static_animation_après_entraînement.png
├── convergence_avant_entraînement_seed_{SEED}.png
├── convergence_après_entraînement_seed_{SEED}.png
├── training_metrics.png
└── nca_model.pth
```

### Format sauvegarde modèle
```python
{
    'model_state_dict': Dict,        # Poids réseau
    'config': Config,                # Configuration complète
    'training_seed': int,            # Seed entraînement
    'visualization_seed': int,       # Seed visualisation
    'loss_history': List[float],     # Historique perte
    'lr_history': List[float]        # Historique learning rate
}
```

## POINTS TECHNIQUES AVANCÉS

### Gestion de la Stabilité
- **Gradient clipping** : Limite à norm=1.0
- **Delta scaling** : Facteur 0.1 pour changements graduels
- **Clamping** : Valeurs dans [0,1] après chaque update

### Support Multi-GPU
- **Auto-détection** : `torch.cuda.is_available()`
- **Tensors sur device** : Transfert automatique
- **Générateurs avec seed** : Reproductibilité GPU

### Gestion Mémoire
- **Clonage explicite** : Évite modifications accidentelles
- **Cache GPU-resident** : Évite transfers CPU↔GPU
- **Cleanup automatique** : `plt.close()` systématique

## POINTS D'EXTENSION

### Facilités d'expérimentation
1. **Nouveaux obstacles** : Modifier `generate_obstacles()`
2. **Nouvelles architectures** : Hériter de `ImprovedNCA`
3. **Nouvelles métriques** : Étendre `NCAVisualizer`
4. **Nouveaux optimiseurs** : Modifier `NCATrainer.__init__()`

### Hyperparamètres critiques
- `NCA_STEPS` : Balance apprentissage vs stabilité
- `CACHE_SIZE` : Balance mémoire vs variété
- `DELTA_SCALE` : Contrôle vitesse convergence
- `HIDDEN_SIZE` : Capacité modèle vs overfitting

## DEBUGGING & DIAGNOSTICS

### Signaux de santé système
- **Mode optimisé activé** : Messages "🚀 Utilisation updater optimisé"
- **Cache initialisé** : Messages "✅ Cache des séquences créé"
- **Backend détecté** : Messages détection matplotlib

### Métriques de performance
- **Temps par époque** : Devrait diminuer avec optimisations
- **Convergence loss** : Doit décroître régulièrement
- **Stabilité gradients** : Vérifier via gradient clipping

### Points de défaillance communs
1. **Mémoire GPU insuffisante** : Réduire `CACHE_SIZE` ou `GRID_SIZE`
2. **Obstacles trop grands** : Vérifier `MAX_OBSTACLE_SIZE < GRID_SIZE//2`
3. **Backend matplotlib** : Problèmes affichage selon environnement

## COMPARAISON AVEC VERSIONS PRÉCÉDENTES

### Nouvelles fonctionnalités v6
- ✅ **Support obstacles complet** (génération + visualisation)
- ✅ **Cache séquences optimisé** (gain performance majeur)
- ✅ **Extraction patches vectorisée** (GPU-natif)
- ✅ **Seeds séparées** (entraînement vs visualisation)
- ✅ **Architecture modulaire** (updaters interchangeables)
- ✅ **Visualisation 4 panneaux** (NCA + Cible + Diff + Obstacles)
- ✅ **Sauvegarde métadonnées** (reproductibilité complète)

### Rétrocompatibilité
- ✅ Interface arguments ligne de commande préservée
- ✅ Format sorties compatible (ajouts non-breaking)
- ✅ Fallback mode standard si optimisations désactivées

---
**Note** : Cette spécification couvre l'état du système au 2024-12-29.
Pour modifications futures, mettre à jour cette spec en conséquence.

# Spécification : Neural Cellular Automaton - Diffusion de Chaleur avec Obstacles (Mode Optimisé)

## Vue d'ensemble

Ce projet implémente un **Neural Cellular Automaton (NCA)** optimisé pour apprendre la diffusion de chaleur en présence d'obstacles. Le système combine simulation physique, apprentissage automatique et optimisations de performance.

### Objectif
Entraîner un NCA à reproduire fidèlement le comportement de diffusion thermique d'un simulateur physique, tout en gérant des obstacles qui bloquent la propagation de la chaleur.

## Architecture du Système

### 1. Configuration Centralisée (`Config`)

**Paramètres de base :**
- `DEVICE` : GPU/CPU automatiquement détecté
- `SEED` : Graine aléatoire configurable (défaut: 123)
- `GRID_SIZE` : Taille de la grille (défaut: 16x16)
- `SOURCE_INTENSITY` : Intensité de la source de chaleur (1.0)

**Paramètres d'entraînement :**
- `N_EPOCHS` : 100 époques
- `NCA_STEPS` : 20 étapes temporelles par séquence
- `LEARNING_RATE` : 1e-3 avec AdamW
- `BATCH_SIZE` : 4 séquences par batch

**Paramètres de visualisation :**
- `PREVIS_STEPS` : 30 étapes (animation pré-entraînement)
- `POSTVIS_STEPS` : 50 étapes (animation post-entraînement)
- Détection automatique du mode interactif vs sauvegarde

**Paramètres des obstacles :**
- `MIN_OBSTACLES` / `MAX_OBSTACLES` : 1-3 obstacles par grille
- `MIN_OBSTACLE_SIZE` / `MAX_OBSTACLE_SIZE` : Carrés de 2x2 à 4x4

**Optimisations de performance :**
- `USE_OPTIMIZATIONS` : Active les optimisations (défaut: True)
- `USE_SEQUENCE_CACHE` : Cache de 200 séquences pré-calculées
- `USE_VECTORIZED_PATCHES` : Extraction vectorisée des patches
- `USE_MIXED_PRECISION` : Précision mixte (désactivé par défaut)

### 2. Simulateur Physique (`DiffusionSimulator`)

**Méthode de simulation :**
- Convolution 2D avec noyau 3x3 uniforme (1/9 pour chaque cellule)
- Simule l'équation de diffusion de chaleur discrétisée
- Kernel : `torch.ones((1, 1, 3, 3)) / 9.0`

**Gestion des obstacles :**
- Génération aléatoire d'obstacles rectangulaires
- Vérification de non-chevauchement avec les sources
- Obstacles bloquent complètement la diffusion (valeur fixée à 0.0)

**Conditions aux limites :**
- Sources de chaleur : conditions de Dirichlet (valeur fixe)
- Obstacles : conditions de Dirichlet (température = 0)
- Bords : padding replicatif lors de la convolution

### 3. Architecture du NCA (`ImprovedNCA`)

**Entrée du réseau :**
- Patch 3x3 autour de chaque cellule (9 valeurs)
- Feature "est source" (1 valeur booléenne)
- Feature "est obstacle" (1 valeur booléenne)
- **Total : 11 features d'entrée**

**Architecture du réseau :**
- Réseau fully-connected avec 3 couches cachées
- 128 neurones par couche cachée
- Activation ReLU + BatchNorm1d + Dropout(0.1)
- Sortie : 1 valeur (delta à appliquer)
- Facteur d'échelle : 0.1 (stabilité d'entraînement)

**Contraintes de sortie :**
- Delta dans [-0.1, 0.1] via Tanh + scaling
- Clamp final des valeurs dans [0.0, 1.0]

### 4. Système de Mise à Jour

**Version Standard (`NCAUpdater`) :**
- Boucles Python pour extraction des patches
- Traitement batch des patches valides
- Skip automatique des obstacles

**Version Optimisée (`OptimizedNCAUpdater`) :**
- Extraction vectorisée via `F.unfold()`
- Opérations GPU natives (pas de boucles Python)
- Gain de performance significatif

### 5. Système d'Entraînement (`NCATrainer`)

**Optimiseur :**
- AdamW avec weight decay (1e-4)
- CosineAnnealingLR scheduler
- Gradient clipping (max_norm=1.0)

**Fonction de perte :**
- MSE entre séquence prédite et séquence cible
- Accumulation sur les 20 étapes temporelles
- Moyennage pour la perte finale

**Cache de séquences optimisé (`OptimizedSequenceCache`) :**
- Pré-génération de 200 séquences d'entraînement
- Stockage direct sur GPU
- Mélange périodique pour la variété
- Accès cyclique aux séquences

### 6. Système de Visualisation (`NCAVisualizer`)

**Mode interactif :**
- Détection automatique des backends (Qt5Agg/TkAgg)
- Animation temps réel avec matplotlib
- Comparaison NCA vs simulation cible + obstacles

**Mode sauvegarde :**
- Export des animations en fichiers .npy
- Génération de GIFs via script séparé
- Métriques d'entraînement sauvegardées

## Arguments de Ligne de Commande

**Script principal (`__6__nca_heat_diffuse_fast_mode.py`) :**
```bash
--seed INT          # Graine d'entraînement (défaut: 123)
--vis-seed INT      # Graine de visualisation (défaut: 3333)  
--epochs INT        # Nombre d'époques (défaut: 100)
--grid-size INT     # Taille de grille (défaut: 16)
--batch-size INT    # Taille de batch (défaut: 4)
```

**Script de visualisation (`6__visualize_heat_diffuse_fast_mode.py`) :**
```bash
--seed INT          # Seed spécifique à visualiser
--output-dir STR    # Répertoire de sortie spécifique
--fps INT           # Images par seconde des GIFs (défaut: 8)
```

## Fichiers de Sortie

**Structure du répertoire de sortie :**
```
__6__nca_outputs_heat_diffuse_fast_mode_seed_{SEED}/
├── animation_avant_entraînement.npy      # Animation pré-entraînement
├── animation_après_entraînement.npy      # Animation post-entraînement
├── gif_animation_avant_entraînement.gif  # GIF pré-entraînement
├── gif_animation_après_entraînement.gif  # GIF post-entraînement
├── static_animation_avant_entraînement.png # Image statique pré
├── static_animation_après_entraînement.png # Image statique post
├── convergence_avant_entraînement_seed_{SEED}.png  # Graphique convergence pré
├── convergence_après_entraînement_seed_{SEED}.png  # Graphique convergence post
├── training_metrics.png                  # Métriques d'entraînement
└── nca_model.pth                        # Modèle sauvegardé
```

## Fonctionnalités Avancées

### Optimisations de Performance
1. **Cache de séquences** : Évite la recomputation des séquences d'entraînement
2. **Extraction vectorisée** : Remplace les boucles Python par des opérations GPU
3. **Batch processing** : Traitement groupé des patches pour l'efficacité

### Reproductibilité
- Seeds séparés pour entraînement et visualisation
- Configuration centralisée des paramètres
- Nommage des répertoires avec seed

### Robustesse
- Détection automatique du mode d'affichage
- Gestion d'erreurs pour les backends matplotlib
- Gradient clipping et régularisation
- Normalisation par batch

## Algorithme d'Entraînement

1. **Initialisation** : Grille vide avec sources activées
2. **Déroulement temporel** : 20 étapes NCA avec accumulation de perte
3. **Optimisation** : Backpropagation + gradient clipping
4. **Mise à jour** : Paramètres du réseau via AdamW
5. **Scheduling** : Réduction progressive du learning rate

## Contraintes Physiques

- **Sources** : Valeur fixe (1.0) maintenue à chaque étape
- **Obstacles** : Valeur fixe (0.0) - bloquent la diffusion
- **Bords** : Gestion automatique via padding replicatif
- **Valeurs** : Clamp dans [0.0, 1.0] pour la stabilité

## Métriques et Validation

- **Perte MSE** : Comparaison pixel par pixel avec simulation cible
- **Convergence visuelle** : Animations avant/après entraînement
- **Stabilité** : Gradient clipping et monitoring du learning rate
- **Performance** : Temps d'entraînement avec/sans optimisations

