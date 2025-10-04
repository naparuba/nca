 # Spécification: NCA Modulaire avec Obstacles Progressifs et Intensités Variables (Version 8__)

## Vue d'ensemble

Cette version étend le système modulaire v7__ en ajoutant une **quatrième étape** où le NCA apprend à gérer des **intensités de source variables**. L'innovation principale est l'apprentissage avec des intensités différentes entre les simulations, tout en maintenant une intensité fixe pendant chaque simulation individuelle.

## Architecture Modulaire Étendue (4 Étapes)

### Étape 1: Apprentissage de Base (Sans Obstacles)
- **Objectif**: Diffusion de chaleur pure sans obstacles
- **Durée**: 30% du temps d'entraînement total (150 époques sur 500)
- **Environnement**: Grille vide avec source centrale d'intensité 1.0
- **Métriques**: Convergence stricte (seuil: 0.0002)
- **Curriculum**: Learning rate standard, pas d'obstacles

### Étape 2: Introduction d'un Obstacle Unique  
- **Objectif**: Adaptation au contournement d'obstacles
- **Durée**: 30% du temps d'entraînement total (150 époques sur 500)
- **Environnement**: Un obstacle rectangulaire, source d'intensité 1.0
- **Métriques**: Convergence stricte (seuil: 0.0002)
- **Curriculum**: Learning rate réduit (×0.8), un seul obstacle

### Étape 3: Obstacles Multiples
- **Objectif**: Gestion de configurations complexes
- **Durée**: 20% du temps d'entraînement total (100 époques sur 500)  
- **Environnement**: 2-4 obstacles variés, source d'intensité 1.0
- **Métriques**: Convergence tolérante (seuil: 0.015)
- **Curriculum**: Learning rate réduit (×0.6), validation de connectivité

### Étape 4: Intensités Variables Entre Simulations (**NOUVEAU**)
- **Objectif**: Apprentissage avec intensités de source variables
- **Durée**: 20% du temps d'entraînement total (100 époques sur 500)
- **Environnement**: 1-2 obstacles, **intensité variable par simulation**
- **Innovation**: 
  - Intensité **fixe pendant chaque simulation** (0.0 à 1.0)
  - Intensité **différente à chaque nouvelle simulation**
  - Curriculum progressif des plages d'intensité: [0.5,1.0] → [0.0,1.0]
- **Métriques**: Convergence adaptée (seuil: 0.001)
- **Curriculum**: Learning rate réduit (×0.4), élargissement progressif des intensités

## Composants Techniques Nouveaux

### 1. Configuration Étendue (`ModularConfig`)
```python
# Nouveaux ratios d'étapes
STAGE_1_RATIO = 0.3  # 30% - Sans obstacles (réduit de 50%)
STAGE_2_RATIO = 0.3  # 30% - Un obstacle  
STAGE_3_RATIO = 0.2  # 20% - Obstacles multiples
STAGE_4_RATIO = 0.2  # 20% - Intensités variables (**NOUVEAU**)

# Configuration intensités variables
VARIABLE_INTENSITY_TRAINING = True
MIN_SOURCE_INTENSITY = 0.0
MAX_SOURCE_INTENSITY = 1.0  
DEFAULT_SOURCE_INTENSITY = 1.0

# Configuration étape 4
STAGE_4_SOURCE_CONFIG = {
    'intensity_distribution': 'uniform',
    'sample_per_simulation': True,
    'fixed_during_simulation': True, 
    'intensity_range_expansion': True,
    'initial_range': [0.5, 1.0],  # Début: sources moyennes à fortes
    'final_range': [0.0, 1.0]     # Fin: toute la plage
}

# Seuils de convergence renforcés
CONVERGENCE_THRESHOLDS = {
    1: 0.0002,  # Plus strict (était 0.01)
    2: 0.0002,  # Plus strict (était 0.02)  
    3: 0.015,   # Plus strict (était 0.05)
    4: 0.001    # Nouveau seuil pour intensités variables
}
```

### 2. Gestionnaire d'Intensités (`SimulationIntensityManager`)
```python
class SimulationIntensityManager:
    """Gère les intensités variables pour l'étape 4"""
    
    def sample_simulation_intensity(self, epoch_progress: float) -> float:
        """Échantillonne une intensité pour toute la simulation"""
        # Calcule la plage progressive selon l'avancement
        # Retourne une intensité uniforme dans cette plage
        
    def get_progressive_range(self, epoch_progress: float) -> Tuple[float, float]:
        """Élargit progressivement la plage d'intensités"""
        # 0% → [0.5, 1.0] (sources moyennes à fortes)
        # 50% → [0.25, 1.0] (ajout sources faibles)  
        # 100% → [0.0, 1.0] (toute la plage, y compris éteint)
        
    def validate_intensity(self, intensity: float) -> float:
        """Valide et ajuste l'intensité dans [0.0, 1.0]"""
        
    def get_intensity_statistics(self) -> Dict[str, float]:
        """Statistiques des intensités utilisées (moyenne, écart-type, min, max)"""
```

### 3. Simulateur de Diffusion Étendu (`DiffusionSimulator`)
```python
class DiffusionSimulator:
    def step(self, grid, source_mask, obstacle_mask, source_intensity=None):
        """Pas de diffusion avec support intensité variable"""
        # Si source_intensity fourni: applique cette valeur constante
        # Sinon: comportement original (intensité 1.0)
        
    def generate_stage_sequence(self, stage, n_steps, size, seed=None, source_intensity=None):
        """Génère séquence adaptée à chaque étape"""
        # Étapes 1-3: intensité standard (1.0)
        # Étape 4: utilise source_intensity spécifiée
        # Retourne: (séquence, source_mask, obstacle_mask, intensité_utilisée)
```

### 4. Entraîneur Modulaire Étendu (`ModularTrainer`)
```python
class ModularTrainer:
    def train_stage_4(self, max_epochs: int) -> Dict[str, Any]:
        """Entraînement spécialisé pour l'étape 4"""
        for epoch in range(max_epochs):
            # Calcule progression de l'époque (0.0 → 1.0)
            epoch_progress = epoch / max(max_epochs - 1, 1)
            
            for batch_idx in range(cfg.BATCH_SIZE):
                # **INNOVATION**: Échantillonne nouvelle intensité par simulation
                intensity = self.intensity_manager.sample_simulation_intensity(epoch_progress)
                
                # Génère simulation avec intensité fixe
                target_seq, source_mask, obstacle_mask, used_intensity = \
                    simulator.generate_stage_sequence(stage=4, source_intensity=intensity)
                
                # Entraîne avec cette intensité constante
                loss = self.train_step(target_seq, source_mask, obstacle_mask, 4, intensity)
        
        # Retourne métriques étendues avec statistiques d'intensité
```

### 5. Mise à Jour des NCA Updaters
```python
class OptimizedNCAUpdater:
    def step(self, grid, source_mask, obstacle_mask, source_intensity=None):
        """NCA avec support intensité variable"""
        # ...existing code pour extraction de patches...
        
        # **NOUVEAU**: Application d'intensité spécifique
        if source_intensity is not None:
            new_grid[source_mask] = source_intensity  # Étape 4
        else:
            new_grid[source_mask] = grid[source_mask]  # Étapes 1-3
```

## Fonctionnalités Avancées

### Logique de Convergence Améliorée (`CurriculumScheduler`)
- **Critères renforcés**: Convergence + stabilité + patience doublée
- **Validation sur 10 époques** (au lieu de 5)
- **Stabilité**: Variance < 0.001 sur les 5 dernières pertes
- **Évite les fausses convergences** qui donnaient "convergence à l'époque 10"

### Système de Visualisation Étendu (`ProgressiveVisualizer`)
- **Affichage d'intensité dans tous les GIFs**: `(I=X.XXX)` dans les titres
- **Étapes 1-3**: Affichent `I=1.000` (intensité standard)
- **Étape 4**: Affiche `I=0.XXX` (intensité variable échantillonnée)
- **Débogage facilité**: Corrélation visuelle entre performance et intensité

### Cache Optimisé par Étape (`OptimizedSequenceCache`)
- **Caches séparés** pour étapes 1-3 (150, 200, 250 séquences)
- **Pas de cache pour étape 4** (intensités variables incompatibles avec cache)
- **Libération automatique** des caches précédents pour économiser la mémoire

## Curriculum d'Intensité Progressif (Étape 4)

### Phase 1: Sources Moyennes à Fortes (0-25% des époques)
- **Plage**: [0.5, 1.0]  
- **Objectif**: Apprentissage sur sources standard et réduites
- **Exemples**: 0.5, 0.7, 0.9, 1.0

### Phase 2: Ajout des Sources Faibles (25-50% des époques)  
- **Plage**: [0.3, 1.0]
- **Objectif**: Introduction progressive des sources très faibles
- **Exemples**: 0.3, 0.4, 0.6, 0.8, 1.0

### Phase 3: Sources Très Faibles (50-75% des époques)
- **Plage**: [0.1, 1.0] 
- **Objectif**: Gestion des sources quasi-éteintes
- **Exemples**: 0.1, 0.2, 0.5, 0.9, 1.0

### Phase 4: Plage Complète (75-100% des époques)
- **Plage**: [0.0, 1.0]
- **Objectif**: Maîtrise complète, y compris source éteinte
- **Exemples**: 0.0, 0.1, 0.3, 0.7, 1.0

## Métriques et Monitoring

### Métriques par Étape
- **Étape 1**: Convergence pure, temps d'apprentissage
- **Étape 2**: Adaptation aux obstacles, robustesse  
- **Étape 3**: Gestion de complexité, validation de connectivité
- **Étape 4**: **NOUVEAU** - Adaptation aux intensités, statistiques d'intensité

### Métriques Spéciales Étape 4
```python
stage_4_metrics = {
    'intensity_stats_history': [  # Historique par époque
        {'mean': 0.75, 'std': 0.2, 'min': 0.5, 'max': 1.0, 'range': [[0.5, 1.0]]},
        # ...
    ],
    'global_intensity_stats': {   # Statistiques globales
        'count': 400, 'mean': 0.68, 'std': 0.25, 'min': 0.0, 'max': 1.0
    }
}
```

### Visualisations Étendues
- **Progression curriculum**: Graphique des pertes par étape avec seuils
- **Comparaison d'étapes**: Performance, époques utilisées, efficacité
- **Statistiques d'intensité** (étape 4): Distribution des intensités utilisées
- **Animations avec intensité**: Tous les GIFs affichent `I=X.XXX`

## Cas d'Usage Réalistes

### Applications Industrielles
- **Chauffage industriel**: Équipements de puissances fixes variables (0% à 100%)
- **Systèmes de refroidissement**: Adaptation à différents niveaux requis
- **Contrôle thermique**: Robustesse selon conditions opérationnelles
- **Maintenance préventive**: Fonctionnement même avec équipements dégradés

### Avantages de l'Approche
- **Simplicité**: Intensité fixe par simulation (pas de variation temporelle)
- **Robustesse**: Apprentissage sur large gamme d'intensités  
- **Réalisme**: Simule équipements réels avec puissances fixes différentes
- **Évolutivité**: Base pour futures extensions (sources multiples, variations temporelles)

## Validation et Tests

### Critères de Validation
- **Convergence sur toutes intensités**: Performance uniforme de 0.0 à 1.0
- **Stabilité temporelle**: Pas de divergence pendant les simulations
- **Robustesse aux cas limites**: Gestion correcte de l'intensité 0.0 (éteint)
- **Performance comparative**: Maintien de la qualité des étapes 1-3

### Tests Spécifiques Étape 4
- **Test intensité nulle**: Source éteinte → grille reste à zéro
- **Test intensité maximale**: Source à 1.0 → convergence comme étapes précédentes  
- **Test intensités intermédiaires**: Convergence proportionnelle à l'intensité
- **Test cohérence temporelle**: Intensité constante pendant toute la simulation

## Améliorations par Rapport à v7__

### Nouvelles Capacités
1. **Gestion d'intensités variables** entre simulations
2. **Curriculum progressif d'intensité** pour apprentissage graduel
3. **Visualisation d'intensité** dans tous les GIFs pour débogage
4. **Métriques d'intensité étendues** avec statistiques détaillées
5. **Logique de convergence renforcée** évitant les fausses convergences

### Optimisations
1. **Seuils de convergence plus stricts** pour meilleure qualité
2. **Architecture modulaire propre** sans conditions multiples `if stage == 4`
3. **Gestion mémoire améliorée** avec libération des caches
4. **Code simplifié et maintenable** avec responsabilités claires

### Robustesse
1. **Validation d'intensité** avec ajustements automatiques
2. **Gestion des cas limites** (intensité 0.0, connectivité)  
3. **Monitoring étendu** avec métriques par étape
4. **Débogage facilité** par affichage d'intensité dans visualisations

## Configuration Recommandée

```python
# Entraînement complet recommandé
TOTAL_EPOCHS = 500        # Augmenté pour 4 étapes
STAGE_1_EPOCHS = 150      # 30% - Base solide
STAGE_2_EPOCHS = 150      # 30% - Obstacles simples  
STAGE_3_EPOCHS = 100      # 20% - Obstacles complexes
STAGE_4_EPOCHS = 100      # 20% - Intensités variables

# Seuils de convergence stricts
CONVERGENCE_THRESHOLDS = {1: 0.0002, 2: 0.0002, 3: 0.015, 4: 0.001}

# Architecture réseau
HIDDEN_SIZE = 128         # Capacité suffisante
N_LAYERS = 3             # Profondeur équilibrée
LEARNING_RATE = 1e-3     # Learning rate de base
```

Cette version 8__ représente une extension majeure du système modulaire, ajoutant la capacité cruciale de gérer des intensités de source variables tout en maintenant la robustesse et la performance des étapes précédentes.
