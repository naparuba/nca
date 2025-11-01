# Spécifications Complètes - NCA Modulaire v11
## Architecture Multi-Canaux avec Apprentissage par Curriculum

*Version v11 - État Actuel au 22 octobre 2025*

---

## Vue d'Ensemble du Système

Le NCA Modulaire v11 est un **automate cellulaire neuronal** qui apprend à simuler la **diffusion de chaleur** dans un environnement avec obstacles, en utilisant un **apprentissage par curriculum progressif** (plusieurs stages de difficulté croissante).

### Principes Fondamentaux

1. **Architecture Multi-Canaux** : La grille contient 2 couches d'information par cellule (température + obstacles)
2. **Apprentissage par Pénalités** : Le modèle apprend à respecter les contraintes physiques via des pénalités dans la loss, pas par forçage
3. **Curriculum Progressif** : Apprentissage graduel sur 3 stages de complexité croissante
4. **Référence Physique** : Chaque séquence d'entraînement est générée par une simulation physique de référence

---

## Architecture des Données

### 1. Grille Multi-Canaux : `grid[2, H, W]`

**Structure** : Tenseur PyTorch de forme `[2, H, W]` avec :
- **`grid[0, :, :]`** = **Canal TEMPERATURE** : Valeur de chaleur dans [0.0, 1.0]
- **`grid[1, :, :]`** = **Canal OBSTACLE** : Présence d'obstacle (0.0 = libre, 1.0 = obstacle)

**Avantages** :
- ✅ Toute l'information physique de la simulation est dans la grille
- ✅ Le modèle peut lire directement les propriétés physiques des cellules
- ✅ Extensible : on peut ajouter d'autres canaux (conductivité, type de matériau, etc.)
- ✅ Cohérent avec la physique : chaque point de l'espace a plusieurs propriétés

**Constantes de couches** (dans `base_stage.py`) :
```python
class REALITY_LAYER:
    TEMPERATURE = 0  # Canal de température
    OBSTACLE = 1     # Canal d'obstacles
```

### 2. Masques Binaires

Deux masques booléens `[H, W]` accompagnent chaque séquence :

- **`source_mask[H, W]`** : Positions des sources de chaleur (True = source)
- **`obstacle_mask[H, W]`** : Positions des obstacles (True = obstacle)

**Usage** :
- Les masques servent à **initialiser** les grilles et à **forcer certaines contraintes**
- `source_mask` : Force la température des sources à rester constante pendant la simulation
- `obstacle_mask` : Utilisé pour calculer les pénalités d'entraînement (pas pour forcer les valeurs)

---

## Architecture du Modèle NCA

### 1. Entrées du Modèle : 18 Features par Cellule

Pour chaque cellule de la grille, le modèle reçoit :

- **9 valeurs** : Patch 3x3 des **voisins sur le canal TEMPERATURE** (incluant la cellule elle-même)
- **9 valeurs** : Patch 3x3 des **voisins sur le canal OBSTACLE** (incluant la cellule elle-même)
- **Total : 18 features**

**Extraction des patches** (dans `nca_model.py`) :
```python
# Padding de la grille pour gérer les bords
grid_padded = F.pad(grid.unsqueeze(0), (1, 1, 1, 1), mode='replicate')

# Extraction des patches 3x3 pour chaque couche
patches_heat = F.unfold(grid_padded[:, 0:1, :, :], kernel_size=3, stride=1)      # [1, 9, H*W]
patches_obstacle = F.unfold(grid_padded[:, 1:2, :, :], kernel_size=3, stride=1)  # [1, 9, H*W]

# Concaténation → 18 features par cellule
patches = torch.cat([patches_heat, patches_obstacle], dim=1)  # [1, 18, H*W]
```

**Pourquoi 18 features ?**
- Le modèle doit connaître son **voisinage thermique** pour calculer la diffusion
- Il doit aussi connaître les **obstacles voisins** pour adapter son comportement
- L'information "suis-je un obstacle ?" est dans le patch obstacle (position centrale)

### 2. Architecture du Réseau

**Couches cachées** (configurables dans `config.py`) :
- **3 couches** de **128 neurones** avec :
  - `Linear` → `BatchNorm1d` → `ReLU` → `Dropout(0.1)`

**Couche de sortie** :
- `Linear(128, 2)` → `Tanh` → Scaling par 0.1
- Produit **2 deltas par cellule** :
  - `delta[0]` : Changement de température
  - `delta[1]` : Changement d'obstacle (doit apprendre à produire ≈0)

**Paramètres** :
```python
HIDDEN_SIZE = 128      # Taille des couches cachées
N_LAYERS = 3           # Nombre de couches
delta_scale = 0.1      # Facteur de scaling des deltas (stabilité)
```

### 3. Méthode `run_step` : Un Pas de Simulation

**Signature** :
```python
def run_step(self, grid, source_mask):
    # grid: [2, H, W]        - Grille complète (température + obstacles)
    # source_mask: [H, W]    - Masque des sources
    # Returns: [2, H, W]     - Nouvelle grille après un pas
```

**Algorithme** :
1. **Extraction des patches** : Pour toutes les cellules, extraire les 18 features
2. **Prédiction** : Appliquer le réseau neuronal → `deltas [H*W, 2]`
3. **Application des deltas** :
   - `new_grid[0] = grid[0] + deltas[:, 0]` (température)
   - `new_grid[1] = grid[1] + deltas[:, 1]` (obstacles)
4. **Clamping** : `new_grid = clamp(new_grid, 0.0, 1.0)`
5. **Forçage des sources** : `new_grid[0, source_mask] = grid[0, source_mask]`
6. **Retour** : Nouvelle grille

**Points clés** :
- ✅ Le modèle traite **TOUTES** les cellules (pas de masquage des obstacles)
- ✅ Les deltas sont appliqués sur **les 2 couches**
- ✅ Seules les **sources** sont forcées (contrainte physique stricte)
- ⚠️ Les **obstacles** ne sont PAS forcés → le modèle doit apprendre à les conserver

---

## Génération des Données de Référence

### 1. Classe `BaseStage` : Génération des Séquences

**Rôle** : Classe abstraite pour définir un stage d'apprentissage.

**Méthode principale** : `generate_simulation_temporal_sequence(n_steps, size)`

**Algorithme** :
1. **Initialisation de la grille** :
   ```python
   grid = torch.zeros((2, size, size))  # 2 couches : température + obstacles
   ```

2. **Placement de la source** (position aléatoire) :
   ```python
   grid[REALITY_LAYER.TEMPERATURE, i0, j0] = CONFIG.SOURCE_INTENSITY  # = 1.0
   ```

3. **Génération des obstacles** (selon le stage) :
   ```python
   obstacle_mask = self.generate_obstacles(size, (i0, j0))
   grid[REALITY_LAYER.OBSTACLE, obstacle_mask] = CONFIG.OBSTACLE_FULL_BLOCK_VALUE  # = 1.0
   ```

4. **Simulation physique** (diffusion de chaleur sur `n_steps` pas) :
   - Diffusion par convolution 3x3 moyenne sur le canal température
   - Conservation de la source à intensité constante
   - Température à 0.0 dans les obstacles

5. **Stockage** : Chaque état temporel est stocké dans un `RealityWorld`

### 2. Méthode `_play_diffusion_step` : Simulation Physique

**Rôle** : Calcule un pas de diffusion de chaleur (simulation de référence).

**Algorithme** :
```python
# 1. Diffusion sur la température uniquement (convolution 3x3 moyenne)
x = grid[REALITY_LAYER.TEMPERATURE].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
new_grid_heat = F.conv2d(x, kernel_avg_3x3, padding=1).squeeze()

# 2. Mise à jour de la grille
new_grid = grid.clone()
new_grid[REALITY_LAYER.TEMPERATURE] = new_grid_heat

# 3. Contraintes physiques strictes
new_grid[REALITY_LAYER.TEMPERATURE][source_mask] = CONFIG.SOURCE_INTENSITY  # Sources constantes
new_grid[REALITY_LAYER.TEMPERATURE][obstacle_mask] = 0.0                     # Obstacles à 0°

# 4. La couche obstacles reste inchangée (les obstacles ne bougent pas)
```

**Kernel de diffusion** : Moyenne 3x3
```python
kernel_avg_3x3 = torch.ones((1, 1, 3, 3)) / 9.0
```

### 3. Stages Implémentés

**Stage 1 - `Stage1NoObstacle`** :
- **Objectif** : Apprentissage de base de la diffusion
- **Obstacles** : Aucun
- **Couleur** : Vert

**Stage 2 - `Stage2OneObstacle`** :
- **Objectif** : Apprendre à contourner un obstacle
- **Obstacles** : 1 rectangle aléatoire
- **Couleur** : Orange (probablement)

**Stage 3 - `Stage3FewObstacles`** :
- **Objectif** : Gérer plusieurs obstacles
- **Obstacles** : Plusieurs rectangles aléatoires
- **Couleur** : Rouge (probablement)

**Gestion** : La classe `StageManager` orchestre les stages.

---

## Processus d'Entraînement

### 1. Classe `Trainer` : Orchestration de l'Apprentissage

**Rôle** : Gère l'entraînement complet par curriculum progressif.

**Composants** :
- **Modèle** : Instance de `NCA`
- **Optimiseur** : `AdamW` avec weight decay
- **Loss** : `MSELoss` (Mean Squared Error)

### 2. Méthode `_train_step` : Un Pas d'Entraînement

**Algorithme** :
```python
# 1. Récupération de la séquence de référence
reality_worlds = sequence.get_reality_worlds()  # États de référence [t=0, t=1, ..., t=20]
source_mask = sequence.get_source_mask()
obstacle_mask = sequence.get_obstacle_mask()

# 2. Initialisation de la prédiction
grid_pred = torch.zeros((2, H, W))
grid_pred[REALITY_LAYER.TEMPERATURE][source_mask] = CONFIG.SOURCE_INTENSITY
grid_pred[REALITY_LAYER.OBSTACLE][obstacle_mask] = CONFIG.OBSTACLE_FULL_BLOCK_VALUE

# 3. Déroulement temporel (20 pas par défaut)
total_loss = 0
for t_step in range(CONFIG.NCA_STEPS):
    target = reality_worlds[t_step + 1].get_as_tensor()  # État de référence au temps t+1
    grid_pred = model.run_step(grid_pred, source_mask)   # Prédiction du modèle
    
    # 4. Calcul de la loss (MSE entre prédiction et référence)
    step_loss = MSE(grid_pred, target)
    total_loss += step_loss

# 5. Loss moyenne sur la séquence
avg_loss = total_loss / CONFIG.NCA_STEPS

# 6. Backpropagation avec gradient clipping
avg_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

**Points clés** :
- ✅ Le modèle apprend à **prédire l'évolution temporelle** de la diffusion
- ✅ La loss compare **toute la grille** (température + obstacles)
- ✅ Pas de pénalité explicite sur les obstacles dans le code actuel (simplification)
- ✅ Le gradient clipping évite les explosions de gradients

### 3. Learning Rate Adaptatif

**Stratégie** : Décroissance par stage + décroissance cosine par époque

```python
# Réduction progressive par stage (1.0 → 0.6)
stage_lr = base_lr * (1.0 - (stage_nb - 1) / (nb_stages - 1) * 0.4)

# Décroissance cosine au sein du stage
cos_factor = 0.5 * (1 + cos(π * epoch_in_stage / NB_EPOCHS_BY_STAGE))
final_lr = stage_lr * (0.1 + 0.9 * cos_factor)  # Minimum 10% du LR de stage
```

**Avantages** :
- ✅ LR plus élevé au début de chaque stage (exploration)
- ✅ Décroissance progressive pour la convergence fine
- ✅ Ne descend jamais sous 10% du LR de base

### 4. Entraînement Complet : `train_full_curriculum`

**Algorithme** :
```python
for stage in [Stage1, Stage2, Stage3]:
    # 1. Génération du cache de séquences (250 par défaut)
    stage.generate_reality_sequences_for_training()
    
    # 2. Entraînement du stage (50 époques par défaut)
    for epoch in range(CONFIG.NB_EPOCHS_BY_STAGE):
        for batch in range(CONFIG.BATCH_SIZE):
            sequence = stage.get_sequences_for_training()  # Séquence aléatoire du cache
            loss = _train_step(sequence)
        
        # Affichage périodique
        if epoch % 10 == 0:
            print(f"Époque {epoch} | Loss: {avg_loss:.6f}")
    
    # 3. Sauvegarde du checkpoint du stage
    stage.save_stage_checkpoint(model, optimizer)
```

**Paramètres** :
```python
NB_EPOCHS_BY_STAGE = 50    # Époques par stage
BATCH_SIZE = 4             # Séquences par époque
STAGE_CACHE_SIZE = 250     # Séquences pré-générées
NCA_STEPS = 20             # Pas temporels par séquence
```

---

## Rôles des Objets Principaux

### 1. `RealityWorld`

**Rôle** : Wrapper pour un état de grille à un instant donné.

**Structure** :
```python
class RealityWorld:
    _world: Tensor  # [2, H, W]
    
    def get_as_tensor(self):
        return self._world
```

**Usage** : Stockage des états de référence dans les séquences temporelles.

### 2. `SimulationTemporalSequence`

**Rôle** : Conteneur pour une séquence temporelle complète.

**Structure** :
```python
class SimulationTemporalSequence:
    _reality_worlds: List[RealityWorld]  # [t=0, t=1, ..., t=N]
    _source_mask: Tensor                 # [H, W]
    _obstacle_mask: Tensor               # [H, W]
```

**Usage** : Unité d'entraînement passée à `_train_step`.

### 3. `NCA` (Modèle)

**Rôle** : Réseau neuronal qui prédit les changements de grille.

**Méthodes clés** :
- `forward(x)` : Forward pass du réseau (18 inputs → 2 outputs)
- `run_step(grid, source_mask)` : Simulation d'un pas temporel

**État** : Poids du réseau (appris pendant l'entraînement)

### 4. `BaseStage` (et sous-classes)

**Rôle** : Définit un niveau de difficulté dans le curriculum.

**Responsabilités** :
- Génération des obstacles adaptés au niveau
- Création des séquences de référence
- Stockage des métriques d'entraînement
- Sauvegarde des checkpoints

**Méthode abstraite** : `generate_obstacles(size, source_pos)`

### 5. `Trainer`

**Rôle** : Orchestre l'entraînement complet.

**Responsabilités** :
- Gestion de l'optimiseur et du learning rate
- Calcul des losses
- Backpropagation et mise à jour des poids
- Coordination des stages

### 6. `ProgressiveVisualizer`

**Rôle** : Génère les visualisations post-entraînement.

**Responsabilités** :
- Comparaison visuelle NCA vs Référence
- Animations GIF de la diffusion
- Graphiques de progression (loss, learning rate)
- Résumé visuel du curriculum

### 7. `StageManager`

**Rôle** : Gestionnaire global des stages.

**Responsabilités** :
- Instanciation des stages
- Attribution des numéros de stage
- Accès centralisé aux stages

---

## Gestion des Températures

### Initialisation
- **Source** : `grid[0, i0, j0] = 1.0` (intensité maximale)
- **Reste** : `grid[0, :, :] = 0.0` (température nulle)

### Contraintes Physiques
1. **Sources** : Température **forcée** à rester constante (1.0)
   - Dans `_play_diffusion_step` (simulation de référence)
   - Dans `run_step` (prédiction du modèle)

2. **Obstacles** : Température **maintenue** à 0.0
   - Dans `_play_diffusion_step` uniquement (référence)
   - Le modèle doit **apprendre** à maintenir les obstacles froids

### Diffusion
- **Mécanisme** : Convolution 3x3 avec kernel moyen (diffusion isotrope)
- **Clamping** : Valeurs dans [0.0, 1.0]
- **Évolution** : La chaleur se propage des zones chaudes vers les zones froides

---

## Gestion des Obstacles

### Représentation
- **Dans la grille** : `grid[1, :, :] = 1.0` pour les obstacles, `0.0` ailleurs
- **Masque externe** : `obstacle_mask[H, W]` (booléen)

### Rôle dans la Simulation de Référence
- Les obstacles **bloquent** la chaleur : `temperature[obstacle_mask] = 0.0`
- Ils sont **statiques** : `grid[1, :, :]` ne change jamais

### Rôle dans l'Apprentissage
- Le modèle reçoit l'information des obstacles via les **patches 3x3**
- Il doit apprendre à :
  1. **Ne pas modifier** la couche obstacles (`delta[1] ≈ 0`)
  2. **Maintenir la température à 0** dans les obstacles
  3. **Contourner** les obstacles pour la diffusion

### Apprentissage par la Loss
- La loss MSE compare `grid_pred` et `target` sur **toutes les cellules**
- Si le modèle modifie les obstacles, la loss augmente
- Le modèle apprend donc **implicitement** à les respecter


---

## Configuration et Hyperparamètres

### Grille et Simulation
```python
GRID_SIZE = 16                        # Taille de la grille (16×16)
SOURCE_INTENSITY = 1.0                # Intensité de la source
OBSTACLE_FULL_BLOCK_VALUE = 1.0      # Valeur des obstacles
NCA_STEPS = 20                        # Pas temporels par séquence
POSTVIS_STEPS = 50                    # Pas pour les visualisations
```

### Modèle
```python
HIDDEN_SIZE = 128      # Taille des couches cachées
N_LAYERS = 3           # Nombre de couches
```

### Entraînement
```python
LEARNING_RATE = 1e-3          # Learning rate de base
BATCH_SIZE = 4                # Séquences par époque
NB_EPOCHS_BY_STAGE = 50       # Époques par stage
STAGE_CACHE_SIZE = 250        # Taille du cache de séquences
```

### Optimisations
```python
USE_MIXED_PRECISION = False   # Précision mixte (désactivé)
```

---

## Flux de Données Complet

### 1. Génération des Données (Préparation)
```
BaseStage.generate_reality_sequences_for_training()
  ↓
Pour chaque séquence (250×) :
  1. Générer obstacles selon le stage
  2. Placer source aléatoire
  3. Simuler 20 pas de diffusion (_play_diffusion_step)
  4. Stocker dans SimulationTemporalSequence
  ↓
Cache prêt pour l'entraînement
```

### 2. Entraînement (Boucle Principale)
```
Trainer.train_full_curriculum()
  ↓
Pour chaque stage (3×) :
  Pour chaque époque (50×) :
    Pour chaque batch (4×) :
      1. Récupérer séquence du cache
      2. Initialiser grid_pred avec sources + obstacles
      3. Pour chaque pas temporel (20×) :
         a. grid_pred = model.run_step(grid_pred, source_mask)
         b. loss += MSE(grid_pred, target)
      4. avg_loss = loss / 20
      5. Backpropagation + update
  ↓
  Sauvegarder checkpoint du stage
  ↓
Modèle entraîné
```

### 3. Visualisation (Post-Entraînement)
```
ProgressiveVisualizer.visualize_stage_results()
  ↓
Pour chaque stage :
  1. Générer séquence de test (seed fixe)
  2. Simuler avec le modèle (50 pas)
  3. Comparer avec la référence
  4. Créer GIF animé
  ↓
Créer résumé global (loss curves, LR curves)
```

---

## Points d'Attention et Limitations

### 1. Simplicité de la Loss
- **État actuel** : Loss MSE globale sans pénalité explicite sur les obstacles
- **Conséquence** : Le modèle apprend implicitement à respecter les obstacles

### 2. Forçage des Sources
- Les sources sont **forcées** à rester constantes dans `run_step`
- Pourquoi ? Contrainte physique stricte (source de chaleur externe)
- Alternative : Laisser le modèle apprendre, mais risque de sources qui s'éteignent

### 3. Seed et Reproductibilité
- **SEED** : Utilisé pour la génération des données d'entraînement
- **VISUALIZATION_SEED** : Utilisé pour les visualisations post-entraînement
- Important pour la reproductibilité des expériences

### 4. Taille de la Grille
- Actuellement **16×16** (petit pour des tests rapides)
- Peut être augmenté, mais attention au temps de calcul et à la mémoire

---

## Concepts Clés à Retenir

### Architecture Multi-Canaux
✅ **La grille contient plusieurs couches d'information** : température ET obstacles
✅ **Tout est dans la grille** : Pas besoin de passer des dizaines de paramètres

### Apprentissage par Curriculum
✅ **Progression graduelle** : Du simple (sans obstacles) au complexe (plusieurs obstacles)
✅ **Spécialisation** : Chaque stage cible une compétence spécifique

### Apprentissage Implicite des Contraintes
✅ **Pas de forçage** : Les obstacles ne sont pas forcés après prédiction
✅ **Apprentissage par la loss** : Le modèle apprend à respecter les contraintes via l'erreur
✅ **Plus naturel** : Le modèle comprend vraiment la physique au lieu de se la faire imposer

### Simulation de Référence
✅ **Vérité terrain** : Chaque séquence d'entraînement est une simulation physique correcte
✅ **Apprentissage supervisé** : Le modèle apprend à imiter la simulation de référence
✅ **Garantie de cohérence** : Les données d'entraînement sont physiquement cohérentes

---

*Document mis à jour le 22 octobre 2025*
*État actuel du code v11 analysé et documenté*
