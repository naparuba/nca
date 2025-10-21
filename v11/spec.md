# Spécifications Complètes - NCA Modulaire v11
## Architecture de Visualisation Modulaire et Système de Stages Simplifié

*Version v11 - Mise à jour 20 octobre 2025*

---

## Vue d'Ensemble du Système

Le NCA Modulaire v11 représente une **évolution majeure vers la modularisation** en se concentrant sur la **séparation claire des responsabilités** et l'**apprentissage par pénalités** plutôt que par contraintes hard-codées.

### Innovation Principale : Apprentissage des Contraintes par Pénalités

- **Apprentissage des obstacles** : Le modèle apprend à respecter les obstacles via des pénalités fortes dans la fonction de perte
- **Pas de forçage explicite** : Les contraintes ne sont plus imposées après prédiction, mais apprises
- **Architecture multi-canaux (à venir)** : Grille avec plusieurs canaux d'information (chaleur, obstacles, sources)

---

## État Actuel de l'Architecture (20 octobre 2025)

### 1. Architecture du Modèle NCA

#### Structure des Entrées (Actuelle - 10 features)

Le modèle reçoit pour chaque cellule :
- **9 valeurs** : Patch 3x3 des VOISINS (ne contient PAS la cellule centrale)
- **1 valeur** : Flag "suis-je un obstacle ?" (0.0 ou 1.0)

**Pourquoi le flag obstacle est nécessaire ?**
Le patch 3x3 extrait par `F.unfold` contient uniquement les 9 cellules AUTOUR de la cellule traitée, pas la cellule elle-même. Donc si une cellule est un obstacle, cette information n'est pas dans son patch 3x3. Le modèle DOIT recevoir explicitement le flag "suis-je un obstacle ?" pour pouvoir apprendre à ne pas se modifier.

#### Apprentissage par Pénalités

**Principe** : Le modèle n'a plus de forçage explicite des valeurs après prédiction.

**Mécanisme** :
1. **Initialisation** : Les obstacles sont mis à 0.0 dans la grille initiale
2. **Prédiction libre** : Le modèle NCA produit des deltas pour TOUTES les cellules, y compris les obstacles
3. **Pénalité forte** : La fonction de perte calcule `MSE(grid_pred[obstacle_mask], 0) × OBSTACLE_PENALTY_WEIGHT`
4. **Apprentissage** : Le modèle apprend progressivement à produire delta≈0 pour les obstacles

**Avantages** :
- Le modèle apprend vraiment la contrainte au lieu de se la faire imposer
- Plus cohérent avec l'esprit des automates cellulaires neuronaux
- Permet de vérifier si le modèle a bien appris (pas de "triche" par forçage)

**Configuration** :
- `OBSTACLE_PENALTY_WEIGHT = 50.0` : Poids de la pénalité (configurable dans `config.py`)

#### Utilisation de `obstacle_mask`

**Trois utilisations distinctes** :
1. ✅ **Dans `base_stage.py`** : Initialise les obstacles à 0 dans la grille de référence
2. ✅ **Dans `run_step` (nca_model.py)** : Fourni comme feature d'entrée (10ème dimension) car le patch 3x3 ne contient pas la cellule elle-même
3. ✅ **Dans `trainer.py`** : Calcule la pénalité sur les cellules marquées comme obstacles

**Important** : `obstacle_mask` n'est JAMAIS utilisé pour forcer les valeurs après prédiction.

---

## Modifications Prévues : Grille Multi-Canaux

### Problème Identifié

**Architecture actuelle** : Grille 1D (un seul canal)
- `grid[i, j]` = valeur de chaleur (entre 0.0 et 1.0)
- Ambiguïté : Un obstacle à 0.0 ressemble à une zone sans chaleur
- Nécessite de passer `obstacle_mask` séparément à `run_step`

**Conséquence** : L'information des obstacles n'est pas "dans" la grille physique mais dans un paramètre séparé.

### Solution : Grille Multi-Canaux (2D)

**Nouvelle architecture** : Grille avec 2 canaux
- `grid[0, i, j]` = Canal de chaleur (valeur entre 0.0 et 1.0)
- `grid[1, i, j]` = Canal obstacle (0.0 = pas d'obstacle, 1.0 = obstacle)

**Avantages** :
- ✅ Toute l'information est dans la grille elle-même
- ✅ Plus cohérent physiquement : plusieurs propriétés par cellule
- ✅ Extensible : on peut ajouter d'autres canaux (température, pression, etc.)
- ✅ Le modèle lit directement dans `grid[1, :, :]` les obstacles

### Impacts sur le Code

#### 1. `base_stage.py` - Génération des Séquences

**Changements nécessaires** :
```python
# AVANT (actuel)
grid = torch.zeros((size, size), device=CONFIG.DEVICE)
grid[i0, j0] = CONFIG.SOURCE_INTENSITY

# APRÈS (à implémenter)
grid = torch.zeros((2, size, size), device=CONFIG.DEVICE)
grid[0, i0, j0] = CONFIG.SOURCE_INTENSITY  # Canal chaleur
grid[1, :, :] = obstacle_mask.float()       # Canal obstacle
```

**Méthode `_play_diffusion_step`** :
```python
# Travailler uniquement sur le canal 0 (chaleur)
x = grid[0].unsqueeze(0).unsqueeze(0)
new_grid_heat = F.conv2d(x, self._kernel_avg_3x3, padding=1).squeeze()

# Reconstruire la grille complète
new_grid = grid.clone()
new_grid[0] = new_grid_heat
# grid[1] reste inchangé (les obstacles ne bougent pas)
```

#### 2. `nca_model.py` - Modèle NCA

**Changements nécessaires** :

**Architecture d'entrée** : 18 features au lieu de 10
- 9 valeurs du patch 3x3 sur le canal chaleur
- 9 valeurs du patch 3x3 sur le canal obstacle
- Total : 18 features

**Extraction des patches** :
```python
# Extraire les patches sur les 2 canaux
grid_padded = F.pad(grid.unsqueeze(0), (1, 1, 1, 1), mode='replicate')
patches_heat = F.unfold(grid_padded[:, 0:1, :, :], kernel_size=3, stride=1)  # [1, 9, H*W]
patches_obstacle = F.unfold(grid_padded[:, 1:2, :, :], kernel_size=3, stride=1)  # [1, 9, H*W]

# Concaténer
patches = torch.cat([patches_heat, patches_obstacle], dim=1)  # [1, 18, H*W]
patches = patches.squeeze(0).transpose(0, 1)  # [H*W, 18]
```

**Signature de `run_step`** :
```python
# AVANT (actuel)
def run_step(self, grid, obstacle_mask):
    # grid: [H, W]
    # obstacle_mask: [H, W]

# APRÈS (à implémenter)
def run_step(self, grid):
    # grid: [2, H, W] avec grid[0]=chaleur, grid[1]=obstacles
    # Plus besoin de obstacle_mask en paramètre !
```

**Application des deltas** :
```python
# Les deltas ne s'appliquent QUE sur le canal chaleur
new_grid = grid.clone()
new_grid[0] = new_grid[0].flatten() + deltas.squeeze()
new_grid[0] = torch.clamp(new_grid[0], 0.0, 1.0)
# new_grid[1] reste inchangé (obstacles constants)
```

#### 3. `trainer.py` - Entraînement

**Changements nécessaires** :

**Initialisation** :
```python
# AVANT (actuel)
grid_pred = torch.zeros_like(reality_worlds[0].get_as_tensor())
grid_pred[source_mask] = CONFIG.SOURCE_INTENSITY

# APRÈS (à implémenter)
grid_pred = reality_worlds[0].get_as_tensor().clone()
# Déjà initialisé avec les 2 canaux par le stage
grid_pred[0, source_mask] = CONFIG.SOURCE_INTENSITY
```

**Appel au modèle** :
```python
# AVANT (actuel)
grid_pred = self._model.run_step(grid_pred, obstacle_mask)

# APRÈS (à implémenter)
grid_pred = self._model.run_step(grid_pred)
# Plus besoin de passer obstacle_mask !
```

**Calcul de la pénalité** :
```python
# La pénalité utilise toujours obstacle_mask (vérification externe)
# On compare le canal chaleur aux positions d'obstacles
obstacle_values = grid_pred[0][obstacle_mask]
target_obstacle_values = torch.zeros_like(obstacle_values)
obstacle_penalty = self._loss_fn(obstacle_values, target_obstacle_values) * CONFIG.OBSTACLE_PENALTY_WEIGHT

# Perte standard ne compare QUE le canal chaleur
step_loss = self._loss_fn(grid_pred[0], target[0])
```

#### 4. `reality_world.py` - Monde de Référence

**Changements nécessaires** :
```python
# Stocker une grille multi-canaux [2, H, W]
# Pas de changement de structure, juste adaptation aux dimensions
```

#### 5. `visualizer.py` - Visualisation

**Changements nécessaires** :
```python
# Visualiser uniquement le canal chaleur grid[0]
# Pour les overlays d'obstacles, utiliser grid[1]

# Exemple :
heat_map = grid[0].detach().cpu().numpy()
obstacle_map = grid[1].detach().cpu().numpy()
```

---

## Configuration Actuelle

### Paramètres du Modèle (config.py)

```python
HIDDEN_SIZE = 128           # Taille des couches cachées
N_LAYERS = 3               # Nombre de couches dans le réseau
OBSTACLE_PENALTY_WEIGHT = 50.0  # Poids de la pénalité pour les obstacles
```

### Paramètres d'Entraînement

```python
NCA_STEPS = 20             # Nombre de pas de simulation par séquence
LEARNING_RATE = 1e-3       # Learning rate de base
BATCH_SIZE = 4             # Taille des batchs
NB_EPOCHS_BY_STAGE = 200   # Nombre d'époques par stage
```

---

## Flux d'Apprentissage Actuel

### 1. Génération des Données (base_stage.py)

```
1. Créer grille [H, W] avec obstacles à 0
2. Placer source à SOURCE_INTENSITY
3. Simuler N pas de diffusion avec _play_diffusion_step
4. Stocker séquence temporelle dans SimulationTemporalSequence
```

### 2. Entraînement (trainer.py)

```
Pour chaque batch :
  1. Initialiser grid_pred avec obstacles à 0, sources à SOURCE_INTENSITY
  2. Pour chaque pas temporel :
     a. grid_pred = model.run_step(grid_pred, obstacle_mask)
     b. Calculer MSE(grid_pred, target)
     c. Calculer pénalité MSE(grid_pred[obstacle_mask], 0) × 50
  3. Loss totale = MSE moyenne + Pénalité moyenne
  4. Backpropagation et mise à jour
```

### 3. Prédiction (nca_model.py)

```
run_step(grid, obstacle_mask):
  1. Extraire patches 3x3 pour toutes les cellules → [H*W, 9]
  2. Ajouter flag obstacle → [H*W, 10]
  3. Passer dans le réseau neuronal → deltas [H*W, 1]
  4. Appliquer deltas sur la grille
  5. Clamper dans [0, 1]
  6. Retourner nouvelle grille (SANS forçage des obstacles)
```

---

## TODO Liste pour la Transition Multi-Canaux

### Phase 1 : Modifications de Base
- [ ] Modifier `base_stage.generate_simulation_temporal_sequence` pour créer grille [2, H, W]
- [ ] Adapter `_play_diffusion_step` pour travailler sur canal 0 uniquement
- [ ] Mettre à jour `reality_world.py` si nécessaire

### Phase 2 : Modèle NCA
- [ ] Changer `input_size` de 10 à 18 dans `nca_model.py`
- [ ] Modifier extraction des patches pour gérer 2 canaux
- [ ] Adapter `run_step` pour accepter `grid[2, H, W]` sans `obstacle_mask`
- [ ] Modifier application des deltas sur canal 0 uniquement

### Phase 3 : Entraînement
- [ ] Adapter initialisation dans `trainer._train_step`
- [ ] Modifier appel à `run_step` (sans obstacle_mask)
- [ ] Adapter calcul de perte pour canal 0 uniquement
- [ ] Vérifier que la pénalité fonctionne toujours correctement

### Phase 4 : Visualisation
- [ ] Adapter `visualizer.py` pour afficher `grid[0]`
- [ ] Utiliser `grid[1]` pour overlays d'obstacles

### Phase 5 : Tests et Validation
- [ ] Tester génération de séquences
- [ ] Vérifier dimensions des tenseurs à chaque étape
- [ ] Valider que l'apprentissage fonctionne
- [ ] Comparer performances avant/après

---

## Notes Techniques Importantes

### Pourquoi F.unfold ne contient pas la cellule centrale ?

`F.unfold(kernel_size=3)` extrait des patches 3x3, mais ces patches sont des **voisinages** :
```
Pour une cellule en position (i, j), le patch contient :
[0] = grid[i-1, j-1]  (haut-gauche)
[1] = grid[i-1, j]    (haut)
[2] = grid[i-1, j+1]  (haut-droit)
[3] = grid[i, j-1]    (gauche)
[4] = grid[i, j]      (CENTRE - la cellule elle-même)
[5] = grid[i, j+1]    (droite)
[6] = grid[i+1, j-1]  (bas-gauche)
[7] = grid[i+1, j]    (bas)
[8] = grid[i+1, j+1]  (bas-droit)
```

**CORRECTION** : En fait, `F.unfold` CONTIENT bien la cellule centrale en position [4] ! Mais dans le contexte des NCA, on veut parfois séparer la cellule centrale de son voisinage.

**AVEC ARCHITECTURE MULTI-CANAUX** : La cellule centrale sera dans le patch obstacle (canal 1), ce qui donne au modèle l'information "suis-je un obstacle ?" directement dans les données.

---

## Justification de l'Architecture Multi-Canaux

### Cohérence Physique
Dans une simulation physique réelle, chaque point de l'espace a plusieurs propriétés :
- Température
- Densité du matériau
- Conductivité thermique
- etc.

L'architecture multi-canaux reflète cette réalité.

### Extensibilité Future
Avec cette architecture, on peut facilement ajouter :
- Canal 2 : Type de matériau (bois, métal, etc.)
- Canal 3 : Sources de chaleur variables
- Canal 4 : Zones de refroidissement
- etc.

### Simplicité du Code
Plus besoin de passer des dizaines de masques en paramètres : tout est dans la grille !

---

## Références et Ressources

### Architecture du Modèle
- Input actuel : 10 features (9 voisins + 1 flag obstacle)
- Input futur : 18 features (9 voisins chaleur + 9 voisins obstacle)
- Hidden layers : 3 couches de 128 neurones avec BatchNorm et Dropout
- Output : 1 delta par cellule, scalé par 0.1

### Fonction de Perte
- Perte principale : MSE sur toute la grille
- Pénalité obstacles : MSE sur obstacles × 50.0
- Perte totale : Moyenne sur NCA_STEPS pas temporels

---

*Document mis à jour le 20 octobre 2025*
*Prochaine étape : Implémentation de l'architecture multi-canaux*
