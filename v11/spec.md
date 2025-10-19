# Spécifications Complètes - NCA Modulaire v11
## Architecture de Visualisation Modulaire et Système de Stages Simplifié

*Version v11 - 19 octobre 2025*

---

## Vue d'Ensemble du Système

Le NCA Modulaire v11 représente une **évolution majeure vers la modularisation** en se concentrant sur la **séparation claire des responsabilités** et l'**architecture de visualisation avancée**. Cette version privilégie la simplicité d'architecture tout en conservant la puissance du système modulaire de stages.

### Innovation Principale : Architecture Modulaire Épurée

- **Séparation visualisation/simulation** : Découplage complet entre génération de données et visualisation
- **Système de stages simplifié** : Architecture modulaire basée sur des classes de base abstraites
- **Visualisation progressive** : Système de visualisation spécialisé par étape avec comparaisons temporelles
- **Configuration centralisée** : Point unique de configuration pour tous les modules

---

## Analyse des Concepts et Architecture Originale

### Concepts Métier Fondamentaux

#### Neural Cellular Automaton (NCA) - Cœur du Système
- **Paradigme** : Apprentissage distribué avec règles cellulaires neuronales
- **Architecture cellulaire** : Grille 2D où chaque cellule évolue selon son voisinage
- **Règle d'évolution** : Réseau neuronal apprend les transitions d'état locales
- **Contraintes physiques** : Respect des conditions aux limites (sources, obstacles)

#### Diffusion de Chaleur - Modèle Physique
- **Source thermique** : Points d'émission de chaleur d'intensité variable
- **Obstacles** : Zones imperméables avec température fixée
- **Propagation** : Diffusion par convolution locale avec voisinage 3x3
- **Équilibre thermique** : Convergence vers un état stable

#### RealityWorld - Abstraction de l'Environnement
- **Monde de référence** : État cible que le NCA doit apprendre à reproduire
- **Représentation tensorielle** : Grille 2D avec valeurs de température normalisées [0,1]
- **Masques de contraintes** : Séparation entre sources, obstacles et zones libres

#### SimulationTemporalSequence - Séquence d'Apprentissage
- **Séquence temporelle** : Suite ordonnée d'états de référence
- **Génération procédurale** : Création automatique de séquences d'entraînement
- **Cohérence temporelle** : Respect des lois physiques entre les pas de temps

---

## Spécifications Fonctionnelles

### 1. Architecture Modulaire des Stages

#### Classe BaseStage - Interface Abstraite
**Responsabilité** : Définir le contrat commun à tous les stages d'apprentissage

**Méthodes clés** :
- `get_stage_nb()` : Identification unique du stage
- `generate_simulation_temporal_sequence()` : Génération des données d'entraînement
- `get_loss_history()` : Historique des pertes pour le suivi de convergence
- `get_metrics_lrs()` : Évolution du learning rate au cours du temps
- `get_color()` : Couleur distinctive pour les visualisations

**Architecture découplée** :
- Chaque stage encapsule sa logique métier spécifique
- Interface standardisée permettant l'extensibilité
- Gestion autonome des paramètres d'entraînement

#### Stage1 - Apprentissage de Base
**Objectif** : Diffusion pure sans obstacles

**Caractéristiques** :
- Environnement : Grille vide avec source centrale
- Complexité minimale pour établir les bases
- Convergence rapide vers l'équilibre thermique simple

#### Stage2 - Introduction d'un Obstacle
**Objectif** : Apprentissage de la contournement d'obstacle unique

**Caractéristiques** :
- Un seul obstacle positionné stratégiquement
- Apprentissage de la diffusion avec redirection
- Complexité géométrique contrôlée

#### Stage3 - Obstacles Multiples
**Objectif** : Gestion de configurations complexes

**Caractéristiques** :
- Plusieurs obstacles avec interactions
- Patterns de diffusion plus sophistiqués
- Généralisation des stratégies de contournement

### 2. Système de Visualisation Modulaire

#### ProgressiveVisualizer - Moteur de Visualisation
**Responsabilité** : Génération automatique de visualisations comparatives par stage

**Fonctionnalités principales** :

##### Visualisation par Stage
- **Méthode** : `visualize_stage_results(model, stage)`
- **Génération de séquences de test** : Seed fixe pour reproductibilité
- **Comparaison cible/prédiction** : Visualisation côte à côte en temps réel
- **Métriques de convergence** : Graphiques d'évolution de l'erreur MSE

##### Animations Comparatives
- **Format** : GIF avec comparaison side-by-side
- **Contenu** : Évolution temporelle cible vs NCA
- **Annotations** : Obstacles mis en évidence par contours cyan
- **Paramètres** : Colormap 'hot', normalisation [0,1], 5 FPS

##### Graphiques de Convergence
- **Métriques** : Erreur MSE temporelle
- **Visualisation** : Courbe d'évolution avec grille
- **Sauvegarde** : PNG haute résolution (150 DPI)
- **Métadonnées** : Seed de génération, numéro d'étape

#### Résumé de Curriculum
**Méthode** : `create_curriculum_summary()`

**Visualisations globales** :
- **Progression multi-stages** : Évolution des pertes par étape
- **Learning rate adaptatif** : Suivi des ajustements par stage
- **Comparaison inter-stages** : Couleurs distinctives par étape
- **Échelles logarithmiques** : Meilleure lisibilité des convergences

---

## Spécifications Techniques

### 1. Architecture de Classes

#### Structure Modulaire
```
v11/
├── config.py              # Configuration centralisée (vide - en attente)
├── main.py                # Point d'entrée principal (vide - en attente)
├── nca_model.py           # Modèle NCA (vide - en attente)
├── reality_world.py       # Abstraction du monde physique (vide - en attente)
├── simulation_temporal_sequence.py  # Séquences temporelles (vide - en attente)
├── stage_manager.py       # Gestionnaire de stages (vide - en attente)
├── trainer.py             # Moteur d'entraînement (vide - en attente)
├── torched.py            # Utilitaires PyTorch (vide - en attente)
├── visualizer.py         # ✅ Système de visualisation modulaire
└── stages/               # Modules de stages
    ├── base_stage.py           # Interface abstraite (vide - en attente)
    ├── stage_1_no_obstacle.py  # Stage 1 (vide - en attente)
    ├── stage_2_one_obstacle.py # Stage 2 (vide - en attente)
    └── stage_3_few_obstacles.py # Stage 3 (vide - en attente)
```

#### État Actuel du Développement
**Module Implémenté** : `visualizer.py`
- Architecture complète et fonctionnelle
- Système de visualisation par stages
- Génération d'animations comparatives
- Graphiques de convergence et résumés globaux

**Modules en Attente** : Tous les autres fichiers sont vides
- Architecture définie mais implémentation manquante
- Structure modulaire préparée pour le développement futur

### 2. Dépendances et Configuration

#### Dépendances PyTorch
- **torch** : Tenseurs et opérations GPU/CPU
- **torch.no_grad()** : Optimisation mémoire pour les visualisations
- **Gestion des devices** : Support CPU/GPU transparent

#### Dépendances Visualisation
- **matplotlib.pyplot** : Génération de graphiques
- **matplotlib.animation** : Création d'animations GIF
- **numpy** : Opérations matricielles pour les métriques

#### Configuration Système
- **CONFIG** : Objet de configuration centralisé (importé mais non défini)
- **Graines aléatoires** : Reproductibilité avec seeds fixes
- **Chemins de sortie** : Structure de dossiers automatique par stage

### 3. Flux de Données et Sécurité Mémoire

#### Gestion des Tenseurs
**Principe** : Sécurité maximale avec `.detach().cpu().numpy()`
- **Détachement** : Suppression des gradients pour les visualisations
- **Migration CPU** : Conversion automatique pour matplotlib
- **Copie sécurisée** : `.clone()` pour éviter les références partagées

#### Modes d'Exécution
**Entraînement vs Évaluation** :
- `model.eval()` : Mode évaluation pour les visualisations
- `model.train()` : Retour en mode entraînement après visualisation
- `torch.no_grad()` : Contexte sans gradient pour l'efficacité

---

## Spécifications de Développement Futur

### 1. Modules à Implémenter

#### Configuration Centralisée (config.py)
**Objectif** : Point unique de configuration pour tous les modules

**Paramètres attendus** :
- `VISUALIZATION_SEED` : Graine pour reproductibilité
- `POSTVIS_STEPS` : Nombre de pas pour visualisations
- `GRID_SIZE` : Taille de la grille de simulation
- `SOURCE_INTENSITY` : Intensité des sources thermiques
- `OUTPUT_DIR` : Répertoire de sauvegarde
- `NB_EPOCHS_BY_STAGE` : Épochs par stage

#### Modèle NCA (nca_model.py)
**Responsabilité** : Implémentation du Neural Cellular Automaton

**Interface attendue** :
- `step(grid, source_mask, obstacle_mask)` : Évolution d'un pas de temps
- `eval()/train()` : Gestion des modes d'exécution
- Support des masques de contraintes

#### Gestionnaire de Stages (stage_manager.py)
**Responsabilité** : Orchestration des stages modulaires

**Fonctionnalités** :
- `STAGE_MANAGER.get_stages()` : Accès aux stages configurés
- Système de registre extensible
- Gestion des séquences d'entraînement

### 2. Intégration avec l'Architecture Existante

#### Continuité avec v10
**Héritage conceptuel** :
- Architecture modulaire basée sur BaseStage
- Système de registre de stages extensible
- Configuration découplée par stage

**Évolutions v11** :
- Simplification de l'architecture
- Focus sur la visualisation modulaire
- Préparation pour développements futurs

#### Points d'Extension
**Nouveaux stages** :
- Héritage de BaseStage
- Implémentation des méthodes abstraites
- Intégration automatique dans les visualisations

**Nouveaux types de visualisation** :
- Extension de ProgressiveVisualizer
- Nouveaux formats de sortie
- Métriques personnalisées

---

## Contraintes et Limitations

### 1. État de Développement
**Limitation majeure** : Architecture partiellement implémentée
- Seul le module de visualisation est fonctionnel
- Dépendances sur des modules non implémentés
- Configuration centralisée manquante

### 2. Dépendances d'Implémentation
**Modules requis pour fonctionnement complet** :
- CONFIG avec paramètres complets
- NCA avec interface `step()`
- STAGE_MANAGER avec accès aux stages
- BaseStage avec méthodes de génération

### 3. Sécurité et Robustesse
**Gestion d'erreurs** : Selon les instructions, pas de code de fallback
- Échec immédiat avec exceptions claires
- Validation stricte des entrées
- Gestion propre des ressources mémoire

---

## Conclusion

La version v11 représente une **transition architecturale** vers un système modulaire épuré avec un focus sur la visualisation avancée. Bien que l'implémentation soit actuellement partielle, l'architecture définie permet un développement futur cohérent et extensible.

**Points forts** :
- Architecture de visualisation complète et robuste
- Séparation claire des responsabilités
- Extensibilité préparée pour nouveaux stages
- Gestion sécurisée de la mémoire GPU/CPU

**Développements prioritaires** :
- Configuration centralisée (config.py)
- Modèle NCA avec interface standardisée
- Gestionnaire de stages modulaire
- Implémentation des stages de base

Cette architecture modulaire offre une base solide pour l'évolution future du système tout en maintenant la flexibilité et la maintenabilité du code.
