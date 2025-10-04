# Architecture NCA Modulaire - Proposition d'évolution

## 1. Problématique actuelle

L'architecture actuelle a séparé les stages en modules indépendants, mais il reste quelques points d'amélioration pour les découpler complètement :

- **Gestionnaire de stage centralisé** : Tous les stages sont gérés par un gestionnaire central, ce qui peut limiter l'indépendance
- **Paramètres partagés** : Certains paramètres sont encore partagés entre stages via la configuration globale
- **Logique d'exécution séquentielle** : Les stages sont exécutés en séquence fixe, limitant la flexibilité
- **Code de visualisation redondant** : Chaque stage utilise la même logique de visualisation

## 2. Architecture proposée : "StagePlugins"

Pour améliorer la modularité, nous proposons une architecture basée sur des plugins de stages complètement autonomes :

### 2.1 Structure de plugins

```
nca/
├── core/
│   ├── base_model.py         # Modèle NCA de base
│   ├── simulator.py          # Simulateur de diffusion de base
│   ├── trainer.py            # Entraîneur générique
│   └── utils.py              # Utilitaires partagés
├── stages/
│   ├── registry.py           # Registre des stages disponibles
│   ├── base_stage.py         # Interface de base pour les stages
│   ├── stage1/               # Stage sans obstacles (complètement indépendant)
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration spécifique au stage 1
│   │   ├── environment.py    # Générateur d'environnement pour stage 1
│   │   ├── trainer.py        # Extensions d'entraînement pour stage 1
│   │   └── visualizer.py     # Visualisations spécifiques au stage 1
│   ├── stage2/               # Stage un obstacle
│   │   ├── ...               # Structure similaire
│   ├── stage3/               # Stage obstacles multiples
│   │   ├── ...               # Structure similaire
│   ├── stage4/               # Stage intensités variables
│   │   ├── ...               # Structure similaire
│   └── custom/               # Espace pour stages personnalisés
├── pipeline/
│   ├── curriculum.py         # Gestionnaire de curriculum
│   ├── executor.py           # Exécuteur de pipeline modulaire
│   └── visualizer.py         # Visualiseur global
└── main.py                   # Point d'entrée principal
```

### 2.2 Interface des plugins de stages

Chaque stage sera un plugin complet avec :

```python
# Interface du plugin
class StagePlugin:
    # Métadonnées obligatoires
    id = 0                    # Identifiant unique du stage
    name = "Stage Base"       # Nom du stage
    version = "1.0"           # Version du plugin
    dependencies = []         # Stages dont dépend ce stage
    
    # Méthodes obligatoires
    @classmethod
    def get_config_schema(cls) -> Dict:
        """Retourne le schéma des paramètres configurables"""
        
    @classmethod
    def create_stage(cls, global_config) -> BaseStage:
        """Crée une instance du stage avec la configuration"""
    
    @classmethod
    def get_environment_generator(cls):
        """Retourne le générateur d'environnement pour ce stage"""
    
    @classmethod
    def get_visualizations(cls):
        """Retourne les visualisations spécifiques à ce stage"""
```

### 2.3 Système de dépendances entre stages

Les dépendances entre stages seraient explicites :

```python
# Exemple pour Stage 4
class Stage4Plugin(StagePlugin):
    id = 4
    name = "Intensités Variables"
    dependencies = [3]  # Dépend du Stage 3
```

Le gestionnaire de curriculum vérifierait les dépendances et s'assurerait que les stages sont exécutés dans le bon ordre.

### 2.4 Pipeline modulaire d'exécution

Un système de pipeline flexible permettrait d'exécuter différentes séquences de stages :

```python
# Exemple d'utilisation
pipeline = StagePipeline()

# Configuration des stages
pipeline.add_stage(Stage1Plugin)
pipeline.add_stage(Stage2Plugin)
pipeline.add_stage(Stage4Plugin)  # On peut sauter Stage3 si on veut

# Exécution
results = pipeline.execute()
```

### 2.5 Registre de stages auto-découvert

Les stages pourraient être découverts automatiquement :

```python
# Registre auto-découvert
registry = StageRegistry()
registry.discover_stages('stages')  # Recherche automatique dans le dossier stages

# Liste des stages disponibles
available_stages = registry.list_available_stages()
```

## 3. Bénéfices de la nouvelle architecture

- **Découplage complet** : Chaque stage est un module indépendant avec sa propre configuration
- **Extensibilité simple** : Ajouter un stage = créer un nouveau dossier de plugin
- **Flexibilité d'exécution** : Possibilité d'exécuter n'importe quelle séquence de stages compatibles
- **Réutilisation ciblée** : Chaque stage peut réutiliser des composants spécifiques d'autres stages
- **Testabilité améliorée** : Chaque stage peut être testé indépendamment

## 4. Exemples d'extension

### Ajouter un nouveau stage

```python
# stages/custom/stage5/plugin.py
from stages.base_stage import StagePlugin

class Stage5Plugin(StagePlugin):
    id = 5
    name = "Sources Multiples"
    dependencies = [2]  # Ne dépend que du Stage 2
    
    @classmethod
    def get_config_schema(cls):
        return {
            'min_sources': 2,
            'max_sources': 4,
            'source_spacing': 3
        }
    
    @classmethod
    def create_stage(cls, global_config):
        from .stage5 import Stage5
        return Stage5(global_config)
```

### Configuration personnalisée du pipeline

```python
# Configuration d'un pipeline spécifique
pipeline = StagePipeline(config)

# Séquence personnalisée (ex: pour tester uniquement Stage 1 et Stage 4)
pipeline.set_stages([Stage1Plugin, Stage4Plugin])

# Exécution avec hooks personnalisés
pipeline.add_hook('post_stage', my_custom_hook)
results = pipeline.execute()
```

## 5. Recommandation d'implémentation

Pour évoluer vers cette architecture, nous recommandons:

1. Créer d'abord la structure de base du système de plugins
2. Migrer progressivement chaque stage dans son propre dossier de plugin
3. Implémenter le système de dépendances et le pipeline modulaire
4. Convertir le code de visualisation pour le rendre spécifique à chaque stage
5. Créer des exemples d'extension pour valider l'architecture

Cette approche permettrait une transition progressive tout en conservant la fonctionnalité actuelle.
