# Analyse de Refactoring - Architecture Modulaire v11

## Date: 2025-01-06

---

## 🔴 PROBLÈMES IDENTIFIÉS

### 1. **COUPLAGE FORT AVEC LES NUMÉROS DE STAGES**

#### 1.1 Dans les Classes de Configuration
**Fichier**: `stages/stage1/train.py`, `stages/stage2/train.py`, etc.

**Problème**:
```python
class Stage1Config(StageConfig):
    def __init__(self):
        super().__init__(
            stage_id=1,  # ❌ HARD-CODÉ - impossible de réorganiser
            name="Sans obstacles",
            # ...
        )
```

**Impact**: 
- Impossible d'insérer un nouveau stage entre stage1 et stage2 sans tout renommer
- Impossible de réorganiser la séquence sans modifier le code interne des stages
- Les stages "connaissent" leur numéro, ce qui viole le principe de responsabilité unique

---

#### 1.2 Dans le StageRegistry
**Fichier**: `stages/stage_manager.py`

**Problème**:
```python
def _register_default_stages(self):
    """Enregistre les stages par défaut."""
    self.register_stage(1, Stage1)  # ❌ Numérotation explicite
    self.register_stage(2, Stage2)  # ❌ Hard-codée
    self.register_stage(3, Stage3)
    self.register_stage(4, Stage4)
    self.register_stage(5, Stage5)
```

**Impact**:
- Ajout de stage nécessite modification du code
- Pas de découplage réel
- L'ordre est implicite dans les numéros

---

#### 1.3 Dans les Imports
**Fichier**: `stages/__init__.py`

**Problème**:
```python
from .stage1 import Stage1, Stage1Config, Stage1Visualizer
from .stage2 import Stage2, Stage2Config, Stage2Visualizer
from .stage3 import Stage3, Stage3Config, Stage3Visualizer
from .stage4 import Stage4, Stage4Config, Stage4Visualizer

__all__ = [  # ❌ __all__ INUTILE et redondant
    'BaseStage', 'StageConfig', 'StageEnvironmentValidator', 'ModularStageManager',
    'Stage1', 'Stage1Config', 'Stage1Visualizer',
    'Stage2', 'Stage2Config', 'Stage2Visualizer',
    'Stage3', 'Stage3Config', 'Stage3Visualizer',
    'Stage4', 'Stage4Config', 'Stage4Visualizer'
]
```

**Impact**:
- Liste `__all__` redondante et sans valeur ajoutée
- Imports explicites par numéro de stage
- Maintenance cauchemardesque

---

#### 1.4 Dans les Visualiseurs
**Fichier**: `stages/visualizers/__init__.py`

**Problème**:
```python
from ..stage1 import Stage1Visualizer
from ..stage2 import Stage2Visualizer
from ..stage3 import Stage3Visualizer
from ..stage4 import Stage4Visualizer

STAGE_VISUALIZERS = {
    1: Stage1Visualizer,  # ❌ Mapping hard-codé par numéro
    2: Stage2Visualizer,
    3: Stage3Visualizer,
    4: Stage4Visualizer,
}

def get_visualizer(stage_id: int):  # ❌ Fonction qui dépend du numéro
    if stage_id == 5 and 5 not in STAGE_VISUALIZERS:  # ❌ Cas spécial dégueulasse
        from ..stage5.visualizer import Stage5Visualizer
        STAGE_VISUALIZERS[5] = Stage5Visualizer
```

**Impact**:
- Code spécial pour Stage5 (WTF?)
- Impossible de gérer dynamiquement les stages
- Couplage fort entre numéro et classe

---

### 2. **NOMS DE CLASSES AVEC NUMÉROS**

**Problème généralisé**:
- `Stage1`, `Stage2`, `Stage3`, `Stage4`, `Stage5`
- `Stage1Config`, `Stage2Config`, etc.
- `Stage1Visualizer`, `Stage2Visualizer`, etc.

**Impact**:
- Les noms portent la notion d'ordre
- Réorganisation = renommer toutes les classes
- Violation du principe DRY (Don't Repeat Yourself)

---

### 3. **UTILISATION DE `stage_id` PARTOUT**

**Problème**: Le `stage_id` est utilisé comme:
- Clé dans les dictionnaires
- Paramètre de méthodes
- Identifiant dans les checkpoints
- Clé dans les résultats

**Impact**:
- Toute l'architecture repose sur des entiers fragiles
- Aucune flexibilité pour réorganiser

---

## ✅ ARCHITECTURE CIBLE

### Principes de Conception

1. **Les stages ne connaissent PAS leur position dans la séquence**
2. **Une SEULE liste ordonnée définit la séquence d'exécution**
3. **Chargement dynamique des stages depuis le filesystem**
4. **Identification par nom/slug, pas par numéro**
5. **Suppression totale de `__all__`**

---

## 🔧 MODIFICATIONS REQUISES

### Modification 1: Suppression de `stage_id` dans StageConfig

**Avant**:
```python
class Stage1Config(StageConfig):
    def __init__(self):
        super().__init__(
            stage_id=1,  # ❌ À SUPPRIMER
            name="Sans obstacles",
            # ...
        )
```

**Après**:
```python
class NoObstaclesConfig(StageConfig):
    def __init__(self):
        super().__init__(
            name="no_obstacles",  # Identifiant unique, slug
            description="Apprentissage de base de la diffusion sans obstacles",
            # ... reste identique
        )
```

**Changements**:
- Supprimer `stage_id` de `StageConfig.__init__`
- Remplacer par un `name` qui sert de slug unique
- Renommer toutes les classes sans numéros

---

### Modification 2: Renommage de toutes les classes

**Mapping proposé**:

| Ancien nom | Nouveau nom | Slug |
|------------|-------------|------|
| `Stage1` | `NoObstaclesStage` | `no_obstacles` |
| `Stage2` | `SingleObstacleStage` | `single_obstacle` |
| `Stage3` | `MultipleObstaclesStage` | `multiple_obstacles` |
| `Stage4` | `VariableIntensityStage` | `variable_intensity` |
| `Stage5` | `TimeAttenuationStage` | `time_attenuation` |

**Avantages**:
- Noms descriptifs
- Indépendants de l'ordre
- Facilement réorganisables

---

### Modification 3: Chargement Dynamique des Stages

**Nouveau fichier**: `stages/registry.py`

```python
"""
Registre auto-découvert des stages disponibles.
Charge dynamiquement tous les stages depuis le filesystem.
"""

import importlib
from pathlib import Path
from typing import List, Dict, Type, Optional
from .base_stage import BaseStage


class StageAutoRegistry:
    """
    Registre qui auto-découvre les stages depuis le filesystem.
    Plus besoin d'imports explicites.
    """
    
    def __init__(self, stages_dir: Path = None):
        if stages_dir is None:
            stages_dir = Path(__file__).parent
        
        self.stages_dir = stages_dir
        self._stages: Dict[str, Type[BaseStage]] = {}
        self._discover_stages()
    
    def _discover_stages(self):
        """
        Parcourt le répertoire stages/ et charge automatiquement
        tous les modules qui contiennent une classe Stage.
        """
        # Parcours de tous les sous-répertoires
        for stage_path in self.stages_dir.iterdir():
            if not stage_path.is_dir():
                continue
            
            # Ignore les répertoires spéciaux
            if stage_path.name.startswith('_') or stage_path.name == 'visualizers':
                continue
            
            # Tentative de chargement du module train.py
            train_module_path = stage_path / 'train.py'
            if not train_module_path.exists():
                continue
            
            try:
                # Import dynamique du module
                module_name = f"stages.{stage_path.name}.train"
                module = importlib.import_module(module_name)
                
                # Recherche de la classe Stage dans le module
                stage_class = self._find_stage_class(module)
                
                if stage_class:
                    # Récupération du slug depuis la config
                    temp_instance = stage_class()
                    slug = temp_instance.config.name
                    
                    self._stages[slug] = stage_class
                    print(f"✅ Stage '{slug}' découvert et enregistré")
                
            except Exception as e:
                print(f"⚠️  Impossible de charger {stage_path.name}: {e}")
    
    def _find_stage_class(self, module) -> Optional[Type[BaseStage]]:
        """
        Trouve la classe qui hérite de BaseStage dans un module.
        """
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Vérification que c'est une classe héritant de BaseStage
            if (isinstance(attr, type) and 
                issubclass(attr, BaseStage) and 
                attr is not BaseStage):
                return attr
        
        return None
    
    def get_stage(self, slug: str) -> Type[BaseStage]:
        """Récupère une classe de stage par son slug."""
        if slug not in self._stages:
            raise ValueError(f"Stage '{slug}' non trouvé. Stages disponibles: {list(self._stages.keys())}")
        return self._stages[slug]
    
    def list_stages(self) -> List[str]:
        """Liste tous les slugs de stages disponibles."""
        return list(self._stages.keys())
    
    def create_stage(self, slug: str, device: str = "cpu", **kwargs) -> BaseStage:
        """Crée une instance d'un stage par son slug."""
        stage_class = self.get_stage(slug)
        return stage_class(device=device, **kwargs)
```

---

### Modification 4: Nouvelle Gestion de la Séquence

**Nouveau fichier**: `stages/sequence.py`

```python
"""
Définition de la séquence d'exécution des stages.
C'EST LE SEUL ENDROIT où l'ordre est défini.
"""

from typing import List


class StageSequence:
    """
    Définit l'ordre d'exécution des stages.
    Ceci est la SEULE source de vérité pour l'ordre.
    """
    
    # 🎯 SÉQUENCE MAÎTRE - L'ORDRE EST DÉFINI ICI ET NULLE PART AILLEURS
    DEFAULT_SEQUENCE = [
        'no_obstacles',
        'single_obstacle',
        'multiple_obstacles',
        'variable_intensity',
        'time_attenuation',
    ]
    
    def __init__(self, custom_sequence: List[str] = None):
        """
        Initialise la séquence.
        
        Args:
            custom_sequence: Séquence personnalisée (optionnel)
        """
        self.sequence = custom_sequence if custom_sequence else self.DEFAULT_SEQUENCE.copy()
    
    def get_sequence(self) -> List[str]:
        """Retourne la séquence ordonnée des stages."""
        return self.sequence
    
    def get_position(self, slug: str) -> int:
        """Retourne la position d'un stage dans la séquence (0-indexed)."""
        try:
            return self.sequence.index(slug)
        except ValueError:
            raise ValueError(f"Stage '{slug}' non trouvé dans la séquence")
    
    def insert_before(self, new_slug: str, reference_slug: str):
        """Insère un nouveau stage avant un stage de référence."""
        position = self.get_position(reference_slug)
        self.sequence.insert(position, new_slug)
    
    def insert_after(self, new_slug: str, reference_slug: str):
        """Insère un nouveau stage après un stage de référence."""
        position = self.get_position(reference_slug)
        self.sequence.insert(position + 1, new_slug)
    
    def remove(self, slug: str):
        """Retire un stage de la séquence."""
        self.sequence.remove(slug)
    
    def reorder(self, new_sequence: List[str]):
        """Remplace complètement la séquence."""
        self.sequence = new_sequence.copy()
```

---

### Modification 5: Refactoring du StageManager

**Changements dans `stage_manager.py`**:

```python
class ModularStageManager:
    """
    Gestionnaire principal des stages modulaires.
    Utilise maintenant le système de registry auto-découvert.
    """
    
    def __init__(self, global_config: Any, device: str = "cpu", 
                 custom_sequence: List[str] = None):
        self.global_config = global_config
        self.device = device
        
        # 🎯 Nouveau système
        self.registry = StageAutoRegistry()
        self.sequence = StageSequence(custom_sequence)
        
        # Stages actifs indexés par SLUG, pas par numéro
        self.active_stages: Dict[str, BaseStage] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Configuration d'exécution
        self.total_epochs_planned = global_config.TOTAL_EPOCHS
        self.epochs_per_stage = self._calculate_epochs_per_stage()
    
    def _calculate_epochs_per_stage(self) -> Dict[str, int]:
        """
        Calcule le nombre d'époques par stage selon leur configuration.
        Utilise maintenant les SLUGS.
        """
        epochs_per_stage = {}
        
        for slug in self.sequence.get_sequence():
            temp_stage = self.registry.create_stage(slug, self.device)
            ratio = temp_stage.config.epochs_ratio
            
            raw_epochs = self.total_epochs_planned * ratio
            epochs = int(raw_epochs)
            
            epochs_per_stage[slug] = epochs
        
        # Ajustement pour atteindre le total exact
        total_calculated = sum(epochs_per_stage.values())
        if total_calculated != self.total_epochs_planned:
            last_slug = self.sequence.get_sequence()[-1]
            adjustment = self.total_epochs_planned - total_calculated
            epochs_per_stage[last_slug] += adjustment
        
        return epochs_per_stage
```

---

### Modification 6: Suppression de `__all__`

**Dans tous les fichiers `__init__.py`**:

**Avant**:
```python
__all__ = [
    'Stage1', 'Stage1Config', 'Stage1Visualizer',
    'Stage2', 'Stage2Config', 'Stage2Visualizer',
    # ...
]
```

**Après**:
```python
# ❌ SUPPRIMÉ - Inutile et redondant
```

**Justification**:
- `__all__` n'a de valeur que pour `from module import *` qui est une MAUVAISE pratique
- Les imports explicites sont préférables
- Maintenance inutile

---

### Modification 7: Refactoring des Visualiseurs

**Nouveau système**:

```python
class VisualizerRegistry:
    """
    Registre auto-découvert des visualiseurs.
    Chaque stage fournit son propre visualiseur.
    """
    
    def __init__(self):
        self._visualizers: Dict[str, Type] = {}
        self._discover_visualizers()
    
    def _discover_visualizers(self):
        """Charge automatiquement les visualiseurs depuis chaque stage."""
        stages_dir = Path(__file__).parent
        
        for stage_path in stages_dir.iterdir():
            if not stage_path.is_dir() or stage_path.name.startswith('_'):
                continue
            
            visualizer_path = stage_path / 'visualizer.py'
            if not visualizer_path.exists():
                continue
            
            try:
                module_name = f"stages.{stage_path.name}.visualizer"
                module = importlib.import_module(module_name)
                
                # Recherche de la classe Visualizer
                visualizer_class = self._find_visualizer_class(module)
                
                if visualizer_class:
                    # Utilise le slug du stage parent
                    slug = stage_path.name  # ou extrait depuis le train.py
                    self._visualizers[slug] = visualizer_class
            
            except Exception as e:
                print(f"⚠️  Impossible de charger visualiseur {stage_path.name}: {e}")
    
    def get_visualizer(self, slug: str):
        """Récupère un visualiseur par slug de stage."""
        if slug not in self._visualizers:
            return None  # Visualiseur optionnel
        return self._visualizers[slug]()
```

---

## 📋 PLAN D'EXÉCUTION

### Phase 1: Préparation (Sans casser l'existant)
1. ✅ Créer `stages/registry.py` avec `StageAutoRegistry`
2. ✅ Créer `stages/sequence.py` avec `StageSequence`
3. ✅ Modifier `base_stage.py` pour supprimer `stage_id` de `StageConfig`

### Phase 2: Refactoring des Stages (Un par un)
4. Renommer `stage1/` → `no_obstacles/`
5. Renommer classe `Stage1` → `NoObstaclesStage`
6. Adapter `NoObstaclesConfig` (supprimer `stage_id`)
7. Répéter pour tous les stages

### Phase 3: Refactoring du Manager
8. Adapter `ModularStageManager` pour utiliser `StageAutoRegistry`
9. Remplacer tous les usages de `stage_id` par des slugs
10. Mettre à jour les méthodes de sauvegarde/chargement

### Phase 4: Refactoring des Visualiseurs
11. Adapter le système de visualiseurs pour l'auto-découverte
12. Supprimer le mapping hard-codé

### Phase 5: Nettoyage
13. Supprimer tous les `__all__`
14. Supprimer les imports explicites inutiles
15. Nettoyer les fichiers `__init__.py`

### Phase 6: Tests et Validation
16. Tester le chargement automatique
17. Tester l'insertion de nouveaux stages
18. Tester la réorganisation de la séquence
19. Valider les checkpoints et visualisations

---

## 🎯 BÉNÉFICES ATTENDUS

1. **Flexibilité totale**: Insérer/supprimer/réorganiser des stages sans toucher au code
2. **Découplage complet**: Les stages ne connaissent pas leur position
3. **Maintenance simplifiée**: Une seule source de vérité pour l'ordre
4. **Extensibilité**: Ajouter un nouveau stage = créer un répertoire
5. **Code propre**: Suppression de toute redondance et magie avec les numéros

---

## ⚠️ POINTS D'ATTENTION

1. **Compatibilité checkpoints**: Les anciens checkpoints avec `stage_id` devront être migrés
2. **Tests**: Chaque modification doit être testée isolément
3. **Documentation**: Mettre à jour la doc pour le nouveau système
4. **Performance**: L'auto-découverte est faite une fois au démarrage, pas de surcoût

---

## 📊 MÉTRIQUES DE SUCCÈS

- ✅ Aucune référence à `Stage1`, `Stage2`, etc. dans le code
- ✅ Aucun `stage_id` en paramètre
- ✅ Aucun `__all__` dans les fichiers
- ✅ Possibilité d'insérer un stage sans modifier le code existant
- ✅ Séquence définie dans UN SEUL fichier


