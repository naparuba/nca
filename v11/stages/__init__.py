"""
Système de stages modulaires pour NCA v11.

REFACTORING v11:
- Auto-découverte des stages via StageAutoRegistry
- Plus d'imports explicites de Stage1, Stage2, etc.
- Plus de __all__ inutile
- Identification par slug uniquement

Pour ajouter un nouveau stage:
1. Créer un répertoire stages/mon_stage/
2. Créer train.py avec une classe héritant de BaseStage
3. Créer visualizer.py avec la classe de visualisation
4. Créer __init__.py pour exposer les classes
5. Ajouter le slug dans sequence.py si nécessaire
"""

from .base_stage import BaseStage, StageConfig, StageEnvironmentValidator
from .stage_manager import ModularStageManager
from .registry import StageAutoRegistry
from .sequence import StageSequence

# C'est tout ! Plus besoin d'importer explicitement les stages.
# Ils sont auto-découverts par le StageAutoRegistry.
