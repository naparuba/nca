"""
Système de stages modulaires pour NCA v9.
Chaque stage a maintenant son propre répertoire avec train.py et visualizer.py.
"""

from .base_stage import BaseStage, StageConfig, StageEnvironmentValidator
from .stage_manager import ModularStageManager

# Import des stages depuis leurs nouveaux répertoires
from .stage1 import Stage1, Stage1Config, Stage1Visualizer
from .stage2 import Stage2, Stage2Config, Stage2Visualizer
from .stage3 import Stage3, Stage3Config, Stage3Visualizer
from .stage4 import Stage4, Stage4Config, Stage4Visualizer
from .stage5 import Stage5, Stage5Config, Stage5Visualizer
from .stage6 import Stage6, Stage6Config, Stage6Visualizer

__all__ = [
    'BaseStage', 'StageConfig', 'StageEnvironmentValidator', 'ModularStageManager',
    'Stage1', 'Stage1Config', 'Stage1Visualizer',
    'Stage2', 'Stage2Config', 'Stage2Visualizer',
    'Stage3', 'Stage3Config', 'Stage3Visualizer',
    'Stage4', 'Stage4Config', 'Stage4Visualizer',
    'Stage5', 'Stage5Config', 'Stage5Visualizer',
    'Stage6', 'Stage6Config', 'Stage6Visualizer'
]
