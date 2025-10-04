"""
Package de stages modulaires pour NCA.
Architecture découplée permettant l'ajout facile de nouveaux stages.
"""

from .base_stage import BaseStage, StageConfig, StageEnvironmentValidator
from .stage1 import Stage1, Stage1Config
from .stage2 import Stage2, Stage2Config
from .stage3 import Stage3, Stage3Config
from .stage4 import Stage4, Stage4Config, IntensityManager
from .stage_manager import StageRegistry, ModularStageManager

__all__ = [
    # Classes de base
    'BaseStage', 'StageConfig', 'StageEnvironmentValidator',
    
    # Stages concrets
    'Stage1', 'Stage1Config',
    'Stage2', 'Stage2Config',
    'Stage3', 'Stage3Config',
    'Stage4', 'Stage4Config', 'IntensityManager',
    
    # Gestionnaire
    'StageRegistry', 'ModularStageManager'
]
