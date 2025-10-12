import json
from abc import ABC
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from config import CONFIG


class BaseStage(ABC):
    NAME = 'UNSET'
    DISPLAY_NAME = 'Étape non définie'
    
    
    def __init__(self):
        self._stage_nb = -1  # Numéro de l'étape, à définir dans les sous-classes
    
    
    def get_name(self):
        return self.NAME
    
    
    def get_display_name(self):
        return self.DISPLAY_NAME
    
    
    def set_stage_nb(self, stage_nb: int):
        self._stage_nb = stage_nb
    
    
    def get_stage_nb(self):
        return self._stage_nb
    
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def _get_stage_dir(self):
        # type: () -> Path
        stage_dir = Path(CONFIG.OUTPUT_DIR) / f"stage_{self.get_stage_nb()}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    
    
    def save_stage_checkpoint(self, metrics: Dict[str, Any], model_state_dict: Dict, optimizer_state_dict: Dict):
        """Sauvegarde le checkpoint d'une étape."""
        
        stage_dir = self._get_stage_dir()
        
        # Sauvegarde du modèle
        model_path = stage_dir / "model_checkpoint.pth"
        torch.save({
            'model_state_dict':     model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'stage_nb':             self.get_stage_nb(),
            'metrics':              metrics,
            'config':               CONFIG.__dict__
        }, model_path)
        
        # Sauvegarde des métriques en JSON
        metrics_path = stage_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"💾 Checkpoint étape {self.get_stage_nb()} sauvegardé: {stage_dir}")
