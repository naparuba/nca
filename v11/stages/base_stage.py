import json
from abc import ABC
from pathlib import Path
from typing import Tuple, Dict

import torch
from config import CONFIG


class BaseStage(ABC):
    NAME = 'UNSET'
    DISPLAY_NAME = 'Étape non définie'
    
    COLOR = 'black'
    
    
    def __init__(self):
        self._stage_nb = -1  # Numéro de l'étape, à définir dans les sous-classes
        
        # Metrics
        self._metrics_epochs_trained = 0
        self._metrics_loss_history = []
        self._metrics_stage_lrs = []
    
    
    def get_name(self):
        return self.NAME
    
    
    def get_display_name(self):
        return self.DISPLAY_NAME
    
    
    def get_color(self):
        return self.COLOR
    
    
    def set_stage_nb(self, stage_nb: int):
        self._stage_nb = stage_nb
    
    
    def set_metrics(self, epochs_trained: int, loss_history: list, stage_lrs: list):
        self._metrics_epochs_trained = epochs_trained
        self._metrics_loss_history = loss_history
        self._metrics_stage_lrs = stage_lrs
    
    
    def get_stage_nb(self):
        return self._stage_nb
    
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        """
                Génère un environnement d'obstacles adapté à l'étape courante.

                Args:
                    size: Taille de la grille
                    source_pos: Position de la source (i, j)

                Returns:
                    Masque des obstacles [H, W]
                """
        raise NotImplementedError("Subclasses should implement this method.")
    
    
    def _get_stage_dir(self):
        # type: () -> Path
        stage_dir = Path(CONFIG.OUTPUT_DIR) / f"stage_{self.get_stage_nb()}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    
    
    def get_metrics(self):
        """Retourne les métriques de l'étape."""
        
        return {
            'stage_nb':       self.get_stage_nb(),
            'epochs_trained': self._metrics_epochs_trained,
            'loss_history':   self._metrics_loss_history,
        }
    
    
    def get_loss_history(self):
        return self._metrics_loss_history
    
    
    def get_metrics_epochs_trained(self):
        return self._metrics_epochs_trained
    
    
    def get_metrics_lrs(self):
        return self._metrics_stage_lrs
    
    
    def save_stage_checkpoint(self, model_state_dict: Dict, optimizer_state_dict: Dict):
        """Sauvegarde le checkpoint d'une étape."""
        
        stage_dir = self._get_stage_dir()
        
        metrics = self.get_metrics()
        
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
