import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from config import CONFIG
from stage_manager import STAGE_MANAGER
from torched import AdamW
from visualizer import get_visualizer

if TYPE_CHECKING:
    from simulation_sequence import SimulationTemporalSequence
    from nca_model import NCA


class Trainer:
    """
    Système d'entraînement modulaire progressif.
    Gère l'apprentissage par étapes avec transitions automatiques.
    """
    
    
    def __init__(self, model):
        # type: (NCA) -> None
        self._model = model
        
        # Optimiseur et planificateur
        self._optimizer = AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=1e-4)
    
    
    def train_full_curriculum(self) -> None:
        print(f"\n🚀 === DÉBUT ENTRAÎNEMENT MODULAIRE ===")
        print(f"🎯 Seed: {CONFIG.SEED}")
        print(f"📊 Époques totales prévues: {CONFIG.TOTAL_EPOCHS}")
        print(f"🔄 Époques par étapes: {CONFIG.NB_EPOCHS_BY_STAGE}")
        
        start_time = time.time()
        self._model.train()
        
        # Entraînement séquentiel
        for stage in STAGE_MANAGER.get_stages():
            stage.train(self._model, self._optimizer)
            
            get_visualizer().evaluate_model_stage(self._model, stage)
            
            # Use the current model state to compute the visualizations for this stage
            get_visualizer().visualize_stage_results(self._model, stage)
        
        # Métriques globales
        total_time = time.time() - start_time
        
        print(f"\n🎉 === ENTRAÎNEMENT MODULAIRE TERMINÉ ===")
        print(f"⏱️  Temps total: {total_time / 60:.1f} minutes")
        print(f"📊 Époques totales: {CONFIG.TOTAL_EPOCHS}")
        
        # Sauvegarde du modèle final et des métriques
        self._save_final_model()
    
    
    def _save_final_model(self):
        """Sauvegarde le modèle final et toutes les métriques."""
        # Modèle final
        final_model_path = Path(CONFIG.OUTPUT_DIR) / "final_model.pth"
        torch.save({
            'model_state_dict':     self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'config':               CONFIG.__dict__
        }, final_model_path)
        
        print(f"💾 Modèle final et métriques sauvegardés: {CONFIG.OUTPUT_DIR}")
