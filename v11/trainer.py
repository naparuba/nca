import time
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import torch

from config import CONFIG, DEVICE
from stage_manager import STAGE_MANAGER
from stages.base_stage import REALITY_LAYER
from torched import AdamW, get_MSELoss
from visualizer import get_visualizer

if TYPE_CHECKING:
    from stages.base_stage import BaseStage
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
        self._loss_fn = get_MSELoss()
    
    
    def _train_step(self, sequence):
        # type: (SimulationTemporalSequence) -> float
        
        """
        Un pas d'entraînement adapté à l'étape courante.
        
        Le modèle apprend à respecter les contraintes des obstacles via une pénalité forte
        dans la fonction de perte, plutôt que par forçage explicite après prédiction.

        Args:
            sequence: Sequence d'entraînement
        Returns:
            Perte pour ce pas
        """
        
        self._optimizer.zero_grad()
        
        reality_worlds = sequence.get_reality_worlds()
        source_mask = sequence.get_source_mask()
        obstacle_mask = sequence.get_obstacle_mask()
        
        # Initialisation
        grid_pred = torch.zeros_like(reality_worlds[0].get_as_tensor())
        grid_pred[REALITY_LAYER.TEMPERATURE][source_mask] = CONFIG.SOURCE_INTENSITY  # Set Les sources
        grid_pred[REALITY_LAYER.OBSTACLE][obstacle_mask] = CONFIG.OBSTACLE_FULL_BLOCK_VALUE  # Set Les obstacles
        
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        # Déroulement temporel
        for t_step in range(CONFIG.NCA_STEPS):
            target = reality_worlds[t_step + 1].get_as_tensor()
            grid_pred = self._model.run_step(grid_pred, source_mask)
            
            # Perte standard sur la prédiction globale de la température
            step_loss = self._loss_fn(grid_pred, target)
            
            total_loss = total_loss + step_loss
        
        avg_loss = total_loss / CONFIG.NCA_STEPS
        
        # combined_loss = avg_loss
        combined_loss_float = avg_loss.item()
        
        # print(f' avg_loss: {avg_loss.item():.6f}, '
        #        f' obstacle_penalty: {avg_obstacle_penalty.item():.6f}, '
        #        f' cold_zone_penalty: {avg_cold_zone_penalty.item():.6f}, '
        #        f' combined_loss: {combined_loss_float:.6f}')
        
        # Backpropagation avec clipping
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        
        self._optimizer.step()
        
        if combined_loss_float == float('nan'):
            raise ValueError("NaN loss encountered during training step.")
        
        # On retourne la perte combinée pour le monitoring
        return combined_loss_float
    
    
    def _adjust_learning_rate(self, stage_nb, epoch_in_stage):
        # type: (int, int) -> None
        """Ajuste le learning rate selon l'étape et la progression."""
        
        base_lr = CONFIG.LEARNING_RATE
        
        # Réduction progressive par étape, from 1.0 -> 0.6
        stage_lr = base_lr * (1.0 - ((stage_nb - 1) / (len(STAGE_MANAGER.get_stages()) - 1)) * 0.4)
        
        # Décroissance cosine au sein de l'étape
        cos_factor = 0.5 * (1 + np.cos(np.pi * epoch_in_stage / CONFIG.NB_EPOCHS_BY_STAGE))
        final_lr = stage_lr * (0.1 + 0.9 * cos_factor)  # Ne descends pas sous 10% du LR de base
        
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = final_lr
    
    
    def _train_stage(self, stage, max_epochs):
        # type: (BaseStage, int) -> None
        """
        Entraînement complet d'une étape spécifique.

        Args:
            stage: Serieux?
            max_epochs: Nombre maximum d'époques pour cette étape

        Returns:
            Dictionnaire avec les métriques de l'étape
        """
        stage_nb = stage.get_stage_nb()
        print(f"\n🎯 === ÉTAPE {stage_nb} - DÉBUT ===")
        print(f"📋 {stage.get_display_name()}")
        print(f"⏱️  Maximum {max_epochs} époques")
        
        # Initialisation du cache pour cette étape
        stage.generate_reality_sequences_for_training()
        
        # Métriques de l'étape
        stage_losses = []
        stage_lrs = []
        epoch_in_stage = 0
        
        # Boucle d'entraînement de l'étape
        for epoch_in_stage in range(max_epochs):
            epoch_losses = []  # type: List[float]
            
            # Ajustement du learning rate si curriculum activé
            self._adjust_learning_rate(stage_nb, epoch_in_stage)
            
            # Entraînement par batch
            for _ in range(CONFIG.BATCH_SIZE):
                sequence = stage.get_sequences_for_training()
                loss = self._train_step(sequence)  # type: float
                epoch_losses.append(loss)
            
            # Statistiques de l'époque
            avg_epoch_loss = np.mean(epoch_losses)
            stage_losses.append(avg_epoch_loss)
            current_lr = self._optimizer.param_groups[0]['lr']
            stage_lrs.append(current_lr)
            
            # Affichage périodique
            if epoch_in_stage % 10 == 0 or epoch_in_stage == max_epochs - 1:
                print(f"  Époque {epoch_in_stage:3d}/{max_epochs - 1} | "
                      f"Loss: {avg_epoch_loss:.6f} | "
                      f"LR: {current_lr:.2e}")
        
        stage.set_metrics(epoch_in_stage + 1, stage_losses, stage_lrs)
        
        print(f"✅ === ÉTAPE {stage_nb} - TERMINÉE ===")
        print(f"📊 Époques entraînées: {epoch_in_stage + 1}/{max_epochs}")
        
        # Sauvegarde du checkpoint d'étape
        stage = STAGE_MANAGER.get_stage(stage_nb)
        stage.save_stage_checkpoint(self._model.state_dict(), self._optimizer.state_dict())
    
    
    def train_full_curriculum(self) -> None:
        print(f"\n🚀 === DÉBUT ENTRAÎNEMENT MODULAIRE ===")
        print(f"🎯 Seed: {CONFIG.SEED}")
        print(f"📊 Époques totales prévues: {CONFIG.TOTAL_EPOCHS}")
        print(f"🔄 Époques par étapes: {CONFIG.NB_EPOCHS_BY_STAGE}")
        
        start_time = time.time()
        self._model.train()
        
        # Entraînement séquentiel
        for stage in STAGE_MANAGER.get_stages():
            self._train_stage(stage, CONFIG.NB_EPOCHS_BY_STAGE)
            
            # Use the current model state to compute the visualizations for this stage
            get_visualizer().visualize_stage_results(self._model, stage)
            
        
        # Métriques globales
        total_time = time.time() - start_time
        
        print(f"\n🎉 === ENTRAÎNEMENT MODULAIRE TERMINÉ ===")
        print(f"⏱️  Temps total: {total_time / 60:.1f} minutes")
        print(f"📊 Époques totales: {CONFIG.TOTAL_EPOCHS}")
        
        # Sauvegarde du modèle final et des métriques
        self._save_final_model()
        
        return
    
    
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
