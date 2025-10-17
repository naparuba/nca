import time
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import torch
from torch import optim as optim, nn as nn

from config import CONFIG, DEVICE
from nca_model import NCA
from sequences import OptimizedSequenceCache
from stage_manager import STAGE_MANAGER

if TYPE_CHECKING:
    from stages.base_stage import BaseStage
    from sequence import SimulationSequence


class Trainer:
    """
    Système d'entraînement modulaire progressif.
    Gère l'apprentissage par étapes avec transitions automatiques.
    """
    
    
    def __init__(self, model: NCA):
        self._model = model
        
        # Choix de l'updater optimisé
        print("🚀 Utilisation de l'updater optimisé vectorisé")
        
        # Optimiseur et planificateur
        self._optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=1e-4)
        self._loss_fn = nn.MSELoss()
        
        # Cache optimisé par étape
        self._sequence_cache = OptimizedSequenceCache()
    
    
    def _train_step(self, sequence):
        # type: (SimulationSequence) -> float
        
        """
        Un pas d'entraînement adapté à l'étape courante.

        Args:
            sequence: Sequence d'entraînement
        Returns:
            Perte pour ce pas
        """
        
        self._optimizer.zero_grad()
        
        target_sequence = sequence.get_target_sequence()
        source_mask = sequence.get_source_mask()
        obstacle_mask = sequence.get_obstacle_mask()
        
        # Initialisation
        grid_pred = torch.zeros_like(target_sequence[0])
        grid_pred[source_mask] = CONFIG.SOURCE_INTENSITY
        
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        # Déroulement temporel
        for t_step in range(CONFIG.NCA_STEPS):
            target = target_sequence[t_step + 1]
            grid_pred = self._model.step(grid_pred, source_mask, obstacle_mask)
            
            # Perte pondérée selon l'étape
            step_loss = self._loss_fn(grid_pred, target)
            
            total_loss = total_loss + step_loss
        
        avg_loss = total_loss / CONFIG.NCA_STEPS
        
        # Backpropagation avec clipping
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        
        self._optimizer.step()
        return avg_loss.item()
    
    
    def _adjust_learning_rate(self, stage_nb: int, epoch_in_stage: int):
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
        self._sequence_cache.initialize_stage_cache(stage)
        
        # Métriques de l'étape
        stage_losses = []
        stage_lrs = []
        epoch_in_stage = 0
        
        # Boucle d'entraînement de l'étape
        for epoch_in_stage in range(max_epochs):
            epoch_losses = []  # type: List[float]
            
            # Ajustement du learning rate si curriculum activé
            self._adjust_learning_rate(stage_nb, epoch_in_stage)
            
            # Mélange périodique du cache
            # if epoch_in_stage % 20 == 0:
            #    self._sequence_cache.shuffle_stage_cache(stage_nb)
            
            # Entraînement par batch
            for _ in range(CONFIG.BATCH_SIZE):
                sequence = self._sequence_cache.get_stage_sample(stage)
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
        
        # Libération du cache de l'étape précédente pour économiser la mémoire
        if stage_nb > 1:
            self._sequence_cache.clear_stage_cache(stage_nb - 1)
    
    
    def train_full_curriculum(self) -> None:
        """
        Entraînement complet du curriculum en 3 étapes.

        Returns:
            Métriques complètes de l'entraînement modulaire
        """
        print(f"\n🚀 === DÉBUT ENTRAÎNEMENT MODULAIRE ===")
        print(f"🎯 Seed: {CONFIG.SEED}")
        print(f"📊 Époques totales prévues: {CONFIG.TOTAL_EPOCHS}")
        print(f"🔄 Époques par étapes: {CONFIG.NB_EPOCHS_BY_STAGE}")
        
        start_time = time.time()
        self._model.train()
        
        # Entraînement séquentiel
        for stage in STAGE_MANAGER.get_stages():
            self._train_stage(stage, CONFIG.NB_EPOCHS_BY_STAGE)
        
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
            # 'global_metrics':       global_metrics_,
            'config':               CONFIG.__dict__
        }, final_model_path)
        
        print(f"💾 Modèle final et métriques sauvegardés: {CONFIG.OUTPUT_DIR}")
