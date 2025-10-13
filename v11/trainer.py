import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch import optim as optim, nn as nn

from config import CONFIG
from nca_model import ImprovedNCA
from scheduler import CurriculumScheduler
from sequences import OptimizedSequenceCache
from stage_manager import STAGE_MANAGER
from stages.base_stage import BaseStage
from torching import DEVICE
from updater import OptimizedNCAUpdater


class ModularTrainer:
    """
    Système d'entraînement modulaire progressif.
    Gère l'apprentissage par étapes avec transitions automatiques.
    """
    
    
    def __init__(self, model: ImprovedNCA):
        self._model = model
        
        # Choix de l'updater optimisé
        print("🚀 Utilisation de l'updater optimisé vectorisé")
        self._updater = OptimizedNCAUpdater(model)
        
        # Optimiseur et planificateur
        self._optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=1e-4)
        self._loss_fn = nn.MSELoss()
        
        # Curriculum et métriques
        
        self._curriculum = CurriculumScheduler()
        
        # Cache optimisé par étape
        self._sequence_cache = OptimizedSequenceCache()
        
        # État d'entraînement
        self._stage_histories = {stage_nb: {'losses': [], 'epochs': [], 'lr': []} for stage_nb in [1, 2, 3]}
        self._global_history = {'losses': [], 'stages': [], 'epochs': []}
        self._stage_start_epochs = {}
        self._total_epochs_trained = 0
    
    
    def _train_step(self, target_sequence: List[torch.Tensor], source_mask: torch.Tensor,
                    obstacle_mask: torch.Tensor) -> float:
        """
        Un pas d'entraînement adapté à l'étape courante.

        Args:
            target_sequence: Séquence cible
            source_mask: Masque des sources
            obstacle_mask: Masque des obstacles
            stage_nb: Étape courante d'entraînement

        Returns:
            Perte pour ce pas
        """
        self._optimizer.zero_grad()
        
        # Initialisation
        grid_pred = torch.zeros_like(target_sequence[0])
        grid_pred[source_mask] = CONFIG.SOURCE_INTENSITY
        
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        # Déroulement temporel
        for t_step in range(CONFIG.NCA_STEPS):
            target = target_sequence[t_step + 1]
            grid_pred = self._updater.step(grid_pred, source_mask, obstacle_mask)
            
            # Perte pondérée selon l'étape
            step_loss = self._loss_fn(grid_pred, target)
            
            total_loss = total_loss + step_loss
        
        avg_loss = total_loss / CONFIG.NCA_STEPS
        
        # Backpropagation avec clipping
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        
        self._optimizer.step()
        return avg_loss.item()
    
    
    def _train_stage(self, stage: BaseStage, max_epochs: int) -> Dict[str, Any]:
        """
        Entraînement complet d'une étape spécifique.

        Args:
            stage_nb: Numéro d'étape (1, 2, ou 3)
            max_epochs: Nombre maximum d'époques pour cette étape

        Returns:
            Dictionnaire avec les métriques de l'étape
        """
        stage_nb = stage.get_stage_nb()
        print(f"\n🎯 === ÉTAPE {stage_nb} - DÉBUT ===")
        # stage_name = {1: "Sans obstacles", 2: "Un obstacle", 3: "Obstacles multiples"}[stage_nb]
        print(f"📋 {stage.get_display_name()}")
        print(f"⏱️  Maximum {max_epochs} époques")
        
        self._stage_start_epochs[stage_nb] = self._total_epochs_trained
        
        # Initialisation du cache pour cette étape
        self._sequence_cache.initialize_stage_cache(stage_nb)
        
        # Métriques de l'étape
        stage_losses = []
        epoch_in_stage = 0
        
        # Boucle d'entraînement de l'étape
        for epoch_in_stage in range(max_epochs):
            epoch_losses = []
            
            # Ajustement du learning rate si curriculum activé
            self._curriculum.adjust_learning_rate(self._optimizer, stage_nb, epoch_in_stage)
            
            # Mélange périodique du cache
            if epoch_in_stage % 20 == 0:
                self._sequence_cache.shuffle_stage_cache(stage_nb)
            
            # Entraînement par batch
            for _ in range(CONFIG.BATCH_SIZE):
                batch_sequences = self._sequence_cache.get_stage_batch(stage_nb, 1)
                seq_data = batch_sequences[0]
                target_seq = seq_data['target_seq']
                source_mask = seq_data['source_mask']
                obstacle_mask = seq_data['obstacle_mask']
                
                loss = self._train_step(target_seq, source_mask, obstacle_mask)
                epoch_losses.append(loss)
            
            # Statistiques de l'époque
            avg_epoch_loss = np.mean(epoch_losses)
            stage_losses.append(avg_epoch_loss)
            current_lr = self._optimizer.param_groups[0]['lr']
            
            # Historiques
            self._stage_histories[stage_nb]['losses'].append(avg_epoch_loss)
            self._stage_histories[stage_nb]['epochs'].append(epoch_in_stage)
            self._stage_histories[stage_nb]['lr'].append(current_lr)
            
            self._global_history['losses'].append(avg_epoch_loss)
            self._global_history['stages'].append(stage_nb)
            self._global_history['epochs'].append(self._total_epochs_trained)
            
            self._total_epochs_trained += 1
            
            # Affichage périodique
            if epoch_in_stage % 10 == 0 or epoch_in_stage == max_epochs - 1:
                print(f"  Époque {epoch_in_stage:3d}/{max_epochs - 1} | "
                      f"Loss: {avg_epoch_loss:.6f} | "
                      f"LR: {current_lr:.2e}")
        
        # Résumé de l'étape
        final_loss = stage_losses[-1] if stage_losses else float('inf')
        
        stage_metrics = {
            'stage_nb':       stage_nb,
            'epochs_trained': epoch_in_stage + 1,
            'final_loss':     final_loss,
            'loss_history':   stage_losses
        }
        
        print(f"✅ === ÉTAPE {stage_nb} - TERMINÉE ===")
        print(f"📊 Époques entraînées: {epoch_in_stage + 1}/{max_epochs}")
        print(f"📉 Perte finale: {final_loss:.6f}")
        
        # Sauvegarde du checkpoint d'étape
        stage = STAGE_MANAGER.get_stage(stage_nb)
        stage.save_stage_checkpoint(stage_metrics, self._model.state_dict(), self._optimizer.state_dict())
        
        # Libération du cache de l'étape précédente pour économiser la mémoire
        if stage_nb > 1:
            self._sequence_cache.clear_stage_cache(stage_nb - 1)
        
        return stage_metrics
    
    
    def train_full_curriculum(self) -> Dict[str, Any]:
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
        
        # Entraînement séquentiel des 3 étapes
        all_stage_metrics = {}
        
        metrics = {}
        for stage in STAGE_MANAGER.get_stages():
            stage_nb = stage.get_stage_nb()
            metrics = self._train_stage(stage, CONFIG.NB_EPOCHS_BY_STAGE)
            all_stage_metrics[stage_nb] = metrics
        
        # ÉTAPE 1: Sans obstacles
        # stage_1_metrics = self._train_stage(1, CONFIG.STAGE_1_EPOCHS)
        # all_stage_metrics[1] = stage_1_metrics
        #
        # # ÉTAPE 2: Un obstacle
        # stage_2_metrics = self._train_stage(2, CONFIG.STAGE_2_EPOCHS)
        # all_stage_metrics[2] = stage_2_metrics
        #
        # # ÉTAPE 3: Obstacles multiples
        # stage_3_metrics = self._train_stage(3, CONFIG.STAGE_3_EPOCHS)
        # all_stage_metrics[3] = stage_3_metrics
        
        # Métriques globales
        total_time = time.time() - start_time
        total_epochs_actual = sum(metrics['epochs_trained'] for metrics in all_stage_metrics.values())
        
        global_metrics = {
            'total_epochs_planned': CONFIG.TOTAL_EPOCHS,
            'total_epochs_actual':  total_epochs_actual,
            'total_time_seconds':   total_time,
            'total_time_formatted': f"{total_time / 60:.1f} min",
            'stage_metrics':        all_stage_metrics,
            'final_loss':           metrics['final_loss'],
            'global_history':       self._global_history,
            'stage_histories':      self._stage_histories,
            'stage_start_epochs':   self._stage_start_epochs
        }
        
        print(f"\n🎉 === ENTRAÎNEMENT MODULAIRE TERMINÉ ===")
        print(f"⏱️  Temps total: {total_time / 60:.1f} minutes")
        print(f"📊 Époques totales: {total_epochs_actual}/{CONFIG.TOTAL_EPOCHS}")
        print(f"📉 Perte finale: {global_metrics['final_loss']:.6f}")
        
        # Sauvegarde du modèle final et des métriques
        self._save_final_model(global_metrics)
        
        return global_metrics
    
    
    def _save_final_model(self, global_metrics: Dict[str, Any]):
        """Sauvegarde le modèle final et toutes les métriques."""
        # Modèle final
        final_model_path = Path(CONFIG.OUTPUT_DIR) / "final_model.pth"
        torch.save({
            'model_state_dict':     self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'global_metrics':       global_metrics,
            'config':               CONFIG.__dict__
        }, final_model_path)
        
        # Métriques complètes
        full_metrics_path = Path(CONFIG.OUTPUT_DIR) / "complete_metrics.json"
        with open(full_metrics_path, 'w') as f:
            json.dump(global_metrics, f, indent=2, default=str)
        
        print(f"💾 Modèle final et métriques sauvegardés: {CONFIG.OUTPUT_DIR}")
