from typing import List

import numpy as np

from config import CONFIG


class CurriculumScheduler:
    """
    Gestionnaire de la progression automatique entre les étapes d'apprentissage.
    Décide quand passer à l'étape suivante selon des critères adaptatifs.
    """
    
    
    def __init__(self):
        self.stage_metrics_history = {stage_nb: [] for stage_nb in [1, 2, 3]}
        self.no_improvement_counts = {stage_nb: 0 for stage_nb in [1, 2, 3]}
    
    
    def should_advance_stage(self, current_stage: int, recent_losses: List[float]) -> bool:
        """
        Détermine s'il faut passer à l'étape suivante.

        Args:
            current_stage: Étape courante
            recent_losses: Pertes récentes pour évaluation

        Returns:
            True si on doit avancer à l'étape suivante
        """
        if not recent_losses:
            return False
        
        # Critère secondaire: stagnation (pas d'amélioration)
        if len(recent_losses) >= 2:
            improvement = recent_losses[-2] - recent_losses[-1]
            if improvement < CONFIG.STAGNATION_THRESHOLD:  # Amélioration négligeable
                self.no_improvement_counts[current_stage] += 1
            else:
                self.no_improvement_counts[current_stage] = 0
        
        stagnated = self.no_improvement_counts[current_stage] >= CONFIG.STAGNATION_PATIENCE
        
        return stagnated
    
    
    def adjust_learning_rate(self, optimizer, stage_nb: int, epoch_in_stage: int):
        """Ajuste le learning rate selon l'étape et la progression."""
        
        base_lr = CONFIG.LEARNING_RATE
        
        # Réduction progressive par étape
        stage_multipliers = {1: 1.0, 2: 0.8, 3: 0.6}
        stage_lr = base_lr * stage_multipliers.get(stage_nb, 0.5)
        
        # Décroissance cosine au sein de l'étape
        stage_epochs = {1: CONFIG.STAGE_1_EPOCHS, 2: CONFIG.STAGE_2_EPOCHS, 3: CONFIG.STAGE_3_EPOCHS}
        max_epochs = stage_epochs.get(stage_nb, 50)
        
        cos_factor = 0.5 * (1 + np.cos(np.pi * epoch_in_stage / max_epochs))
        final_lr = stage_lr * (0.1 + 0.9 * cos_factor)  # Ne descends pas sous 10% du LR de base
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = final_lr
