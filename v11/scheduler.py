import numpy as np

from config import CONFIG


class CurriculumScheduler:
    """
    Gestionnaire de la progression automatique entre les étapes d'apprentissage.
    """
    
    
    def adjust_learning_rate(self, optimizer, stage_nb: int, epoch_in_stage: int):
        """Ajuste le learning rate selon l'étape et la progression."""
        
        base_lr = CONFIG.LEARNING_RATE
        
        # Réduction progressive par étape
        stage_multipliers = {1: 1.0, 2: 0.8, 3: 0.6}  # TODO: move into STAGE_MANAGER
        stage_lr = base_lr * stage_multipliers[stage_nb]
        
        # Décroissance cosine au sein de l'étape
        # stage_epochs = {1: CONFIG.STAGE_1_EPOCHS, 2: CONFIG.STAGE_2_EPOCHS, 3: CONFIG.STAGE_3_EPOCHS}
        # max_epochs = stage_epochs.get(stage_nb, 50)
        
        cos_factor = 0.5 * (1 + np.cos(np.pi * epoch_in_stage / CONFIG.NB_EPOCHS_BY_STAGE))
        final_lr = stage_lr * (0.1 + 0.9 * cos_factor)  # Ne descends pas sous 10% du LR de base
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = final_lr
