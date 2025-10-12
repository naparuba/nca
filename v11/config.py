


class ModularConfig:
    """
    Configuration étendue pour l'apprentissage modulaire progressif.
    Hérite des paramètres de base et ajoute la gestion des étapes.
    """
    
    
    def __init__(self):
        # Paramètres matériels de base
        self.SEED = 3333
        
        self.VISUALIZATION_SEED = 3333
        
        self.NB_EPOCHS_BY_STAGE = 200
        
        self.TOTAL_EPOCHS = self.NB_EPOCHS_BY_STAGE * 3
        
        self.STAGNATION_THRESHOLD = 0.000001  # seuil de stagnation pour avancer d'étape
        self.STAGNATION_PATIENCE = self.NB_EPOCHS_BY_STAGE // 5  # if we flat for such epochs, we consider it stagnated
        
        
        # Paramètres de grille
        self.GRID_SIZE = 16
        self.SOURCE_INTENSITY = 1.0
        
        # Paramètres d'entraînement modulaire
        self.TOTAL_EPOCHS = self.TOTAL_EPOCHS  # Augmenté pour l'apprentissage modulaire
        self.NCA_STEPS = 20
        self.LEARNING_RATE = 1e-3
        self.BATCH_SIZE = 4
        
        # Calcul automatique des époques par étape
        self.STAGE_1_EPOCHS = self.NB_EPOCHS_BY_STAGE
        self.STAGE_2_EPOCHS = self.NB_EPOCHS_BY_STAGE
        self.STAGE_3_EPOCHS = self.NB_EPOCHS_BY_STAGE
        
        # Seuils de convergence adaptatifs par étape
        self.CONVERGENCE_THRESHOLDS = {
            1: 0.01,  # Étape 1: convergence stricte
            2: 0.02,  # Étape 2: tolérance accrue
            3: 0.05  # Étape 3: tolérance maximale
        }
        
        # Paramètres de visualisation
        self.PREVIS_STEPS = 30
        self.POSTVIS_STEPS = 50
        self.OUTPUT_DIR = "outputs"
        
        # Paramètres du modèle
        self.HIDDEN_SIZE = 128
        self.N_LAYERS = 3
        
        # Paramètres d'obstacles par étape
        # self.STAGE_OBSTACLE_CONFIG = {
        #     1: {'min_obstacles': 0, 'max_obstacles': 0},  # Pas d'obstacles
        #     2: {'min_obstacles': 1, 'max_obstacles': 1},  # Un seul obstacle
        #     3: {'min_obstacles': 2, 'max_obstacles': 4}  # 2-4 obstacles
        # }
        
        self.MIN_OBSTACLE_SIZE = 2
        self.MAX_OBSTACLE_SIZE = 4
        
        # Optimisations
        self.CACHE_SIZE = 200
        self.USE_MIXED_PRECISION = False


CONFIG = ModularConfig()
