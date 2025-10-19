from torched import DEVICE


class ModularConfig:
    """
    Configuration étendue pour l'apprentissage modulaire progressif.
    Hérite des paramètres de base et ajoute la gestion des étapes.
    """
    
    
    def __init__(self):
        # Paramètres matériels de base
        self.SEED = 3333
        
        self.VISUALIZATION_SEED = 3333
        
        self.NB_EPOCHS_BY_STAGE = 30#200
        
        self.TOTAL_EPOCHS = self.NB_EPOCHS_BY_STAGE * 3
        
        # Paramètres de grille
        self.GRID_SIZE = 16
        self.SOURCE_INTENSITY = 1.0
        
        # Paramètres d'entraînement modulaire
        self.NCA_STEPS = 20
        self.LEARNING_RATE = 1e-3
        self.BATCH_SIZE = 4
        
        # Paramètres de visualisation
        self.POSTVIS_STEPS = 50
        self.OUTPUT_DIR = "outputs"
        
        # Paramètres du modèle
        self.HIDDEN_SIZE = 128
        self.N_LAYERS = 3
        
        self.MIN_OBSTACLE_SIZE = 2
        self.MAX_OBSTACLE_SIZE = 4
        
        # Optimisations
        self.CACHE_SIZE = 200
        self.USE_MIXED_PRECISION = False
        
        # STAGE CACHE SIZES
        self.STAGE_CACHE_SIZE = 250
        
        self.DEVICE = DEVICE


CONFIG = ModularConfig()
