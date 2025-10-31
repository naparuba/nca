from torched import DEVICE


class ModularConfig:
    """
    Configuration étendue pour l'apprentissage modulaire progressif.
    Hérite des paramètres de base et ajoute la gestion des étapes.
    """
    
    
    def __init__(self):
        # Paramètres matériels de base
        self.SEED = 3333
        
        self.NB_EPOCHS_BY_STAGE = 150
        self.NB_EPOCHS_FOR_EVALUATION = 250
        
        self.TOTAL_EPOCHS = self.NB_EPOCHS_BY_STAGE * 3
        
        # Paramètres de grille
        self.GRID_SIZE = 16
        self.SOURCE_INTENSITY = 1.0
        self.OBSTACLE_FULL_BLOCK_VALUE = 1.0
        
        # Paramètres d'entraînement modulaire
        self.NCA_STEPS = 20
        self.LEARNING_RATE = 1e-3
        self.BATCH_SIZE = 1
        
        # Paramètres de visualisation
        self.POSTVIS_STEPS = 50
        self.OUTPUT_DIR = "outputs"
        self.PERFORMANCE_FILE = "evaluation_performances.json"
        
        # Paramètres du modèle
        self.HIDDEN_SIZE = 128
        self.N_LAYERS = 3
        
        # STAGE CACHE SIZES
        self.STAGE_CACHE_SIZE = 250
        
        # Options de ligne de commande (surchargées par argparse)
        self.SKIP_IF_ALREADY = False
        self.VISUALIZATION_ONLY = False
        
        self.DEVICE = DEVICE


CONFIG = ModularConfig()
