from stages.base_stage import BaseStage


class Stage2OneObstacle(BaseStage):
    NAME = 'one_obstacle'
    DISPLAY_NAME = "Un obstable"
    COLOR = 'orange'
    
    # et donner uen vraie taille random
    MIN_OBSTACLE_SIZE = 2
    MAX_OBSTACLE_SIZE = 4
    
    MIN_OBSTACLE_NB = 1
    MAX_OBSTACLE_NB = 1
