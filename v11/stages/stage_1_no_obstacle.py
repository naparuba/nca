from stages.base_stage import BaseStage


class Stage1NoObstacle(BaseStage):
    NAME = 'no_obstacle'
    DISPLAY_NAME = "Sans obstable"
    COLOR = 'green'
    
    # et donner uen vraie taille random
    MIN_OBSTACLE_SIZE = 0
    MAX_OBSTACLE_SIZE = 0
    
    MIN_OBSTACLE_NB = 0
    MAX_OBSTACLE_NB = 0
