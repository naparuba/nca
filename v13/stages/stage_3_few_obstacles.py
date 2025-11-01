from stages.base_stage import BaseStage


class Stage3FewObstacles(BaseStage):
    NAME = 'few_obstacles'
    DISPLAY_NAME = "Obstacles multiples"
    COLOR = 'red'
    
    MIN_OBSTACLE_SIZE = 2
    MAX_OBSTACLE_SIZE = 4
    
    MIN_OBSTACLE_NB = 2
    MAX_OBSTACLE_NB = 4
