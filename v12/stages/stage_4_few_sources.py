from stages.base_stage import BaseStage


class Stage4FewSources(BaseStage):
    NAME = 'few_sources'
    DISPLAY_NAME = "Sources multiples"
    COLOR = 'grey'
    
    MIN_OBSTACLE_SIZE = 2
    MAX_OBSTACLE_SIZE = 4
    
    MIN_OBSTACLE_NB = 2
    MAX_OBSTACLE_NB = 4

    MIN_SOURCES_NB = 2
    MAX_SOURCES_NB = 4