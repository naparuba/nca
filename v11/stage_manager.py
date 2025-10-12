from stages.stage_1_no_obstacle import Stage1NoObstacle
from stages.stage_2_one_obstacle import Stage2OneObstacle
from stages.stage_3_few_obstacles import Stage3FewObstacles


class StageManager:
    
    def __init__(self):
        self._stages = [Stage1NoObstacle(),
                        Stage2OneObstacle(),
                        Stage3FewObstacles(),
                        ]
    
    
    def get_stage(self, stage_nb: int):
        if 1 <= stage_nb < len(self._stages)+1:
            return self._stages[stage_nb-1]
        raise IndexError(f'Stage number {stage_nb} is out of range.')


STAGE_MANAGER = StageManager()
