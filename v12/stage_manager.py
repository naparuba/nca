from typing import List, TYPE_CHECKING

from stages.stage_1_no_obstacle import Stage1NoObstacle
from stages.stage_2_one_obstacle import Stage2OneObstacle
from stages.stage_3_few_obstacles import Stage3FewObstacles
from stages.stage_4_few_sources import Stage4FewSources

if TYPE_CHECKING:
    from stages.base_stage import BaseStage


class StageManager:
    
    def __init__(self):
        self._stages = [Stage1NoObstacle(),
                        Stage2OneObstacle(),
                        Stage3FewObstacles(),
                        Stage4FewSources(),
                        ]
        for stage in self._stages:
            stage.set_stage_nb(self._stages.index(stage) + 1)
    
    
    def get_stages(self):
        # type: () -> List[BaseStage]
        return self._stages
    
    
    def get_stage(self, stage_nb):
        # type: (int) -> BaseStage
        if 1 <= stage_nb < len(self._stages) + 1:
            return self._stages[stage_nb - 1]
        raise IndexError(f'Stage number {stage_nb} is out of range.')


STAGE_MANAGER = StageManager()
