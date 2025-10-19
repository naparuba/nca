from typing import List, TYPE_CHECKING

from reality_world import RealityWorld

if TYPE_CHECKING:
    from torched import Tensor


class SimulationTemporalSequence:
    def __init__(self, reality_worlds, source_mask, obstacle_mask):
        # type: (List[RealityWorld], Tensor, Tensor) -> None
        
        self._reality_worlds = reality_worlds
        self._source_mask = source_mask
        self._obstacle_mask = obstacle_mask
    
    
    def get_reality_worlds(self):
        return self._reality_worlds
    
    
    def get_source_mask(self):
        return self._source_mask
    
    
    def get_obstacle_mask(self):
        return self._obstacle_mask
