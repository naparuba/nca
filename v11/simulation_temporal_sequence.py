from typing import List

from reality_world import RealityWorld


class SimulationTemporalSequence:
    def __init__(self, reality_worlds):
        # type: (List[RealityWorld]) -> None
        
        self._reality_worlds = reality_worlds
    
    
    def get_reality_worlds(self):
        return self._reality_worlds
