from typing import List

import torch


class SimulationSequence:
    def __init__(self, target_sequence, source_mask, obstacle_mask):
        # type: (List[torch.Tensor], torch.Tensor, torch.Tensor) -> None
        
        self._target_sequence = target_sequence
        self._source_mask = source_mask
        self._obstacle_mask = obstacle_mask
    
    
    def get_target_sequence(self):
        return self._target_sequence
    
    
    def get_source_mask(self):
        return self._source_mask
    
    
    def get_obstacle_mask(self):
        return self._obstacle_mask
