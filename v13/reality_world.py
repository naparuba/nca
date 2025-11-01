from torched import Tensor


class RealityWorld:
    
    def __init__(self, world):
        # type: (Tensor) -> None
        self._world = world
    
    
    def get_as_tensor(self):
        return self._world
