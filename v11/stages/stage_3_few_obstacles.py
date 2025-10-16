from typing import Tuple

import torch
from config import CONFIG, DEVICE
from stages.base_stage import BaseStage



class Stage3FewObstacles(BaseStage):
    NAME = 'few_obstacles'
    DISPLAY_NAME = "Obstables multiples"
    COLOR = 'red'
    
    MIN_OBSTACLE_SIZE = 2
    MAX_OBSTACLE_SIZE = 4
    
    
    def _validate_connectivity(self, obstacle_mask: torch.Tensor, source_pos: Tuple[int, int]) -> bool:
        """
        Valide qu'un chemin de diffusion reste possible avec les obstacles.
        Utilise un algorithme de flood-fill simplifié.
        """
        H, W = obstacle_mask.shape
        source_i, source_j = source_pos
        
        # Matrice de visite
        visited = torch.zeros_like(obstacle_mask, dtype=torch.bool)
        visited[obstacle_mask] = True  # Les obstacles sont "déjà visités"
        
        # Flood-fill depuis la source
        stack = [(source_i, source_j)]
        visited[source_i, source_j] = True
        accessible_cells = 1
        
        while stack:
            i, j = stack.pop()
            
            # Parcours des 4 voisins
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < H and 0 <= nj < W and
                        not visited[ni, nj] and not obstacle_mask[ni, nj]):
                    visited[ni, nj] = True
                    stack.append((ni, nj))
                    accessible_cells += 1
        
        # Au moins 50% de la grille doit être accessible pour une bonne diffusion
        total_free_cells = (H * W) - obstacle_mask.sum().item()
        connectivity_ratio = accessible_cells / max(total_free_cells, 1)
        
        return connectivity_ratio >= 0.5
    
    
    def generate_environment(self, size: int, source_pos: Tuple[int, int]) -> torch.Tensor:
        """Étape 3: Obstacles multiples pour gestion de la complexité."""
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=DEVICE)
        
        g = torch.Generator(device=DEVICE)
        g.manual_seed(CONFIG.SEED)
        
        n_obstacles = torch.randint(self.MIN_OBSTACLE_SIZE, self.MAX_OBSTACLE_SIZE + 1, (1,), generator=g, device=DEVICE).item()
        
        source_i, source_j = source_pos
        placed_obstacles = []
        
        for obstacle_idx in range(n_obstacles):
            obstacle_size = torch.randint(self.MIN_OBSTACLE_SIZE, self.MAX_OBSTACLE_SIZE + 1, (1,), generator=g, device=DEVICE).item()
            
            max_pos = size - obstacle_size
            if max_pos <= 1:
                continue
            
            for attempt in range(50):
                i = torch.randint(1, max_pos, (1,), generator=g, device=DEVICE).item()
                j = torch.randint(1, max_pos, (1,), generator=g, device=DEVICE).item()
                
                # Vérifications multiples pour étape 3
                valid_position = True
                
                # 1. Pas de chevauchement avec source
                if i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size:
                    valid_position = False
                
                # 2. Pas de chevauchement avec obstacles existants
                for obs_i, obs_j, obs_size in placed_obstacles:
                    if (i < obs_i + obs_size and i + obstacle_size > obs_i and
                            j < obs_j + obs_size and j + obstacle_size > obs_j):
                        valid_position = False
                        break
                
                if valid_position:
                    obstacle_mask[i:i + obstacle_size, j:j + obstacle_size] = True
                    placed_obstacles.append((i, j, obstacle_size))
                    break
        
        # Validation finale de connectivité
        if not self._validate_connectivity(obstacle_mask, source_pos):
            raise Exception("⚠️  Connectivité non garantie - génération d'un environnement plus simple")
            # return self._generate_stage_2_environment(size, source_pos)
        
        return obstacle_mask
