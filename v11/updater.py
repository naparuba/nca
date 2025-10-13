import torch
from torch.nn import functional as F

from nca_model import ImprovedNCA


class OptimizedNCAUpdater:
    """
    Updater optimisé avec extraction vectorisée des patches.
    """
    
    
    def __init__(self, model: ImprovedNCA):
        self.model = model
    
    
    def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor) -> torch.Tensor:
        """Application optimisée du NCA."""
        H, W = grid.shape
        
        # Extraction vectorisée des patches 3x3
        grid_padded = F.pad(grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        patches = F.unfold(grid_padded, kernel_size=3, stride=1)
        patches = patches.squeeze(0).transpose(0, 1)  # [H*W, 9]
        
        # Features additionnelles
        source_flat = source_mask.flatten().float().unsqueeze(1)  # [H*W, 1]
        obstacle_flat = obstacle_mask.flatten().float().unsqueeze(1)  # [H*W, 1]
        full_patches = torch.cat([patches, source_flat, obstacle_flat], dim=1)  # [H*W, 11]
        
        # Application seulement sur positions valides
        valid_mask = ~obstacle_mask.flatten()
        
        if valid_mask.any():
            valid_patches = full_patches[valid_mask]
            deltas = self.model(valid_patches)
            
            new_grid = grid.clone().flatten()
            new_grid[valid_mask] += deltas.squeeze()
            new_grid = torch.clamp(new_grid, 0.0, 1.0).reshape(H, W)
        else:
            new_grid = grid.clone()
        
        # Contraintes finales
        new_grid[obstacle_mask] = 0.0
        new_grid[source_mask] = grid[source_mask]
        
        return new_grid
