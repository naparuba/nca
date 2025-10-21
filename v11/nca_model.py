import torch
from torch import nn as nn
from torch.nn import functional as F

from config import CONFIG


# =============================================================================
# Modèle NCA: Neural Cellular Automaton optimisé pour l'apprentissage modulaire
# =============================================================================
class NCA(nn.Module):
    
    def __init__(self):
        # type: () -> None
        super().__init__()
        
        input_size = 11  # 9 (patch 3x3) + 1 (source) + 1 (obstacle)
        self.input_size = input_size
        self.hidden_size = CONFIG.HIDDEN_SIZE
        self.n_layers = CONFIG.N_LAYERS
        
        # Architecture profonde avec normalisation
        layers = []
        current_size = input_size
        
        for i in range(self.n_layers):
            layers.append(nn.Linear(current_size, self.hidden_size))
            layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_size = self.hidden_size
        
        # Couche de sortie stabilisée
        layers.append(nn.Linear(self.hidden_size, 1))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.delta_scale = 0.1
    
    
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """Forward pass avec scaling des deltas."""
        delta = self.network(x)
        return delta * self.delta_scale
    
    
    def run_step(self, grid, source_mask, obstacle_mask):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Effectue un pas de simulation NCA.
        
        Le modèle reçoit l'information des obstacles via les features d'entrée (11ème dimension).
        Il doit apprendre à maintenir les obstacles à 0 sans forçage explicite.
        Seules les sources sont forcées pour garantir la conservation de l'intensité.
        
        Args:
            grid: Grille actuelle [H, W]
            source_mask: Masque des sources [H, W]
            obstacle_mask: Masque des obstacles [H, W] (utilisé comme feature, pas comme contrainte hard-coded)
            
        Returns:
            Nouvelle grille après un pas [H, W]
        """
        
        # On récupère la taille de la grille
        H, W = grid.shape  # Exemple: H=16, W=16
        
        # Extraction vectorisée des patches 3x3
        # - grid.unsqueeze(0).unsqueeze(0) : transforme [H, W] en [1, 1, H, W] (format pour unfold)
        # - F.pad(..., (1,1,1,1)) : ajoute 1 pixel de chaque côté pour gérer les bords
        # - mode='replicate' : répète les valeurs des bords
        grid_padded = F.pad(grid.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        
        # F.unfold extrait TOUS les patches 3x3 de la grille
        # Résultat : [1, 9, H*W] - pour chaque position, on a 9 valeurs (le voisinage 3x3)
        patches = F.unfold(grid_padded, kernel_size=3, stride=1)
        
        # - squeeze(0) : enlève la première dimension [9, H*W]
        # - transpose(0, 1) : inverse pour avoir [H*W, 9]
        # Maintenant : 1 ligne par cellule, 9 colonnes pour les 9 voisins
        # C'est ici qu'on a nos VALEURS 0 à 8 (les 9 premières colonnes)
        patches = patches.squeeze(0).transpose(0, 1)  # [H*W, 9]
        
        # Features additionnelles : le modèle reçoit l'information sur les sources et obstacles
        
        # - flatten() : transforme [H, W] en [H*W] (1 dimension)
        # - float() : convertit les booléens en 0.0 ou 1.0
        # - unsqueeze(1) : ajoute une dimension pour avoir [H*W, 1]
        # C'est notre VALEUR 9 (1 colonne avec 0.0 ou 1.0 pour chaque cellule)
        source_flat = source_mask.flatten().float().unsqueeze(1)  # [H*W, 1]
        
        # Même chose pour les obstacles
        # C'est notre VALEUR 10 (1 colonne avec 0.0 ou 1.0 pour chaque cellule)
        obstacle_flat = obstacle_mask.flatten().float().unsqueeze(1)  # [H*W, 1]
        
        # Concaténation des 3 morceaux
        full_patches = torch.cat([patches, source_flat, obstacle_flat], dim=1)  # [H*W, 11]
        # torch.cat concatène horizontalement (dim=1 = colonnes) :
        # - patches [H*W, 9]        → colonnes 0-8
        # - source_flat [H*W, 1]    → colonne 9
        # - obstacle_flat [H*W, 1]  → colonne 10
        # = full_patches [H*W, 11]  → 11 colonnes au total
        
        # Application du modèle sur TOUTES les cellules
        # Le modèle doit apprendre à produire delta=0 pour les obstacles
        deltas = self(full_patches)
        
        # Application des deltas
        new_grid = grid.clone().flatten()
        new_grid += deltas.squeeze()
        new_grid = torch.clamp(new_grid, 0.0, 1.0).reshape(H, W)
        
        # Seules les sources sont forcées (contrainte physique stricte)
        # Les obstacles ne sont PAS forcés ici, le modèle doit apprendre à les respecter
        new_grid[source_mask] = grid[source_mask]
        
        return new_grid
