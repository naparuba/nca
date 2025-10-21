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
        
        input_size = 18  # 9 (patch 3x3)( temperature ) + 9 (patch 3x3)( obstacles )
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
        # Le modèle doit produire 2 deltas : un pour la température, un pour les obstacles
        # Bien que les obstacles doivent rester à 0, c'est au modèle d'apprendre cela via la loss
        layers.append(nn.Linear(self.hidden_size, 2))  # 2 sorties : température + obstacle
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.delta_scale = 0.1
    
    
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """Forward pass avec scaling des deltas."""
        delta = self.network(x)
        return delta * self.delta_scale
    
    
    def run_step(self, grid, source_mask):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Effectue un pas de simulation NCA.
        
        Le modèle reçoit l'information des obstacles et de la température via les patches 3x3.
        Il produit des deltas pour les 2 couches (température + obstacles).
        Le modèle doit apprendre à maintenir les obstacles constants (delta=0 sur la couche obstacles).
        Seules les sources sont forcées pour garantir la conservation de l'intensité.
        
        Args:
            grid: Grille actuelle [2, H, W] - couche 0 = température, couche 1 = obstacles
            source_mask: Masque des sources [H, W]
            
        Returns:
            Nouvelle grille après un pas [2, H, W]
        """
        
        # On récupère la taille de la grille
        NB_LAYERS, H, W = grid.shape  # Exemple: 2, H=16, W=16
        
        # Padding de la grille complète (2 couches)
        # grid.unsqueeze(0) : [2, H, W] → [1, 2, H, W] (batch dimension)
        # F.pad ajoute 1 pixel de chaque côté pour gérer les bords en répliquant les valeurs
        grid_padded = F.pad(grid.unsqueeze(0), (1, 1, 1, 1), mode='replicate')  # [1, 2, H+2, W+2]
        
        # Extraction des patches 3x3 pour chaque couche séparément
        # patches_heat : voisinage 3x3 de la température pour chaque cellule
        patches_heat = F.unfold(grid_padded[:, 0:1, :, :], kernel_size=3, stride=1)  # [1, 9, H*W]
        
        # patches_obstacle : voisinage 3x3 des obstacles pour chaque cellule
        patches_obstacle = F.unfold(grid_padded[:, 1:2, :, :], kernel_size=3, stride=1)  # [1, 9, H*W]
        
        # Concaténation des patches des 2 couches
        # On obtient pour chaque cellule : 9 valeurs de température + 9 valeurs d'obstacles = 18 features
        patches = torch.cat([patches_heat, patches_obstacle], dim=1)  # [1, 18, H*W]
        patches = patches.squeeze(0).transpose(0, 1)  # [H*W, 18]
        
        # Application du modèle sur TOUTES les cellules
        # Le modèle produit 2 deltas par cellule : [H*W, 2]
        # - colonne 0 : delta pour la température
        # - colonne 1 : delta pour les obstacles (doit apprendre à produire ~0)
        deltas = self(patches)  # [H*W, 2]
        
        # Application des deltas sur chaque couche
        # On clone la grille et on reshape pour faciliter l'addition
        new_grid = grid.clone()  # [2, H, W]
        
        # Application des deltas de température (couche 0)
        # On aplatit, on ajoute le delta, on reshape, puis on assigne
        temp_flat = new_grid[0, :, :].flatten() + deltas[:, 0]  # [H*W]
        new_grid[0, :, :] = temp_flat.reshape(H, W)  # [H, W]
        
        # Application des deltas d'obstacles (couche 1)
        # Le modèle doit apprendre à produire delta=0 ici via la loss de pénalité
        obstacle_flat = new_grid[1, :, :].flatten() + deltas[:, 1]  # [H*W]
        new_grid[1, :, :] = obstacle_flat.reshape(H, W)  # [H, W]
        
        # Clamping pour garder les valeurs dans [0, 1]
        new_grid = torch.clamp(new_grid, 0.0, 1.0)
        
        # Seules les sources sont forcées (contrainte physique stricte)
        # On force la température des sources à rester constante
        new_grid[0, source_mask] = grid[0, source_mask]
        
        return new_grid
