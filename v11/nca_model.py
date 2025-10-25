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
        
        input_size = 27  # 9 (patch 3x3)( temperature ) + 9 (patch 3x3)( obstacles ) + 9 (patch 3x3)( sources )
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
        # Le modèle doit produire 3 deltas : un pour la température, un pour les obstacles, un pour la source
        # Bien que les obstacles doivent rester à 0, c'est au modèle d'apprendre cela via la loss
        # de même pour la source (on force la source à rester constante après l'application du delta)
        layers.append(nn.Linear(self.hidden_size, 3))  # 3 sorties : température + obstacle + sources
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        self.delta_scale = 0.1
    
    
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """Forward pass avec scaling des deltas."""
        delta = self.network(x)
        return delta * self.delta_scale
    
    
    def run_step(self, grid):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Effectue un pas de simulation NCA.
        
        Le modèle reçoit l'information des obstacles et de la température via les patches 3x3.
        Il produit des deltas pour les 3 couches (température + obstacles + sources).
        Le modèle doit apprendre à maintenir les obstacles constants (delta=0 sur la couche obstacles), pareil sur les sources.
        Seules les sources sont forcées pour garantir la conservation de l'intensité.
        
        Args:
            grid: Grille actuelle [3, H, W] - couche 0 = température, couche 1 = obstacles, couche 2 = sources
            source_mask: Masque des sources [H, W]
            
        Returns:
            Nouvelle grille après un pas [3, H, W]
        """
        
        # On récupère la taille de la grille
        NB_LAYERS, H, W = grid.shape  # Exemple: 3, H=16, W=16
        
        # Padding de la grille complète (3 couches)
        # grid.unsqueeze(0) : [3, H, W] → [1, 3, H, W] (batch dimension)
        # F.pad ajoute 1 pixel de chaque côté pour gérer les bords en répliquant les valeurs
        grid_padded = F.pad(grid.unsqueeze(0), (1, 1, 1, 1), mode='replicate')  # [1, 3, H+2, W+2]
        
        # Extraction des patches 3x3 pour chaque couche séparément
        # patches_heat : voisinage 3x3 de la température pour chaque cellule
        patches_heat = F.unfold(grid_padded[:, 0:1, :, :], kernel_size=3, stride=1)  # [1, 9, H*W]
        
        # patches_obstacle : voisinage 3x3 des obstacles pour chaque cellule
        patches_obstacle = F.unfold(grid_padded[:, 1:2, :, :], kernel_size=3, stride=1)  # [1, 9, H*W]
        
        # patches_source : voisinage 3x3 des sources pour chaque cellule
        patches_source = F.unfold(grid_padded[:, 2:3, :, :], kernel_size=3, stride=1)  # [1, 9, H*W]
        
        # Concaténation des patches des 3 couches
        # On obtient pour chaque cellule : 9 valeurs de température + 9 valeurs d'obstacles + 9 valeurs de sources = 27 features
        patches = torch.cat([patches_heat, patches_obstacle, patches_source], dim=1)  # [1, 27, H*W]
        patches = patches.squeeze(0).transpose(0, 1)  # [H*W, 27]
        
        # Application du modèle sur TOUTES les cellules
        # Le modèle produit 3 deltas par cellule : [H*W, 3]
        # - colonne 0 : delta pour la température
        # - colonne 1 : delta pour les obstacles (doit apprendre à produire ~0)
        # - colonne 2 : delta pour les sources (doit apprendre à produire ~0)
        deltas = self(patches)  # [H*W, 3]
        
        # Application des deltas sur chaque couche
        # On clone la grille et on reshape pour faciliter l'addition
        new_grid = grid.clone()  # [3, H, W]
        
        # Application des deltas de température (couche 0)
        # On aplatit, on ajoute le delta, on reshape, puis on assigne
        temp_flat = new_grid[0, :, :].flatten() + deltas[:, 0]  # [H*W]
        new_grid[0, :, :] = temp_flat.reshape(H, W)  # [H, W]
        
        # Application des deltas d'obstacles (couche 1)
        # Le modèle doit apprendre à produire delta=0 ici via la loss de pénalité
        obstacle_flat = new_grid[1, :, :].flatten() + deltas[:, 1]  # [H*W]
        new_grid[1, :, :] = obstacle_flat.reshape(H, W)  # [H, W]
        
        # Application des sources (couche 2)
        # Le modèle doit apprendre à produire delta=0 ici via la loss de pénalité
        obstacle_flat = new_grid[2, :, :].flatten() + deltas[:, 2]  # [H*W]
        new_grid[2, :, :] = obstacle_flat.reshape(H, W)  # [H, W]
        
        # Clamping pour garder les valeurs dans [0, 1]
        new_grid = torch.clamp(new_grid, 0.0, 1.0)
        
        return new_grid
