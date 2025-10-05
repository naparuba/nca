"""
Générateur d'environnements pour la simulation de diffusion thermique.
Classe utilitaire découplée des stages pour générer des environnements de complexité variable.
"""

import torch
from typing import Tuple, Optional, List
from .base_stage import StageEnvironmentValidator


class EnvironmentGenerator:
    """
    Générateur d'environnements découplé des stages.
    Responsable de créer des environnements de simulation pour les différents stages.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def generate_empty_environment(self, size: int) -> torch.Tensor:
        """
        Génère un environnement vide (aucun obstacle).
        
        Args:
            size: Taille de la grille
            
        Returns:
            Masque d'obstacles vide
        """
        return torch.zeros((size, size), dtype=torch.bool, device=self.device)
    
    def generate_single_obstacle_environment(self, size: int, source_pos: Tuple[int, int],
                                           min_obstacle_size: int = 2, max_obstacle_size: int = 4,
                                           seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement avec un seul obstacle rectangulaire.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            min_obstacle_size: Taille minimale d'obstacle
            max_obstacle_size: Taille maximale d'obstacle
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles avec un obstacle unique
            
        Raises:
            RuntimeError: Si impossible de générer un environnement valide
        """
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)

        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None

        # Taille d'obstacle aléatoire
        obstacle_size = torch.randint(
            min_obstacle_size,
            max_obstacle_size + 1,
            (1,),
            generator=g,
            device=self.device
        ).item()

        # Placement en évitant la source et les bords
        max_pos = size - obstacle_size
        if max_pos <= 1:
            raise RuntimeError(f"EnvironmentGenerator: Grille trop petite ({size}x{size}) "
                             f"pour placer un obstacle de taille {obstacle_size}")

        source_i, source_j = source_pos

        # Tentatives de placement d'obstacle
        placed = False
        for attempt in range(100):
            i = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
            j = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()

            # Vérifier non-chevauchement avec source
            if not (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
                obstacle_mask[i:i+obstacle_size, j:j+obstacle_size] = True
                placed = True
                break

        if not placed:
            raise RuntimeError(f"EnvironmentGenerator: Impossible de placer un obstacle "
                             f"sans chevaucher la source (seed: {seed})")

        # Validation de connectivité
        if not StageEnvironmentValidator.validate_connectivity(obstacle_mask, source_pos):
            raise RuntimeError(f"EnvironmentGenerator: Environnement généré sans connectivité "
                             f"suffisante (seed: {seed})")

        return obstacle_mask
    
    def generate_complex_environment(self, size: int, source_pos: Tuple[int, int],
                                    min_obstacles: int = 2, max_obstacles: int = 4,
                                    min_obstacle_size: int = 2, max_obstacle_size: int = 4,
                                    placement_attempts: int = 50,
                                    seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement complexe avec plusieurs obstacles.
        Pour le Stage 3 avec environnements difficiles.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            min_obstacles: Nombre minimum d'obstacles
            max_obstacles: Nombre maximum d'obstacles
            min_obstacle_size: Taille minimale d'obstacle
            max_obstacle_size: Taille maximale d'obstacle
            placement_attempts: Nombre de tentatives de placement par obstacle
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles complexe
            
        Raises:
            RuntimeError: Si impossible de générer un environnement valide
        """
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)

        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None

        # Nombre d'obstacles variable
        n_obstacles = torch.randint(
            min_obstacles,
            max_obstacles + 1,
            (1,),
            generator=g,
            device=self.device
        ).item()

        source_i, source_j = source_pos
        placed_obstacles = []

        # Placement de chaque obstacle avec validation stricte
        for obstacle_idx in range(n_obstacles):
            obstacle_size = torch.randint(
                min_obstacle_size,
                max_obstacle_size + 1,
                (1,),
                generator=g,
                device=self.device
            ).item()

            max_pos = size - obstacle_size
            if max_pos <= 1:
                continue

            # Tentatives de placement avec validation stricte
            placed = False
            for attempt in range(placement_attempts):
                i = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
                j = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()

                if self._is_valid_obstacle_position_strict(i, j, obstacle_size,
                                                        source_pos, placed_obstacles):
                    obstacle_mask[i:i+obstacle_size, j:j+obstacle_size] = True
                    placed_obstacles.append((i, j, obstacle_size))
                    placed = True
                    break
            
            # Si impossible de placer cet obstacle, on continue avec les obstacles déjà placés
            if not placed:
                print(f"⚠️  EnvironmentGenerator: Impossible de placer l'obstacle {obstacle_idx+1}/{n_obstacles}")

        # Validation finale de connectivité - SANS FALLBACK
        if not StageEnvironmentValidator.validate_connectivity(obstacle_mask, source_pos,
                                                             min_connectivity_ratio=0.4):
            raise RuntimeError(f"EnvironmentGenerator: Impossible de générer un environnement complexe "
                             f"avec connectivité suffisante (obstacles placés: {len(placed_obstacles)}/"
                             f"{n_obstacles}, seed: {seed})")

        return obstacle_mask
        
    def generate_variable_intensity_environment(self, size: int, source_pos: Tuple[int, int],
                                              min_obstacles: int = 1, max_obstacles: int = 2,
                                              min_obstacle_size: int = 2, max_obstacle_size: int = 4,
                                              placement_attempts: int = 50,
                                              seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère un environnement modéré pour l'apprentissage des intensités variables.
        Pour le Stage 4 avec intensités variables.
        
        Args:
            size: Taille de la grille
            source_pos: Position de la source (i, j)
            min_obstacles: Nombre minimum d'obstacles (modéré)
            max_obstacles: Nombre maximum d'obstacles (modéré)
            min_obstacle_size: Taille minimale d'obstacle
            max_obstacle_size: Taille maximale d'obstacle
            placement_attempts: Nombre de tentatives de placement par obstacle
            seed: Graine pour la reproductibilité
            
        Returns:
            Masque des obstacles modéré
            
        Raises:
            RuntimeError: Si impossible de générer un environnement valide
        """
        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)

        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None

        # Nombre d'obstacles modéré (focus sur les intensités)
        n_obstacles = torch.randint(
            min_obstacles,
            max_obstacles + 1,
            (1,),
            generator=g,
            device=self.device
        ).item()

        source_i, source_j = source_pos
        placed_obstacles = []

        # Placement d'obstacles avec contraintes simplifiées
        for obstacle_idx in range(n_obstacles):
            obstacle_size = torch.randint(
                min_obstacle_size,
                max_obstacle_size + 1,
                (1,),
                generator=g,
                device=self.device
            ).item()

            max_pos = size - obstacle_size
            if max_pos <= 1:
                continue

            # Placement avec validation modérée (moins stricte)
            placed = False
            for attempt in range(placement_attempts):
                i = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
                j = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()

                if self._is_valid_obstacle_position_moderate(i, j, obstacle_size,
                                                           source_pos, placed_obstacles):
                    obstacle_mask[i:i+obstacle_size, j:j+obstacle_size] = True
                    placed_obstacles.append((i, j, obstacle_size))
                    placed = True
                    break
            
            # Si impossible de placer cet obstacle, on continue
            if not placed:
                print(f"⚠️  EnvironmentGenerator: Impossible de placer l'obstacle {obstacle_idx+1}/{n_obstacles}")

        # Validation finale de connectivité - SANS FALLBACK
        if not StageEnvironmentValidator.validate_connectivity(obstacle_mask, source_pos,
                                                             min_connectivity_ratio=0.6):
            raise RuntimeError(f"EnvironmentGenerator: Impossible de générer un environnement pour intensités "
                             f"variables avec connectivité suffisante (obstacles placés: "
                             f"{len(placed_obstacles)}/{n_obstacles}, seed: {seed})")

        return obstacle_mask
        
    def _is_valid_obstacle_position_strict(self, i: int, j: int, obstacle_size: int,
                                         source_pos: Tuple[int, int],
                                         placed_obstacles: List[Tuple[int, int, int]]) -> bool:
        """
        Validation stricte pour les environnements complexes (Stage 3).
        
        Args:
            i, j: Position proposée
            obstacle_size: Taille de l'obstacle
            source_pos: Position de la source
            placed_obstacles: Obstacles déjà placés
            
        Returns:
            True si la position est valide
        """
        source_i, source_j = source_pos
        
        # 1. Pas de chevauchement avec source
        if (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
            return False
        
        # 2. Pas de chevauchement avec obstacles existants
        for obs_i, obs_j, obs_size in placed_obstacles:
            if (i < obs_i + obs_size and i + obstacle_size > obs_i and
                j < obs_j + obs_size and j + obstacle_size > obs_j):
                return False
        
        # 3. Distance minimale de la source pour éviter l'encerclement
        source_distance = max(abs(i + obstacle_size//2 - source_i),
                            abs(j + obstacle_size//2 - source_j))
        if source_distance < 3:  # Distance minimale stricte
            return False
        
        return True
        
    def _is_valid_obstacle_position_moderate(self, i: int, j: int, obstacle_size: int,
                                           source_pos: Tuple[int, int],
                                           placed_obstacles: List[Tuple[int, int, int]]) -> bool:
        """
        Validation modérée pour les environnements du Stage 4.
        Plus permissif que strict pour se concentrer sur les intensités.
        
        Args:
            i, j: Position proposée
            obstacle_size: Taille de l'obstacle
            source_pos: Position de la source
            placed_obstacles: Obstacles déjà placés
            
        Returns:
            True si la position est valide
        """
        source_i, source_j = source_pos
        
        # 1. Pas de chevauchement avec source
        if (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
            return False
        
        # 2. Pas de chevauchement avec obstacles existants
        for obs_i, obs_j, obs_size in placed_obstacles:
            if (i < obs_i + obs_size and i + obstacle_size > obs_i and
                j < obs_j + obs_size and j + obstacle_size > obs_j):
                return False
        
        return True
