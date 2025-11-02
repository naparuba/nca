"""
Simulation physique de fluides basée sur des ÉTATS DISCRETS avec DENSITÉS.

Système :
- Grille 16x16 en vue de côté avec 2 CANAUX :
  * Canal 0 : TYPE de cellule (0=VIDE, 1=GAZ, 2=EAU)
  * Canal 1 : DENSITÉ/quantité dans la cellule (0.0 à 1.0)
- Gravité : l'EAU tombe, le GAZ monte
- Diffusion du GAZ dans le VIDE

Approche :
1. Types parfaitement discrets (pas de seuils flous)
2. Densités continues pour permettre la diffusion
3. Pas de confusion entre type et valeur
"""

from pathlib import Path
from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

# Constantes physiques
GRID_SIZE = 16
N_STEPS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TYPES DE CELLULES (valeurs entières discrètes)
TYPE_EMPTY = 0  # Vide (rien)
TYPE_GAS = 1  # Gaz léger
TYPE_WATER = 2  # Eau dense

# Indices des canaux
CHANNEL_TYPE = 0  # Canal du type de cellule
CHANNEL_DENSITY = 1  # Canal de la densité


class FluidSimulation:
    """
    Simulateur de fluides avec types discrets et densités continues.
    """
    
    
    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size
        self.device = DEVICE
        
        # Grille à 2 canaux : [type, densité] pour chaque cellule
        # Shape : (2, grid_size, grid_size)
        self.grid = torch.zeros((2, grid_size, grid_size), device=self.device)
        
        # Historique pour visualisation (on stocke tout)
        self.history: List[torch.Tensor] = []
    
    
    def _print_grid_ascii(self, title: str = "Grille"):
        print(f"\n{'=' * 40}")
        print(f"{title}")
        print(f"{'=' * 40}")
        
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                cell_type = int(self.grid[CHANNEL_TYPE, i, j].item())
                
                if cell_type == TYPE_EMPTY:
                    row.append('V')
                elif cell_type == TYPE_GAS:
                    row.append('G')
                else:  # TYPE_WATER
                    row.append('O')
            
            # Calculer la densité moyenne de la ligne
            line_density_avg = self.grid[CHANNEL_DENSITY, i, :].mean().item()
            # Max density on the line:
            line_density_max = self.grid[CHANNEL_DENSITY, i, :].max().item()
            # Minumim density, but not the 0.0 ones:
            non_zero_densities = self.grid[CHANNEL_DENSITY, i, :][self.grid[CHANNEL_DENSITY, i, :] > 0.0].min().item()
            
            print(f"Ligne {i:2d}: {''.join(row)}  | Densité moy: {line_density_avg:.4f}  Max:{line_density_max:.4f}  Min:{non_zero_densities:.4f}")
        
        # Afficher les stats
        nb_empty, nb_gas, nb_water = self._get_stats()
        total_gas_density = self.grid[CHANNEL_DENSITY][self.grid[CHANNEL_TYPE] == TYPE_GAS].sum().item()
        total_water_density = self.grid[CHANNEL_DENSITY][self.grid[CHANNEL_TYPE] == TYPE_WATER].sum().item()
        
        print(
                f"\nSTATS: VIDE={nb_empty}, GAZ={nb_gas} (densité totale={total_gas_density:.2f}), EAU={nb_water} (densité totale={total_water_density:.2f})")
        print(f"{'=' * 40}\n")
    
    
    def initialize_scenario_1(self):
        """
        Scénario 1 : Bloc d'eau en haut, gaz en bas.
        """
        self.grid.fill_(0)
        
        # Bloc d'eau dans la partie supérieure (densité = 1.0)
        self.grid[CHANNEL_TYPE, 2:6, 4:12] = TYPE_WATER
        self.grid[CHANNEL_DENSITY, 2:6, 4:12] = 1.0
        
        # Gaz dans la partie inférieure (densité = 1.0)
        self.grid[CHANNEL_TYPE, 10:14, 2:14] = TYPE_GAS
        self.grid[CHANNEL_DENSITY, 10:14, 2:14] = 1.0
        
        print("🌊 Scénario 1 initialisé : Bloc d'eau en haut, gaz en bas")
    
    
    def _apply_gravity(self):
        """
        PHASE 1 : GRAVITÉ
        
        L'EAU tombe vers le bas si elle est au-dessus de GAZ ou VIDE.
        On swap les types ET les densités.
        """
        new_grid = self.grid.clone()
        
        # Parcours de bas en haut pour que l'eau tombe sans conflit
        for i in range(self.grid_size - 2, -1, -1):
            for j in range(self.grid_size):
                current_type = int(new_grid[CHANNEL_TYPE, i, j].item())
                below_type = int(new_grid[CHANNEL_TYPE, i + 1, j].item())
                
                # L'EAU tombe sur le GAZ ou le VIDE
                if current_type == TYPE_WATER and below_type in [TYPE_GAS, TYPE_EMPTY]:
                    # SWAP des types
                    new_grid[CHANNEL_TYPE, i, j] = below_type
                    new_grid[CHANNEL_TYPE, i + 1, j] = current_type
                    
                    # SWAP des densités
                    temp_density = new_grid[CHANNEL_DENSITY, i, j].clone()
                    new_grid[CHANNEL_DENSITY, i, j] = new_grid[CHANNEL_DENSITY, i + 1, j]
                    new_grid[CHANNEL_DENSITY, i + 1, j] = temp_density
        
        self.grid = new_grid
    
    
    def _apply_buoyancy(self):
        """
        PHASE 2 : FLOTTABILITÉ
        
        Le GAZ monte si il est en-dessous de VIDE.
        """
        new_grid = self.grid.clone()
        
        # Parcours de haut en bas pour que le gaz monte sans conflit
        for i in range(1, self.grid_size):
            for j in range(self.grid_size):
                current_type = int(new_grid[CHANNEL_TYPE, i, j].item())
                above_type = int(new_grid[CHANNEL_TYPE, i - 1, j].item())
                
                # Le GAZ monte dans le VIDE
                if current_type == TYPE_GAS and above_type == TYPE_EMPTY:
                    # SWAP des types
                    new_grid[CHANNEL_TYPE, i, j] = above_type
                    new_grid[CHANNEL_TYPE, i - 1, j] = current_type
                    
                    # SWAP des densités
                    temp_density = new_grid[CHANNEL_DENSITY, i, j].clone()
                    new_grid[CHANNEL_DENSITY, i, j] = new_grid[CHANNEL_DENSITY, i - 1, j]
                    new_grid[CHANNEL_DENSITY, i - 1, j] = temp_density
        
        self.grid = new_grid
    
    
    def _apply_lateral_flow(self):
        """
        PHASE 3 : DÉBORDEMENT ET DÉVERSEMENT DE L'EAU - VERSION CORRIGÉE
        
        L'eau peut s'écouler vers le vide dans plusieurs cas :
        1. Elle tombe dans le vide en dessous (gravité)
        2. Elle se déverse TOUJOURS vers le vide sur les côtés et diagonales
        3. Si elle est complètement bloquée, elle pousse le gaz pour déborder
        """
        new_grid = self.grid.clone()
        
        for i in range(self.grid_size - 1, -1, -1):
            for j in range(self.grid_size):
                current_type = int(new_grid[CHANNEL_TYPE, i, j].item())
                
                # Si c'est de l'EAU
                if current_type != TYPE_WATER:
                    continue
                
                current_density = new_grid[CHANNEL_DENSITY, i, j].item()
                available_slots = []
                
                # 1. Cases vides en dessous (gravité)
                if i < self.grid_size - 1:
                    below_type = int(new_grid[CHANNEL_TYPE, i + 1, j].item())
                    if below_type == TYPE_EMPTY:
                        available_slots.append((i + 1, j))
                
                # 2. TOUJOURS chercher le vide sur les côtés (PAS de condition "if not available_slots")
                # Côtés gauche et droite
                if j > 0 and int(new_grid[CHANNEL_TYPE, i, j - 1].item()) == TYPE_EMPTY:
                    available_slots.append((i, j - 1))
                if j < self.grid_size - 1 and int(new_grid[CHANNEL_TYPE, i, j + 1].item()) == TYPE_EMPTY:
                    available_slots.append((i, j + 1))
                
                # 3. Diagonales vers le bas
                if i < self.grid_size - 1:
                    # En bas à gauche
                    if j > 0 and int(new_grid[CHANNEL_TYPE, i + 1, j - 1].item()) == TYPE_EMPTY:
                        available_slots.append((i + 1, j - 1))
                    # En bas à droite
                    if j < self.grid_size - 1 and int(new_grid[CHANNEL_TYPE, i + 1, j + 1].item()) == TYPE_EMPTY:
                        available_slots.append((i + 1, j + 1))
                
                # 4. Si AUCUNE case vide, essayer de pousser le gaz (débordement forcé)
                if not available_slots:
                    # Vérifier si bloquée en dessous
                    is_blocked = False
                    if i == self.grid_size - 1:
                        is_blocked = True
                    else:
                        below_type = int(new_grid[CHANNEL_TYPE, i + 1, j].item())
                        if below_type == TYPE_WATER:
                            is_blocked = True
                    
                    if is_blocked:
                        # Chercher les cases de GAZ qu'on peut pousser
                        for dj, side_j in [(-1, j - 1), (1, j + 1)]:
                            if 0 <= side_j < self.grid_size:
                                side_type = int(new_grid[CHANNEL_TYPE, i, side_j].item())
                                
                                if side_type == TYPE_GAS:
                                    # Case de GAZ : vérifier si on peut le pousser vers le haut
                                    gas_density = new_grid[CHANNEL_DENSITY, i, side_j].item()
                                    
                                    # Vérifier la case au-dessus
                                    if i > 0:  # Il y a une case au-dessus
                                        above_type = int(new_grid[CHANNEL_TYPE, i - 1, side_j].item())
                                        
                                        if above_type == TYPE_EMPTY:
                                            # Case vide au-dessus : on peut pousser
                                            available_slots.append((i, side_j))
                                        elif above_type == TYPE_GAS:
                                            # Case de gaz au-dessus : vérifier la capacité
                                            above_density = new_grid[CHANNEL_DENSITY, i - 1, side_j].item()
                                            if above_density + gas_density <= 1.0:
                                                # Assez de place pour fusionner les densités
                                                available_slots.append((i, side_j))
                
                # Appliquer le déversement/débordement s'il y a des cases disponibles
                if available_slots:
                    # Calculer la nouvelle densité (répartie équitablement)
                    new_density = current_density / (len(available_slots) + 1)
                    
                    # Mettre à jour la densité de la case courante
                    new_grid[CHANNEL_DENSITY, i, j] = new_density
                    
                    # Répartir dans les cases disponibles
                    for ni, nj in available_slots:
                        slot_type = int(new_grid[CHANNEL_TYPE, ni, nj].item())
                        
                        if slot_type == TYPE_EMPTY:
                            # Case vide : simple déversement
                            new_grid[CHANNEL_TYPE, ni, nj] = TYPE_WATER
                            new_grid[CHANNEL_DENSITY, ni, nj] = new_density
                        
                        elif slot_type == TYPE_GAS:
                            # Case de gaz : pousser le gaz vers le haut
                            gas_density = new_grid[CHANNEL_DENSITY, ni, nj].item()
                            above_type = int(new_grid[CHANNEL_TYPE, ni - 1, nj].item())
                            
                            if above_type == TYPE_EMPTY:
                                # Déplacer le gaz vers le haut
                                new_grid[CHANNEL_TYPE, ni - 1, nj] = TYPE_GAS
                                new_grid[CHANNEL_DENSITY, ni - 1, nj] = gas_density
                            elif above_type == TYPE_GAS:
                                # Fusionner avec le gaz au-dessus
                                new_grid[CHANNEL_DENSITY, ni - 1, nj] += gas_density
                            
                            # Remplacer la case actuelle par de l'eau
                            new_grid[CHANNEL_TYPE, ni, nj] = TYPE_WATER
                            new_grid[CHANNEL_DENSITY, ni, nj] = new_density
        
        self.grid = new_grid
    
    
    def _apply_gas_diffusion(self):
        """
        PHASE 4 : DIFFUSION DU GAZ

        Le gaz se diffuse dans son voisinage 3x3 :
        1. Pour chaque cellule de gaz, on analyse son voisinage 3x3
        2. On compte seulement les cellules VIDES dans ce voisinage
        3. On répartit la densité uniformément entre la cellule source et ses voisins VIDES
        4. Ensuite, on égalise les densités entre cellules de GAZ adjacentes pour un étalement progressif
        """
        new_grid = self.grid.clone()
        
        # Phase 1 : diffusion vers les cellules VIDES (inchangé)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if int(new_grid[CHANNEL_TYPE, i, j].item()) == TYPE_GAS:
                    current_density = new_grid[CHANNEL_DENSITY, i, j].item()
                    if current_density == 0:
                        continue
                    
                    # Analyser le voisinage 3x3
                    empty_neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            # Ne pas considérer la cellule centrale
                            if di == 0 and dj == 0:
                                continue
                            
                            ni, nj = i + di, j + dj
                            # Vérifier les limites de la grille
                            if (0 <= ni < self.grid_size and
                                    0 <= nj < self.grid_size):
                                # Uniquement les cellules VIDES
                                if int(new_grid[CHANNEL_TYPE, ni, nj].item()) == TYPE_EMPTY:
                                    empty_neighbors.append((ni, nj))
                    
                    if empty_neighbors:
                        # Répartir la densité entre la cellule source et les voisins vides
                        total_cells = len(empty_neighbors) + 1  # +1 pour la cellule source
                        density_per_cell = current_density / total_cells
                        
                        # Mettre à jour la cellule source
                        new_grid[CHANNEL_DENSITY, i, j] = density_per_cell
                        
                        # Mettre à jour les voisins vides
                        for ni, nj in empty_neighbors:
                            new_grid[CHANNEL_TYPE, ni, nj] = TYPE_GAS
                            new_grid[CHANNEL_DENSITY, ni, nj] = density_per_cell
        
        # Phase 2 : égalisation des densités entre cellules de GAZ adjacentes
        # On effectue plusieurs passes pour avoir une diffusion progressive
        # Coefficient de diffusion : 0.25 signifie qu'on transfert 25% de la différence
        diffusion_coefficient = 0.25
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if int(new_grid[CHANNEL_TYPE, i, j].item()) == TYPE_GAS:
                    current_density = new_grid[CHANNEL_DENSITY, i, j].item()
                    
                    # Analyser les voisins directs (4-connexité pour la diffusion gaz/gaz)
                    gas_neighbors = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        # Vérifier les limites de la grille
                        if (0 <= ni < self.grid_size and
                                0 <= nj < self.grid_size):
                            # Uniquement les cellules de GAZ
                            if int(new_grid[CHANNEL_TYPE, ni, nj].item()) == TYPE_GAS:
                                neighbor_density = new_grid[CHANNEL_DENSITY, ni, nj].item()
                                gas_neighbors.append((ni, nj, neighbor_density))
                    
                    # Pour chaque voisin de gaz, on égalise progressivement les densités
                    for ni, nj, neighbor_density in gas_neighbors:
                        # Calculer la différence de densité
                        density_diff = current_density - neighbor_density
                        
                        # Transférer une portion de la différence
                        transfer_amount = density_diff * diffusion_coefficient
                        
                        # Mettre à jour les densités
                        new_grid[CHANNEL_DENSITY, i, j] -= transfer_amount
                        new_grid[CHANNEL_DENSITY, ni, nj] += transfer_amount
        
        self.grid = new_grid
    
    
    def _apply_water_condensation(self):
        """
        PHASE 5 : CONDENSATION DE L'EAU (approche locale)

        L'eau cherche à se condenser en poussant le gaz vers le haut :
        1. Pour chaque cellule d'eau, on regarde en dessous et sur les côtés
        2. Si c'est du GAZ, on essaie de le pousser vers le haut
        3. Si le push réussit, on redistribue l'eau dans les cases libérées
        
        Règle de push : on peut pousser le gaz vers le haut SI :
        - La case au-dessus est VIDE ou GAZ
        - Si c'est GAZ, la densité après fusion doit rester < 1.0
        
        PHASE 2 : Concentration vers le bas ET vers les côtés
        4. Pour chaque cellule d'eau, on regarde les 3 cases en dessous (gauche, centre, droite)
           ET les 2 cases sur les côtés (gauche, droite)
        5. Si c'est de l'eau avec densité < 1.0, on transfère la densité
        6. Si on a tout donné, on devient VIDE
        """
        new_grid = self.grid.clone()
        
        # PHASE 1 : Pousser le gaz vers le haut
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # On ne traite que les cellules d'EAU
                if int(new_grid[CHANNEL_TYPE, i, j].item()) != TYPE_WATER:
                    continue
                
                current_density = new_grid[CHANNEL_DENSITY, i, j].item()
                
                # Lister les cases de gaz autour (dessous + côtés)
                gas_neighbors = []
                
                # En dessous
                if i < self.grid_size - 1:
                    if int(new_grid[CHANNEL_TYPE, i + 1, j].item()) == TYPE_GAS:
                        gas_neighbors.append((i + 1, j))
                
                # Côtés
                if j > 0:
                    if int(new_grid[CHANNEL_TYPE, i, j - 1].item()) == TYPE_GAS:
                        gas_neighbors.append((i, j - 1))
                
                if j < self.grid_size - 1:
                    if int(new_grid[CHANNEL_TYPE, i, j + 1].item()) == TYPE_GAS:
                        gas_neighbors.append((i, j + 1))
                
                # Tenter de pousser chaque gaz vers le haut
                freed_slots = []
                
                for gi, gj in gas_neighbors:
                    gas_density = new_grid[CHANNEL_DENSITY, gi, gj].item()
                    
                    # Vérifier si on peut pousser vers le haut
                    if gi > 0:
                        above_type = int(new_grid[CHANNEL_TYPE, gi - 1, gj].item())
                        
                        can_push = False
                        
                        if above_type == TYPE_EMPTY:
                            # Case vide : on peut pousser
                            can_push = True
                        elif above_type == TYPE_GAS:
                            # Case de gaz : vérifier si la fusion ne dépasse pas 1.0
                            above_density = new_grid[CHANNEL_DENSITY, gi - 1, gj].item()
                            if above_density + gas_density < 1.0:
                                can_push = True
                        
                        if can_push:
                            # Pousser le gaz vers le haut
                            if above_type == TYPE_EMPTY:
                                # Déplacer le gaz
                                new_grid[CHANNEL_TYPE, gi - 1, gj] = TYPE_GAS
                                new_grid[CHANNEL_DENSITY, gi - 1, gj] = gas_density
                            elif above_type == TYPE_GAS:
                                # Fusionner avec le gaz au-dessus
                                new_grid[CHANNEL_DENSITY, gi - 1, gj] += gas_density
                            
                            # La case de gaz devient disponible
                            new_grid[CHANNEL_TYPE, gi, gj] = TYPE_EMPTY
                            new_grid[CHANNEL_DENSITY, gi, gj] = 0.0
                            freed_slots.append((gi, gj))
                
                # Si on a libéré des cases, redistribuer l'eau
                if freed_slots:
                    # Calculer la nouvelle densité répartie
                    total_slots = len(freed_slots) + 1  # +1 pour la case d'eau actuelle
                    new_density = current_density / total_slots
                    
                    # Mettre à jour la case d'eau actuelle
                    new_grid[CHANNEL_DENSITY, i, j] = new_density
                    
                    # Redistribuer dans les cases libérées
                    for fi, fj in freed_slots:
                        new_grid[CHANNEL_TYPE, fi, fj] = TYPE_WATER
                        new_grid[CHANNEL_DENSITY, fi, fj] = new_density
        
        # PHASE 2 : Concentration vers le bas ET vers les côtés
        # IMPORTANT: Parcours de HAUT EN BAS pour que les cellules du haut transfèrent vers le bas
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # On ne traite que les cellules d'EAU
                if int(new_grid[CHANNEL_TYPE, i, j].item()) != TYPE_WATER:
                    continue
                
                current_density = new_grid[CHANNEL_DENSITY, i, j].item()
                
                # Si on n'a plus rien, passer
                if current_density <= 0:
                    continue
                
                # Chercher les 3 cases en dessous (gauche, centre, droite)
                water_targets = []
                
                if i < self.grid_size - 1:
                    # En dessous à gauche
                    if j > 0:
                        if int(new_grid[CHANNEL_TYPE, i + 1, j - 1].item()) == TYPE_WATER:
                            water_targets.append((i + 1, j - 1))
                    
                    # En dessous au centre
                    if int(new_grid[CHANNEL_TYPE, i + 1, j].item()) == TYPE_WATER:
                        water_targets.append((i + 1, j))
                    
                    # En dessous à droite
                    if j < self.grid_size - 1:
                        if int(new_grid[CHANNEL_TYPE, i + 1, j + 1].item()) == TYPE_WATER:
                            water_targets.append((i + 1, j + 1))
                
                # Chercher les 2 cases sur les côtés (gauche, droite)
                # Côté gauche
                if j > 0:
                    if int(new_grid[CHANNEL_TYPE, i, j - 1].item()) == TYPE_WATER:
                        water_targets.append((i, j - 1))
                
                # Côté droit
                if j < self.grid_size - 1:
                    if int(new_grid[CHANNEL_TYPE, i, j + 1].item()) == TYPE_WATER:
                        water_targets.append((i, j + 1))
                
                # Transférer la densité aux cases d'eau cibles non saturées
                for wi, wj in water_targets:
                    target_density = new_grid[CHANNEL_DENSITY, wi, wj].item()
                    
                    # Si la case cible n'est pas saturée
                    if target_density < 1.0:
                        # Calculer combien on peut transférer
                        space_available = 1.0 - target_density
                        transfer_amount = min(current_density, space_available)
                        
                        # Transférer
                        new_grid[CHANNEL_DENSITY, i, j] -= transfer_amount
                        new_grid[CHANNEL_DENSITY, wi, wj] += transfer_amount
                        current_density -= transfer_amount
                        
                        # Si on a tout donné, on arrête
                        if current_density <= 0:
                            break
                
                # Si on a tout donné, on devient VIDE
                if new_grid[CHANNEL_DENSITY, i, j].item() <= 0:
                    new_grid[CHANNEL_TYPE, i, j] = TYPE_EMPTY
                    new_grid[CHANNEL_DENSITY, i, j] = 0.0
        
        self.grid = new_grid
    
    
    def _check_water_condensation_debug(self, step_num: int) -> None:
        """
        Fonction de diagnostic pour vérifier que l'eau se condense correctement.
        
        Règle : Si on a au moins 2 lignes COMPLÈTES d'eau en bas (toutes les cellules = eau),
        la dernière ligne devrait avoir une densité de 1.0 (ou proche) après quelques steps.
        """
        # Trouver la dernière ligne qui contient de l'eau
        last_water_line = -1
        for i in range(self.grid_size - 1, -1, -1):
            if (self.grid[CHANNEL_TYPE, i, :] == TYPE_WATER).any():
                last_water_line = i
                break
        
        if last_water_line == -1:
            # Pas d'eau, rien à vérifier
            return
        
        # Compter combien de lignes COMPLÈTES d'eau on a en partant du bas
        complete_water_lines_count = 0
        for i in range(self.grid_size - 1, -1, -1):
            # Vérifier si TOUTES les cellules de cette ligne sont de l'eau
            if (self.grid[CHANNEL_TYPE, i, :] == TYPE_WATER).all():
                complete_water_lines_count += 1
            else:
                break
        
        # Si on a au moins 2 lignes COMPLÈTES d'eau, vérifier la dernière ligne
        if complete_water_lines_count >= 2 and step_num > 5:
            # Calculer la densité moyenne de la dernière ligne complète d'eau
            bottom_line = self.grid_size - 1
            densities = self.grid[CHANNEL_DENSITY, bottom_line, :]
            avg_density = densities.mean().item()
            min_density = densities.min().item()
            max_density = densities.max().item()
            
            print(f"\n⚠️  DEBUG Step {step_num}: Lignes complètes d'eau = {complete_water_lines_count}")
            print(f"    Densité de la ligne {bottom_line} (complète):")
            print(f"    Moyenne: {avg_density:.4f}, Min: {min_density:.4f}, Max: {max_density:.4f}")
            
            # Si la densité moyenne est loin de 1.0, il y a un problème
            if avg_density < 0.95:
                print(f"    ❌ PROBLÈME: La ligne du bas devrait être à densité 1.0 !")
                
                # Afficher les densités de toutes les lignes complètes d'eau
                print(f"\n    Détail des {complete_water_lines_count} lignes COMPLÈTES d'eau:")
                for i in range(self.grid_size - 1, self.grid_size - complete_water_lines_count - 1, -1):
                    densities_line = self.grid[CHANNEL_DENSITY, i, :]
                    avg_line = densities_line.mean().item()
                    print(f"    Ligne {i:2d}: densité moy={avg_line:.4f}")
    
    
    def step(self):
        """
        Un pas de simulation.
        
        Étapes :
        1. Gravité : EAU descend
        2. Flottabilité : GAZ monte
        3. Débordement latéral : EAU s'étale
        4. Diffusion du gaz : GAZ se répartit dans le VIDE
        5. Condensation : l'EAU se condense vers le bas
        
        La conservation de la masse est vérifiée après chaque étape.
        Si une perte de matière est détectée, le programme s'arrête avec une erreur.
        """
        # Vérification initiale
        self._check_mass_conservation()
        
        # 1. Gravité
        self._apply_gravity()
        self._check_mass_conservation("gravité")
        
        # 2. Flottabilité
        self._apply_buoyancy()
        self._check_mass_conservation("flottabilité")
        
        # 3. Débordement latéral
        self._apply_lateral_flow()
        self._check_mass_conservation("débordement latéral")
        
        # 4. Diffusion du gaz
        self._apply_gas_diffusion()
        self._check_mass_conservation("diffusion du gaz")
        
        # 5. Condensation de l'eau
        self._apply_water_condensation()
        self._check_mass_conservation("condensation de l'eau")
    
    
    def _get_stats(self) -> Tuple[int, int, int]:
        """
        Compte le nombre de cellules de chaque type.
        
        Returns:
            (nb_empty, nb_gas, nb_water)
        """
        type_grid = self.grid[CHANNEL_TYPE]
        
        empty_count = (type_grid == TYPE_EMPTY).sum().item()
        gas_count = (type_grid == TYPE_GAS).sum().item()
        water_count = (type_grid == TYPE_WATER).sum().item()
        
        return int(empty_count), int(gas_count), int(water_count)
    
    
    def simulate(self, n_steps: int = N_STEPS, record_every: int = 1):
        """
        Lance la simulation pour n_steps pas de temps.
        
        Args:
            n_steps: Nombre de pas de simulation
            record_every: Enregistrer l'état tous les N pas
        """
        print(f"🚀 Démarrage de la simulation ({n_steps} steps)...")
        
        # Enregistrer l'état initial
        self.history = [self.grid.clone().cpu()]
        
        # Stats initiales
        nb_empty, nb_gas, nb_water = self._get_stats()
        print(f"   État initial : VIDE={nb_empty}, GAZ={nb_gas}, EAU={nb_water}")
        
        # Afficher l'état initial
        self._print_grid_ascii("ÉTAT INITIAL (Step 0)")
        
        for step in range(n_steps):
            self.step()
            
            # DEBUG: Vérifier la condensation de l'eau
            self._check_water_condensation_debug(step + 1)
            
            # Enregistrer l'historique
            if step % record_every == 0:
                self.history.append(self.grid.clone().cpu())
            
            # Affichage de progression
            if step % 40 == 0 or step == n_steps - 1:
                nb_empty, nb_gas, nb_water = self._get_stats()
                print(f"  Step {step:3d}/{n_steps} | VIDE={nb_empty:3d}, GAZ={nb_gas:3d}, EAU={nb_water:3d}")
            
            # Afficher la grille ASCII aux moments clés
            if step < 5 or step in [10, 20, 50, 100, 199]:
                self._print_grid_ascii(f"APRÈS Step {step + 1}")
        
        print(f"✅ Simulation terminée ! {len(self.history)} frames enregistrées")
    
    
    def save_animation(self, output_path: str = "outputs/fluid_simulation.gif", max_frames: int = 50):
        """
        Sauvegarde une animation GIF de la simulation.
        
        Args:
            output_path: Chemin du fichier de sortie
            max_frames: Nombre maximum de frames à inclure dans le GIF
        """
        if not self.history:
            print("⚠️ Aucun historique à visualiser. Lancez d'abord simulate()")
            return
        
        # Limiter le nombre de frames
        step_size = max(1, len(self.history) // max_frames)
        frames_to_plot = self.history[::step_size]
        
        print(f"🎬 Création de l'animation ({len(frames_to_plot)} frames)...")
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        
        def animate(frame_idx):
            ax.clear()
            grid_frame = frames_to_plot[frame_idx].numpy()
            
            # Créer une visualisation combinant type et densité
            # VIDE=blanc, GAZ=jaune (intensité selon densité), EAU=bleu
            visual_grid = np.zeros((self.grid_size, self.grid_size, 3))
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    cell_type = int(grid_frame[CHANNEL_TYPE, i, j])
                    cell_density = grid_frame[CHANNEL_DENSITY, i, j]
                    
                    if cell_type == TYPE_EMPTY:
                        visual_grid[i, j] = [1.0, 1.0, 1.0]  # Blanc
                    elif cell_type == TYPE_GAS:
                        # Jaune avec intensité selon densité
                        visual_grid[i, j] = [1.0, 1.0, 1.0 - cell_density * 0.7]
                    else:  # TYPE_WATER
                        # Bleu
                        visual_grid[i, j] = [0.0, cell_density, cell_density]
            
            im = ax.imshow(visual_grid, origin='upper', interpolation='nearest')
            
            ax.set_title(f'Simulation de Fluides - Frame {frame_idx * step_size}/{len(self.history)}\n'
                         f'Blanc=VIDE | Jaune=GAZ | Bleu=EAU',
                         fontsize=12, fontweight='bold')
            ax.set_xlabel('Position X', fontsize=10)
            ax.set_ylabel('Position Y (gravité ↓)', fontsize=10)
            
            # Grille
            ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
            
            # Compter les types
            type_grid = grid_frame[CHANNEL_TYPE]
            nb_empty = int((type_grid == TYPE_EMPTY).sum())
            nb_gas = int((type_grid == TYPE_GAS).sum())
            nb_water = int((type_grid == TYPE_WATER).sum())
            
            ax.text(0.02, 0.98, f'VIDE: {nb_empty} | GAZ: {nb_gas} | EAU: {nb_water}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            return [im]
        
        
        # Créer l'animation
        anim = animation.FuncAnimation(fig, animate, frames=len(frames_to_plot),
                                       interval=100, blit=False)
        
        # Créer le dossier de sortie
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder
        anim.save(output_path, writer='pillow', fps=10)
        plt.close()
        
        print(f"✅ Animation sauvegardée : {output_path}")
        print(f"   📁 Taille de la grille : {self.grid_size}x{self.grid_size}")
        print(f"   🎞️  Nombre de frames : {len(frames_to_plot)}")
    
    
    def _check_mass_conservation(self, step_name: str = ""):
        """
        Vérifie que la masse totale (gaz + eau) est conservée.
        Si on détecte une perte de matière, on arrête le programme.
        
        Args:
            step_name: Nom de l'étape pour le message d'erreur
        """
        current_gas = self.grid[CHANNEL_DENSITY][self.grid[CHANNEL_TYPE] == TYPE_GAS].sum().item()
        current_water = self.grid[CHANNEL_DENSITY][self.grid[CHANNEL_TYPE] == TYPE_WATER].sum().item()
        
        # On stocke les masses initiales lors de la première vérification
        if not hasattr(self, '_initial_gas'):
            self._initial_gas = current_gas
            self._initial_water = current_water
            return
        
        # Vérification avec une tolérance pour les erreurs de calcul flottant
        gas_diff = abs(current_gas - self._initial_gas)
        water_diff = abs(current_water - self._initial_water)
        
        if gas_diff > 1e-5 or water_diff > 1e-5:
            error_msg = f"""
ERREUR FATALE: Perte de conservation de la matière détectée {f'après {step_name}' if step_name else ''}

GAZ:
- Initial: {self._initial_gas:.10f}
- Actuel:  {current_gas:.10f}
- Diff:    {gas_diff:.10f}

EAU:
- Initial: {self._initial_water:.10f}
- Actuel:  {current_water:.10f}
- Diff:    {water_diff:.10f}
"""
            raise RuntimeError(error_msg)


def main():
    """
    Fonction principale pour tester la simulation.
    """
    print("=" * 60)
    print("🌊 SIMULATION PHYSIQUE DE FLUIDES - TYPES + DENSITÉS")
    print("=" * 60)
    
    # Créer la simulation
    sim = FluidSimulation(grid_size=GRID_SIZE)
    
    # Scénario 1
    sim.initialize_scenario_1()
    
    # Lancer la simulation
    sim.simulate(n_steps=N_STEPS, record_every=4)
    
    # Créer l'animation
    sim.save_animation(output_path="outputs/fluid_simulation.gif", max_frames=50)
    
    print("\n" + "=" * 60)
    print("✅ SIMULATION TERMINÉE AVEC SUCCÈS !")
    print("=" * 60)


if __name__ == "__main__":
    main()
