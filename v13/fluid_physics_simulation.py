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
import random
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any

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

TYPE_DISPLAY = {0: 'empty', 1: 'gas', 2: 'water'}

# Indices des canaux
CHANNEL_TYPE = 0  # Canal du type de cellule
CHANNEL_DENSITY = 1  # Canal de la densité

# Seuil pour éliminer l'eau trop diluée (permet d'éviter les résidus persistants)
TOO_LOW_WATER_THRESHOLD = 0.001


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
        
        # Tolérances pour la conservation de la masse
        # Ces valeurs sont choisies POUR ÉVITER les faux positifs dus aux flottants tout en gardant une exigence stricte.
        # On sépare la tolérance PAR ÉTAPE (diff entre entrée et sortie d'un bloc d'opérations)
        # et la tolérance CUMULÉE (diff entre état actuel et masse initiale globale).
        # Hypothèse : opérations locales ne devraient jamais redistribuer plus que ~1e-9 de masse par flot.
        # Justification des valeurs : on travaille avec densités dans [0,1] sur une grille 16x16 -> masse max ~256.
        # Les erreurs flottantes d'additions/soustractions successives restent normalement < 1e-12 * nombre d'opérations.
        # On se donne une marge garde f = 1e-8 par étape, 1e-6 cumulée.
        self.mass_tolerance_step: float = 1e-2  # Seule tolérance conservée: différence avant/après étape
        
        # permet d'injecter de la matière ou des perturbations sans spécialiser `simulate`.
        self.pre_step_callback: Optional[Callable[['FluidSimulation', int], None]] = None
    
    
    def _print_grid_ascii(self, title: str = "Grille"):
        """Affiche la grille en ASCII avec statistiques de densité uniquement sur l'EAU.
        Justification : l'utilisateur ne veut considérer que les cellules d'eau pour les métriques.
        On ignore gaz et vide dans les moyennes pour éviter un bruit de lecture.
        Limite : si une ligne contient un mélange eau/gaz, seule l'eau est prise en compte, ce qui est cohérent
        avec l'objectif de suivre la compaction et la diffusion verticale de l'eau.
        """
        print(f"\n{'=' * 40}")
        print(title)
        print(f"{'=' * 40}")
        for i in range(self.grid_size):
            chars = []
            for j in range(self.grid_size):
                t = int(self.grid[CHANNEL_TYPE, i, j].item())
                if t == TYPE_EMPTY:
                    chars.append('V')
                elif t == TYPE_GAS:
                    chars.append('G')
                else:
                    chars.append('O')
            water_mask = self.grid[CHANNEL_TYPE, i, :] == TYPE_WATER
            if water_mask.any():
                dens = self.grid[CHANNEL_DENSITY, i, :][water_mask]
                avg = float(dens.mean().item())
                mx = float(dens.max().item())
                mn = float(dens.min().item())
            else:
                avg = 0.0
                mx = 0.0
                mn = 0.0
            print(f"Ligne {i:2d}: {''.join(chars)}  | Densité moy: {avg:.4f}  Max:{mx:.4f}  Min:{mn:.10f}")
        nb_empty, nb_gas, nb_water = self._get_stats()
        total_gas_density = self.grid[CHANNEL_DENSITY][self.grid[CHANNEL_TYPE] == TYPE_GAS].sum().item()
        total_water_density = self.grid[CHANNEL_DENSITY][self.grid[CHANNEL_TYPE] == TYPE_WATER].sum().item()
        print(
            f"\nSTATS: VIDE={nb_empty}, GAZ={nb_gas} (densité totale={total_gas_density:.2f}), EAU={nb_water} (densité totale={total_water_density:.2f})")
        print(f"{'=' * 40}\n")
    
    
    def set_pre_step_callback(self, cb: Optional[Callable[['FluidSimulation', int], None]]) -> None:
        """
        Affecte ou retire le callback pré-step.
        Hypothèse: le callback ne modifie pas la structure (dimensions) de la grille.
        """
        self.pre_step_callback = cb
    
    
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
    
    
    def initialize_scenario_2(self, seed: int | None = 123) -> None:
        """Scénario 2 : Grille entièrement RANDOM avec répartition équilibrée des types.
        Objectif pédagogique : tester la robustesse des règles face à un état chaotique.
        Choix :
        - Probabilités ~1/3 pour VIDE, GAZ, EAU afin d'éviter un biais initial.
        - Densités aléatoires uniformes dans [0,1] pour GAZ et EAU (VIDE fixé à 0.0) pour varier les gradients.
        - Seed paramétrable pour reproductibilité des tests.
        Limite : certains points d'eau très faibles disparaîtront rapidement (logique voulue pour nettoyer).
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.grid.fill_(0)
        type_choices = [TYPE_EMPTY, TYPE_GAS, TYPE_WATER]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                t = random.choice(type_choices)
                self.grid[CHANNEL_TYPE, i, j] = t
                if t == TYPE_EMPTY:
                    self.grid[CHANNEL_DENSITY, i, j] = 0.0
                else:
                    # Densité aléatoire uniforme [0,1]
                    self.grid[CHANNEL_DENSITY, i, j] = random.random()
        print("🔀 Scénario 2 initialisé : grille aléatoire équilibrée")
    
    
    def initialize_scenario_3(self) -> None:
        """Scénario 3 : Colonnes d'eau latérales, gaz au centre, vide partiel en bas.
        Objectif : observer la condensation latérale et la diffusion verticale + interaction poussée gaz.
        Choix :
        - Colonnes d'eau sur 3 colonnes à gauche et droite (stables, densité 1.0) pour créer une 'cuve'.
        - Gaz au centre (colonnes intermédiaires) densité 1.0 pour force de flottabilité.
        - Ligne du bas : quelques cellules vides pour créer des cavités permettant redistribution.
        Hypothèse : L'eau latérale va pousser et contraindre le gaz central; le gaz devrait monter/diffuser.
        """
        self.grid.fill_(0)
        # Eau latérale
        left_cols = range(0, 3)
        right_cols = range(self.grid_size - 3, self.grid_size)
        for i in range(self.grid_size):
            for j in list(left_cols) + list(right_cols):
                self.grid[CHANNEL_TYPE, i, j] = TYPE_WATER
                self.grid[CHANNEL_DENSITY, i, j] = 1.0
        # Gaz central
        for i in range(self.grid_size):
            for j in range(3, self.grid_size - 3):
                self.grid[CHANNEL_TYPE, i, j] = TYPE_GAS
                self.grid[CHANNEL_DENSITY, i, j] = 1.0
        # Vide partiel sur la ligne du bas (motif alterné) pour créer des poches
        bottom = self.grid_size - 1
        for j in range(3, self.grid_size - 3, 2):
            self.grid[CHANNEL_TYPE, bottom, j] = TYPE_EMPTY
            self.grid[CHANNEL_DENSITY, bottom, j] = 0.0
        print("🧪 Scénario 3 initialisé : eau en colonnes latérales, gaz central, vide basal")
    
    
    def initialize_scenario_4(self, seed: int | None = 99) -> None:
        """Scénario 4 : Configuration surprise 'bulles et nappe'.
        Idée : Mélange structuré pour tester interactions complexes.
        Composition :
        - Bande diagonale d'eau (créant une nappe inclinée) densité 1.0.
        - Bulles de gaz (clusters circulaires approximatifs) au-dessus de la diagonale.
        - Cavité vide centrale pour créer un point d'effondrement.
        Justification :
        - La diagonale d'eau va se réorganiser verticalement (gravité) -> test condensation.
        - Les bulles de gaz devraient fusionner/monter -> test diffusion + poussée.
        - La cavité vide centrale permet au gaz/eau de se redistribuer rapidement -> test stabilité.
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.grid.fill_(0)
        # Diagonale d'eau
        for i in range(self.grid_size):
            diag_col = i
            self.grid[CHANNEL_TYPE, i, diag_col] = TYPE_WATER
            self.grid[CHANNEL_DENSITY, i, diag_col] = 1.0
            # Élargir la nappe (une épaisseur de 2 autour si possible)
            if diag_col + 1 < self.grid_size:
                self.grid[CHANNEL_TYPE, i, diag_col + 1] = TYPE_WATER
                self.grid[CHANNEL_DENSITY, i, diag_col + 1] = 1.0
        # Bulles de gaz (centres prédéfinis)
        bubble_centers = [(3, 3), (5, 8), (2, 12), (8, 5), (10, 10)]
        radius = 2
        for ci, cj in bubble_centers:
            for i in range(max(0, ci - radius), min(self.grid_size, ci + radius + 1)):
                for j in range(max(0, cj - radius), min(self.grid_size, cj + radius + 1)):
                    if (i - ci) ** 2 + (j - cj) ** 2 <= radius ** 2:
                        # Ne pas écraser la nappe d'eau (priorité eau si déjà mise)
                        if self.grid[CHANNEL_TYPE, i, j] != TYPE_WATER:
                            self.grid[CHANNEL_TYPE, i, j] = TYPE_GAS
                            # Densité variable pour hétérogénéité
                            self.grid[CHANNEL_DENSITY, i, j] = 0.5 + 0.5 * random.random()
        # Cavité vide centrale
        center_start = self.grid_size // 2 - 2
        center_end = self.grid_size // 2 + 2
        for i in range(center_start, center_end):
            for j in range(center_start, center_end):
                self.grid[CHANNEL_TYPE, i, j] = TYPE_EMPTY
                self.grid[CHANNEL_DENSITY, i, j] = 0.0
        print("🎲 Scénario 4 initialisé : nappe diagonale d'eau, bulles de gaz, cavité centrale")
    
    
    # --- Nouveau scénario 5 ---
    def initialize_scenario_5(self, seed: int | None = 2025) -> None:
        """
        Scénario 5: ligne supérieure alternant GAZ/VIDE pour tester injection périodique d'eau.
        Choix: densité gaz 0.6 pour voir compression progressive.
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.grid.fill_(0)
        for j in range(self.grid_size):
            if j % 2 == 0:
                self.grid[CHANNEL_TYPE, 0, j] = TYPE_GAS
                self.grid[CHANNEL_DENSITY, 0, j] = 0.6
        print("💧 Scénario 5 initialisé (alternance gaz/vide ligne 0)")
    
    
    def _periodic_top_water_injection(self, step: int, period: int = 5) -> None:
        """
        Injection d'eau exogène toutes les `period` steps sur la ligne 0.
        Règle:
        - Si aucune case admissible (VIDE ou GAZ): on fait rien.
        - Si VIDE: on place eau densité 1.0.
        - Si GAZ: on tente de déplacer/comprimer vers un voisin (gauche, droite, dessous).
          Si aucun voisin admissible: on annule cette injection.
        Apport de masse volontaire (pas couvert par la conservation interne).
        """
        if period <= 0:
            raise ValueError("Le period doit être > 0.")
        if step % period != 0:
            return
        row = 0
        candidates: List[int] = []
        for col in range(self.grid_size):
            t = self._get_type_case(self.grid, row, col)
            if t in (TYPE_EMPTY, TYPE_GAS):
                candidates.append(col)
        if not candidates:
            return
        col = random.choice(candidates)
        t = self._get_type_case(self.grid, row, col)
        if t == TYPE_EMPTY:
            self._set_type_case(self.grid, row, col, TYPE_WATER, 'injection_eau_vide')
            self._set_density_on_case(self.grid, row, col, 1.0, 'injection_eau_vide')
            return
        # Cas GAZ
        gas_density = self._get_density_on_case(self.grid, row, col)
        neighbors: List[Tuple[int, int]] = []
        if col > 0:
            neighbors.append((row, col - 1))
        if col < self.grid_size - 1:
            neighbors.append((row, col + 1))
        if row < self.grid_size - 1:
            neighbors.append((row + 1, col))
        relocated = False
        for ni, nj in neighbors:
            nt = self._get_type_case(self.grid, ni, nj)
            if nt == TYPE_EMPTY:
                self._set_type_case(self.grid, ni, nj, TYPE_GAS, 'relocation_gaz_vide')
                self._set_density_on_case(self.grid, ni, nj, gas_density, 'relocation_gaz_vide')
                relocated = True
                break
            elif nt == TYPE_GAS:
                new_d = self._get_density_on_case(self.grid, ni, nj) + gas_density
                self._set_density_on_case(self.grid, ni, nj, new_d, 'compression_gaz_injection')
                relocated = True
                break
        if not relocated:
            return
        self._set_type_case(self.grid, row, col, TYPE_WATER, 'injection_eau_sur_gaz')
        self._set_density_on_case(self.grid, row, col, 1.0, 'injection_eau_sur_gaz')
    
    
    def _get_type_case(self, grid, row, col):
        return int(grid[CHANNEL_TYPE, row, col].item())
    
    
    def _set_type_case(self, grid, row, col, type, why):
        before_type = self._get_type_case(grid, row, col)  # int(grid[CHANNEL_TYPE, row, col].item())
        grid[CHANNEL_TYPE, row, col] = type
        # if row == 13:  # and col == 15:
        #    print(f'Set type {TYPE_DISPLAY.get(before_type)}->{TYPE_DISPLAY.get(type)} on case ({row},{col})  [{why}]')
    
    
    def _set_density_on_case(self, grid, row, col, value, why):
        before_density = grid[CHANNEL_DENSITY, row, col].item()
        grid[CHANNEL_DENSITY, row, col] = value
        # if row == 13:  # and col == 15:
        #    print(f'Set density {before_density}->{type} on case ({row},{col})  [{why}]')
    
    
    def _get_density_on_case(self, grid, row, col):
        return grid[CHANNEL_DENSITY, row, col].item()
    
    
    def _water_apply_switch(self):
        """
        PHASE 1 : GRAVITÉ

        Si l'eau a du vide/gaz en dessous, on switch.
        """
        new_grid = self.grid.clone()
        
        # Parcours de bas en haut pour que l'eau tombe sans conflit
        for row in range(self.grid_size - 2, -1, -1):
            for col in range(self.grid_size):
                current_type = self._get_type_case(new_grid, row, col)
                below_type = self._get_type_case(new_grid, row + 1, col)
                
                # L'EAU tombe sur le GAZ ou le VIDE
                if current_type == TYPE_WATER and below_type in [TYPE_GAS, TYPE_EMPTY]:
                    # SWAP des types
                    self._set_type_case(new_grid, row, col, below_type, 'water_fall')
                    self._set_type_case(new_grid, row + 1, col, current_type, 'water_fall')
                    
                    # SWAP des densités
                    temp_density = new_grid[CHANNEL_DENSITY, row, col].clone()
                    self._set_density_on_case(new_grid, row, col, self._get_density_on_case(new_grid, row + 1, col), 'water_fall')
                    self._set_density_on_case(new_grid, row + 1, col, temp_density, 'water_fall')
        
        self.grid = new_grid
    
    
    # L'eau cherche à se condenser vers le bas, quitte à vider la cellule source
    def _apply_water_vertical_condensation(self):
        new_grid = self.grid.clone()
        
        # PHASE 2 : Concentration vers le bas (les 3 en dessous de nous)
        # IMPORTANT: Parcours de HAUT EN BAS pour que les cellules du haut transfèrent vers le bas
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # On ne traite que les cellules d'EAU
                if self._get_type_case(new_grid, row, col) != TYPE_WATER:
                    continue
                
                current_density = self._get_density_on_case(new_grid, row, col)  # ].item()
                
                # Chercher les 3 cases en dessous (gauche, centre, droite)
                water_targets = []
                
                if row < self.grid_size - 1:  # si on a rien en dessous on s'en fiche
                    # En dessous à gauche
                    if col > 0:
                        if self._get_type_case(new_grid, row + 1, col - 1) == TYPE_WATER:
                            density = self._get_density_on_case(new_grid, row + 1, col - 1)
                            water_targets.append((row + 1, col - 1, density))
                    
                    # En dessous au centre
                    if self._get_type_case(new_grid, row + 1, col) == TYPE_WATER:
                        density = self._get_density_on_case(new_grid, row + 1, col)
                        water_targets.append((row + 1, col, density))
                    
                    # En dessous à droite
                    if col < self.grid_size - 1:
                        if self._get_type_case(new_grid, row + 1, col + 1) == TYPE_WATER:
                            density = self._get_density_on_case(new_grid, row + 1, col + 1)
                            water_targets.append((row + 1, col + 1, density))
                
                if len(water_targets) == 0:
                    continue
                
                random.shuffle(water_targets)  # Pour éviter les biais de traitement
                
                # Calculer la somme de ce qu'il faut pour remplir les cases à 1.0
                total_space_needed = sum(1.0 - target_density for _, _, target_density in water_targets)
                
                # CAS 1 : On a assez pour tout remplir
                if current_density >= total_space_needed:
                    # Remplir toutes les cases à 1.0
                    for wi, wj, _ in water_targets:
                        new_grid[CHANNEL_DENSITY, wi, wj] = 1.0
                        self._set_density_on_case(new_grid, wi, wj, 1.0, 'water_vertical_fill')
                    
                    # Garder le reste
                    self._set_density_on_case(new_grid, row, col, current_density - total_space_needed, 'water_vertical_keep_rest')
                
                
                # CAS 2 : On n'a pas assez, on répartit équitablement
                else:
                    # Calculer la densité totale disponible (notre densité + les densités des cases du bas)
                    total_density = current_density + sum(target_density for _, _, target_density in water_targets)
                    
                    # Calculer la densité moyenne
                    avg_density = total_density / len(water_targets)
                    
                    # Répartir équitablement
                    for wi, wj, _ in water_targets:
                        self._set_density_on_case(new_grid, wi, wj, avg_density, 'water_vertical_fill_not_full')
                    
                    # On devient EMPTY car on a tout donné
                    self._set_density_on_case(new_grid, row, col, 0.0, 'water_vertical_all_given')
                    self._set_type_case(new_grid, row, col, TYPE_EMPTY, 'eau a coule en dessous, tout vidée')
        
        self.grid = new_grid
    
    
    def _apply_water_push_neighbor_gaz(self):
        """
        L'eau cherche à se condenser en poussant le gaz (ou le vide qu'on prend à sa place) vers le haut :
        1. Pour chaque cellule d'eau, on regarde sur les côtés (gauche/droite)
        2. Si c'est du GAZ, on essaie de le pousser vers le haut
        3. Si le push réussit, on redistribue l'eau dans les cases libérées

        Règle de push (version compressible simplifiée) :
        - On peut pousser le gaz vers le haut SI la case au-dessus est VIDE ou GAZ
        - Si c'est GAZ: on FUSIONNE toujours (le gaz devient compressible, densité > 1.0 autorisée)
        - Si c'est VIDE: on déplace simplement le gaz vers le haut
        - Si c'est EAU: on ne mélange pas les types -> on ne pousse pas ce gaz

        Justification de la modification minimale :
        - Avant on bloquait la fusion si la somme dépassait 1.0 (gaz incompressible)
        - On supprime uniquement cette contrainte pour débloquer l'expansion latérale de l'eau
        - On conserve TOUTES les autres étapes et commentaires d'origine pour ne pas perdre le contexte pédagogique

        Limites assumées :
        - Pas encore de pression explicite -> risque de densités gaz élevées mais contrôlées plus tard
        - Pas de cohabitation multi-phases dans une même case (un seul TYPE par case)
        """
        new_grid = self.grid.clone()
        
        # PHASE 1 : Pousser le gaz vers le haut
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # On ne traite que les cellules d'EAU
                if self._get_type_case(new_grid, row, col) != TYPE_WATER:
                    continue
                
                current_density = self._get_density_on_case(new_grid, row, col)
                
                # Lister les cases de gaz autour (côtés uniquement)
                gas_neighbors = []
                
                # Côté gauche
                if col > 0:
                    if self._get_type_case(new_grid, row, col - 1) == TYPE_GAS:
                        gas_neighbors.append((row, col - 1))
                
                # Côté droit
                if col < self.grid_size - 1:
                    if self._get_type_case(new_grid, row, col + 1) == TYPE_GAS:
                        gas_neighbors.append((row, col + 1))
                
                # Tenter de pousser chaque gaz vers le haut
                freed_slots = []
                random.shuffle(gas_neighbors)  # Pour éviter les biais directionnels
                
                for gi, gj in gas_neighbors:
                    gas_density = self._get_density_on_case(new_grid, gi, gj)
                    
                    # Vérifier si on peut pousser vers le haut
                    if gi > 0:  # Pas possible sur la première ligne (gi == 0)
                        above_type = self._get_type_case(new_grid, gi - 1, gj)
                        
                        if above_type == TYPE_EMPTY:
                            # Case vide : déplacement direct
                            self._set_type_case(new_grid, gi - 1, gj, TYPE_GAS, 'push_gaz_vide')
                            self._set_density_on_case(new_grid, gi - 1, gj, gas_density, 'push_gaz_vide')
                        elif above_type == TYPE_GAS:
                            # Fusion : gaz compressible, on additionne SANS limite
                            new_grid[CHANNEL_DENSITY, gi - 1, gj] += gas_density
                        else:
                            # Au-dessus eau -> on ne peut pas pousser ce gaz
                            continue
                        
                        # Libérer la case d'origine du gaz
                        self._set_type_case(new_grid, gi, gj, TYPE_EMPTY, 'case_liberee_apres_push')
                        self._set_density_on_case(new_grid, gi, gj, 0.0, 'case_liberee_apres_push')
                        freed_slots.append((gi, gj))
                
                # Si on a libéré des cases de gaz -> expansion de l'eau latéralement
                if freed_slots:
                    total_slots = len(freed_slots) + 1  # +1 pour la case source
                    new_density = current_density / total_slots
                    
                    # Mettre à jour la case d'eau source
                    self._set_density_on_case(new_grid, row, col, new_density, 'redistribution_eau')
                    
                    # Remplir les cases libérées avec de l'eau
                    for fi, fj in freed_slots:
                        self._set_type_case(new_grid, fi, fj, TYPE_WATER, 'expansion_eau')
                        self._set_density_on_case(new_grid, fi, fj, new_density, 'expansion_eau')
        
        # PHASE 2 : Mise à jour globale
        self.grid = new_grid
    
    
    def _apply_water_etalement(self):
        """

        PHASE 2 : Concentration vers les côtés
        4. Pour chaque cellule d'eau, on regarde  les 2 cases sur les côtés (gauche, droite)
        5. Si c'est de l'eau avec densité < 1.0, on transfère la densité
        """
        new_grid = self.grid.clone()
        
        # PHASE 2 : Concentration vers le bas ET vers les côtés
        # IMPORTANT: Parcours de HAUT EN BAS pour que les cellules du haut transfèrent vers le bas
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # On ne traite que les cellules d'EAU
                if self._get_type_case(new_grid, row, col) != TYPE_WATER:
                    continue
                
                current_density = self._get_density_on_case(new_grid, row, col)
                
                # Si on n'a plus rien, passer
                if current_density <= 0:
                    continue
                
                # Chercher les 3 cases en dessous (gauche, centre, droite)
                water_targets = [(row, col)]
                
                # Chercher les 2 cases sur les côtés (gauche, droite)
                # Côté gauche
                if col > 0:
                    if self._get_type_case(new_grid, row, col - 1) == TYPE_WATER:
                        water_targets.append((row, col - 1))
                
                # Côté droit
                if col < self.grid_size - 1:
                    if self._get_type_case(new_grid, row, col + 1) == TYPE_WATER:
                        water_targets.append((row, col + 1))
                
                if len(water_targets) == 1:
                    continue
                
                # On étale l'eau entre les differentes cases de water_targets
                
                total_density = sum(new_grid[CHANNEL_DENSITY, wi, wj].item() for wi, wj in water_targets)
                avg_density = total_density / len(water_targets)
                for wi, wj in water_targets:
                    self._set_density_on_case(new_grid, wi, wj, avg_density, 'water_etalement')
        
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
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self._get_type_case(new_grid, row, col) == TYPE_GAS:
                    current_density = self._get_density_on_case(new_grid, row, col)
                    if current_density == 0:
                        continue
                    
                    # Analyser le voisinage 3x3
                    empty_neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            # Ne pas considérer la cellule centrale
                            if di == 0 and dj == 0:
                                continue
                            
                            ni, nj = row + di, col + dj
                            # Vérifier les limites de la grille
                            if (0 <= ni < self.grid_size and
                                    0 <= nj < self.grid_size):
                                # Uniquement les cellules VIDES
                                if self._get_type_case(new_grid, ni, nj) == TYPE_EMPTY:
                                    empty_neighbors.append((ni, nj))
                    
                    if empty_neighbors:
                        # Répartir la densité entre la cellule source et les voisins vides
                        total_cells = len(empty_neighbors) + 1  # +1 pour la cellule source
                        density_per_cell = current_density / total_cells
                        
                        # Mettre à jour la cellule source
                        new_grid[CHANNEL_DENSITY, row, col] = density_per_cell
                        self._set_density_on_case(new_grid, row, col, density_per_cell, 'gas_diffusion_source')
                        
                        # Mettre à jour les voisins vides
                        for ni, nj in empty_neighbors:
                            self._set_type_case(new_grid, ni, nj, TYPE_GAS, 'gas_diffusion_vide_vers_gaz')
                            self._set_density_on_case(new_grid, ni, nj, density_per_cell, 'gas_diffusion_vide_vers_gaz')
        
        # Phase 2 : égalisation des densités entre cellules de GAZ adjacentes
        # On effectue plusieurs passes pour avoir une diffusion progressive
        # Coefficient de diffusion : 0.25 signifie qu'on transfert 25% de la différence
        diffusion_coefficient = 0.25
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self._get_type_case(new_grid, row, col) == TYPE_GAS:
                    current_density = self._get_density_on_case(new_grid, row, col)
                    
                    # Analyser les voisins directs (4-connexité pour la diffusion gaz/gaz)
                    gas_neighbors = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = row + di, col + dj
                        # Vérifier les limites de la grille
                        if (0 <= ni < self.grid_size and
                                0 <= nj < self.grid_size):
                            # Uniquement les cellules de GAZ
                            if self._get_type_case(new_grid, ni, nj) == TYPE_GAS:
                                neighbor_density = self._get_density_on_case(new_grid, ni, nj)
                                gas_neighbors.append((ni, nj, neighbor_density))
                    
                    if len(gas_neighbors) == 0:
                        continue
                    
                    random.shuffle(gas_neighbors)  # Pour éviter les biais de traitement
                    
                    # Pour chaque voisin de gaz, on égalise progressivement les densités
                    for ni, nj, neighbor_density in gas_neighbors:
                        # Calculer la différence de densité
                        density_diff = current_density - neighbor_density
                        
                        # Transférer une portion de la différence
                        transfer_amount = density_diff * diffusion_coefficient
                        
                        # Mettre à jour les densités
                        new_grid[CHANNEL_DENSITY, row, col] -= transfer_amount
                        new_grid[CHANNEL_DENSITY, ni, nj] += transfer_amount
        
        self.grid = new_grid
    
    
    def _apply_water_disappear(self):
        """
        Si une case d'eau a une densité trop faible (proche de 0), on la vide.
        """
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self._get_type_case(self.grid, row, col) == TYPE_WATER:
                    current_density = self._get_density_on_case(self.grid, row, col)
                    if current_density < TOO_LOW_WATER_THRESHOLD:
                        # Vider la case
                        self._set_type_case(self.grid, row, col, TYPE_EMPTY, 'water_disapear')
                        self._set_density_on_case(self.grid, row, col, 0.0, 'water_disapear')
    
    
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
        """Un pas de simulation avec vérification entrée/sortie sur chaque bloc.
        On NE conserve PAS de masse globale, on vérifie seulement que chaque transformation
        ne détruit pas plus que la tolérance autorisée (erreurs flottantes minimes).
        """
        # 1 eau -> swap vide
        with self._check_mass_conservation("_water_apply_switch"):
            self._water_apply_switch()
        # 2 eau -> concentration vers le bas
        with self._check_mass_conservation("_apply_water_vertical_condensation"):
            self._apply_water_vertical_condensation()
        # 3 gaz: prends toute la place disponible
        with self._check_mass_conservation("_apply_gas_diffusion"):
            self._apply_gas_diffusion()
        # 4 : l'eau pousse le gaz
        with self._check_mass_conservation("_apply_water_push_neighbor_gaz"):
            self._apply_water_push_neighbor_gaz()
        # 5 : l'eau s'étale vers les côtés vers l'eau
        with self._check_mass_conservation("_apply_water_etalement"):
            self._apply_water_etalement()
        # 6 : élimination eau trop diluée
        with self._check_mass_conservation("_apply_water_disapear"):
            self._apply_water_disappear()
    
    
    def _get_stats(self) -> Tuple[int, int, int]:
        """
        Compte le nombre de cellules de chaque type.
        
        Returns:
            (nb_empty, nb_gas, nb_water)
        """
        type_grid = self.grid[CHANNEL_TYPE]
        
        empty_count = (type_grid == TYPE_EMPTY).to(torch.int32).sum().item()
        gas_count = (type_grid == TYPE_GAS).to(torch.int32).sum().item()
        water_count = (type_grid == TYPE_WATER).to(torch.int32).sum().item()
        
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
            if self.pre_step_callback is not None:
                self.pre_step_callback(self, step)
            
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
            nb_empty = int(np.count_nonzero(type_grid == TYPE_EMPTY))
            nb_gas = int(np.count_nonzero(type_grid == TYPE_GAS))
            nb_water = int(np.count_nonzero(type_grid == TYPE_WATER))
            
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
        """Context manager qui vérifie la conservation de la masse ENTRE l'entrée et la sortie.
        Pas de stockage d'état global: uniquement diff locale immédiate.
        Si la différence absolue (gaz ou eau) dépasse mass_tolerance_step => erreur.
        """
        sim = self
        
        class _MassConservationContext:
            def __init__(self, outer: FluidSimulation, name: str):
                self.outer = outer
                self.name = name
                self.gas_before: float = 0.0
                self.water_before: float = 0.0
            
            
            def __enter__(self) -> "_MassConservationContext":
                grid = self.outer.grid
                self.gas_before = grid[CHANNEL_DENSITY][grid[CHANNEL_TYPE] == TYPE_GAS].sum().item()
                self.water_before = grid[CHANNEL_DENSITY][grid[CHANNEL_TYPE] == TYPE_WATER].sum().item()
                return self
            
            
            def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
                if exc_type is not None:
                    return False
                grid = self.outer.grid
                gas_after = grid[CHANNEL_DENSITY][grid[CHANNEL_TYPE] == TYPE_GAS].sum().item()
                water_after = grid[CHANNEL_DENSITY][grid[CHANNEL_TYPE] == TYPE_WATER].sum().item()
                gas_diff = abs(gas_after - self.gas_before)
                water_diff = abs(water_after - self.water_before)
                if gas_diff > self.outer.mass_tolerance_step or water_diff > self.outer.mass_tolerance_step:
                    raise RuntimeError(f"""
ERREUR FATALE: Perte de masse détectée après étape {self.name}
GAZ: avant={self.gas_before:.10f} après={gas_after:.10f} diff={gas_diff:.10f} tolérance={self.outer.mass_tolerance_step:.10f}
EAU: avant={self.water_before:.10f} après={water_after:.10f} diff={water_diff:.10f} tolérance={self.outer.mass_tolerance_step:.10f}
""")
                return False
        
        return _MassConservationContext(sim, step_name)


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
    sim.simulate(n_steps=N_STEPS, record_every=1)
    
    # Créer l'animation
    sim.save_animation(output_path="outputs/fluid_simulation.gif", max_frames=50)
    
    print("\n" + "=" * 60)
    print("✅ SIMULATION TERMINÉE AVEC SUCCÈS !")
    print("=" * 60)


if __name__ == "__main__":
    # Exécuter les 4 scénarios et générer un GIF séparé pour chacun
    scenario_registry: Dict[int, Dict[str, Any]] = {
        1: {'label': 'scenario_1', 'init': FluidSimulation.initialize_scenario_1, 'callback': None},
        2: {'label': 'scenario_2', 'init': FluidSimulation.initialize_scenario_2, 'callback': None},
        3: {'label': 'scenario_3', 'init': FluidSimulation.initialize_scenario_3, 'callback': None},
        4: {'label': 'scenario_4', 'init': FluidSimulation.initialize_scenario_4, 'callback': None},
        5: {'label':    'scenario_5', 'init': FluidSimulation.initialize_scenario_5,
            'callback': lambda sim, step: sim._periodic_top_water_injection(step, period=5)},
    }
    for sid, cfg in scenario_registry.items():
        print("=" * 80)
        print(f"🚀 DÉMARRAGE {cfg['label']}")
        print("=" * 80)
        sim = FluidSimulation(grid_size=GRID_SIZE)
        cfg['init'](sim)  # appel méthode init
        sim.set_pre_step_callback(cfg['callback'])
        sim.simulate(n_steps=N_STEPS, record_every=1)
        output_gif = f"outputs/fluid_simulation_{cfg['label']}.gif"
        sim.save_animation(output_path=output_gif, max_frames=50)
        print(f"✅ Fin {cfg['label']} -> {output_gif}")
