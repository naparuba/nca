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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

# Constantes physiques
GRID_SIZE = 16
N_STEPS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TYPES DE CELLULES (valeurs entières discrètes)
TYPE_EMPTY = 0   # Vide (rien)
TYPE_GAS = 1     # Gaz léger
TYPE_WATER = 2   # Eau dense

# Indices des canaux
CHANNEL_TYPE = 0      # Canal du type de cellule
CHANNEL_DENSITY = 1   # Canal de la densité


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
        """
        Affiche la grille en ASCII pour debug.
        
        Légende :
        - '.' = VIDE
        - '0-9' = GAZ (chiffre = densité)
        - '#' = EAU
        """
        print(f"\n{'='*40}")
        print(f"{title}")
        print(f"{'='*40}")
        
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
            
            print(f"Ligne {i:2d}: {''.join(row)}")
        
        # Afficher les stats
        nb_empty, nb_gas, nb_water = self._get_stats()
        total_gas_density = self.grid[CHANNEL_DENSITY][self.grid[CHANNEL_TYPE] == TYPE_GAS].sum().item()
        total_water_density = self.grid[CHANNEL_DENSITY][self.grid[CHANNEL_TYPE] == TYPE_WATER].sum().item()
        
        print(f"\nSTATS: VIDE={nb_empty}, GAZ={nb_gas} (densité totale={total_gas_density:.2f}), EAU={nb_water} (densité totale={total_water_density:.2f})")
        print(f"{'='*40}\n")
    
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
        PHASE 3 : DÉBORDEMENT LATÉRAL DE L'EAU
        
        Si une cellule EAU est bloquée en bas, elle peut déborder latéralement.
        """
        new_grid = self.grid.clone()
        
        for i in range(self.grid_size - 1, -1, -1):
            for j in range(self.grid_size):
                current_type = int(new_grid[CHANNEL_TYPE, i, j].item())
                
                # Si c'est de l'EAU
                if current_type == TYPE_WATER:
                    # Vérifier si bloquée en dessous
                    is_blocked = False
                    if i == self.grid_size - 1:
                        is_blocked = True
                    else:
                        below_type = int(new_grid[CHANNEL_TYPE, i + 1, j].item())
                        if below_type == TYPE_WATER:
                            is_blocked = True
                    
                    if is_blocked:
                        current_density = new_grid[CHANNEL_DENSITY, i, j].item()
                        available_slots = []
                        
                        # Chercher les cases adjacentes libres
                        if j > 0:  # Gauche
                            left_type = int(new_grid[CHANNEL_TYPE, i, j - 1].item())
                            if left_type in [TYPE_GAS, TYPE_EMPTY]:
                                available_slots.append((i, j - 1))
                        
                        if j < self.grid_size - 1:  # Droite
                            right_type = int(new_grid[CHANNEL_TYPE, i, j + 1].item())
                            if right_type in [TYPE_GAS, TYPE_EMPTY]:
                                available_slots.append((i, j + 1))
                        
                        # S'il y a des cases disponibles, répartir l'eau
                        if available_slots:
                            # Calculer la nouvelle densité (répartie équitablement)
                            new_density = current_density / (len(available_slots) + 1)
                            
                            # Mettre à jour la densité de la case courante
                            new_grid[CHANNEL_DENSITY, i, j] = new_density
                            
                            # Répartir dans les cases adjacentes
                            for ni, nj in available_slots:
                                new_grid[CHANNEL_TYPE, ni, nj] = TYPE_WATER
                                new_grid[CHANNEL_DENSITY, ni, nj] = new_density
        
        self.grid = new_grid
    
    def _apply_gas_diffusion(self):
        """
        PHASE 4 : DIFFUSION DU GAZ
        
        Le gaz se diffuse dans son voisinage 3x3 VIDE uniquement :
        1. Pour chaque cellule de gaz, on analyse son voisinage 3x3
        2. On compte uniquement les cellules VIDES dans ce voisinage
        3. On répartit la densité uniformément entre la cellule source et ses voisins VIDES
        """
        new_grid = self.grid.clone()
        
        # Phase de diffusion
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
        
        self.grid = new_grid
    
    
    def _apply_water_condensation(self):
        """
        PHASE 5 : CONDENSATION DE L'EAU

        L'eau cherche à se condenser vers le bas :
        1. On cumule les densités totales
        2. On remplit les lignes du bas vers le haut
        3. Si une ligne est bloquée par du gaz, on continue à chercher plus haut
        """
        new_grid = self.grid.clone()
        
        # 1. Calculer la densité totale d'eau et identifier toutes les cases d'eau
        total_water_density = 0.0
        water_positions = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if int(new_grid[CHANNEL_TYPE, i, j].item()) == TYPE_WATER:
                    water_positions.append((i, j))
                    total_water_density += new_grid[CHANNEL_DENSITY, i, j].item()
        
        print(f"DEBUG WATER - Avant condensation: {len(water_positions)} cases eau, densité totale={total_water_density:.2f}")
        
        if not water_positions:
            return
            
        # 2. Vider toutes les cases d'eau
        for i, j in water_positions:
            new_grid[CHANNEL_TYPE, i, j] = TYPE_EMPTY
            new_grid[CHANNEL_DENSITY, i, j] = 0.0
        
        # 3. Calculer le nombre de lignes complètes possibles
        full_lines = int(total_water_density / self.grid_size)
        remaining_density = total_water_density - (full_lines * self.grid_size)
        
        print(f"DEBUG WATER - Lignes complètes possibles: {full_lines}, densité restante={remaining_density:.2f}")
        
        # 4. Remplir depuis le bas, en cherchant des lignes libres
        current_row = self.grid_size - 1
        lines_filled = 0
        density_placed = 0.0
        
        # D'abord les lignes complètes
        while lines_filled < full_lines and current_row >= 0:
            # Vérifier si la ligne est libre (pas de gaz)
            line_has_gas = False
            for j in range(self.grid_size):
                if int(new_grid[CHANNEL_TYPE, current_row, j].item()) == TYPE_GAS:
                    line_has_gas = True
                    break
            
            if not line_has_gas:
                # Remplir toute la ligne avec de l'eau
                for j in range(self.grid_size):
                    new_grid[CHANNEL_TYPE, current_row, j] = TYPE_WATER
                    new_grid[CHANNEL_DENSITY, current_row, j] = 1.0
                    density_placed += 1.0
                lines_filled += 1
            
            current_row -= 1
        
        print(f"DEBUG WATER - Après lignes pleines: {lines_filled} lignes remplies, densité placée={density_placed:.2f}")
        
        # 5. S'il reste des lignes complètes à placer, chercher plus haut
        while lines_filled < full_lines and current_row >= 0:
            if all(int(new_grid[CHANNEL_TYPE, current_row, j].item()) != TYPE_GAS for j in range(self.grid_size)):
                for j in range(self.grid_size):
                    new_grid[CHANNEL_TYPE, current_row, j] = TYPE_WATER
                    new_grid[CHANNEL_DENSITY, current_row, j] = 1.0
                    density_placed += 1.0
                lines_filled += 1
            current_row -= 1
        
        # 6. S'il reste de la densité, chercher la première ligne libre depuis le bas
        if remaining_density > 0:
            current_row = self.grid_size - 1
            while current_row >= 0:
                if all(int(new_grid[CHANNEL_TYPE, current_row, j].item()) != TYPE_GAS for j in range(self.grid_size)):
                    density_per_cell = remaining_density / self.grid_size
                    for j in range(self.grid_size):
                        new_grid[CHANNEL_TYPE, current_row, j] = TYPE_WATER
                        new_grid[CHANNEL_DENSITY, current_row, j] = density_per_cell
                        density_placed += density_per_cell
                    break
                current_row -= 1
        
        # Vérification finale
        final_water_density = 0.0
        final_water_cells = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if int(new_grid[CHANNEL_TYPE, i, j].item()) == TYPE_WATER:
                    final_water_cells += 1
                    final_water_density += new_grid[CHANNEL_DENSITY, i, j].item()
        
        print(f"DEBUG WATER - Final: {final_water_cells} cases eau, densité totale={final_water_density:.2f} (attendu={total_water_density:.2f})")
        
        self.grid = new_grid
    
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
                        visual_grid[i, j] = [0.0, cell_density , cell_density ]
            
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
        
        if gas_diff > 1e-10 or water_diff > 1e-10:
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
