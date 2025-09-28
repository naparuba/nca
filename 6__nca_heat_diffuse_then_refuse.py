import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Tuple, List, Optional
import os
import argparse

# =============================================================================
# Configuration et initialisation
# =============================================================================

class Config:
    """
    Configuration centralisée pour tous les paramètres du modèle.
    Facilite les expérimentations et la reproductibilité.
    """
    def __init__(self, seed: int = 123):
        # Paramètres matériels
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SEED = seed  # Seed configurable via argument

        # Paramètres de grille
        self.GRID_SIZE = 16
        self.SOURCE_INTENSITY = 1.0

        # Paramètres d'entraînement
        self.N_EPOCHS = 100  # Augmenté pour un meilleur apprentissage
        self.NCA_STEPS = 20  # Horizon temporel pour l'apprentissage multi-step
        self.LEARNING_RATE = 1e-3
        self.BATCH_SIZE = 4  # Entraînement par batch pour plus de stabilité

        # Paramètres de visualisation
        self.PREVIS_STEPS = 30
        self.POSTVIS_STEPS = 50
        self.SAVE_ANIMATIONS = True  # Sauvegarde des animations si pas d'affichage interactif
        self.OUTPUT_DIR = "6__nca_outputs_heat_diffuse_then_refuse"  # Nouveau répertoire pour les résultats heat diffuse then refuse

        # Paramètres du modèle
        self.HIDDEN_SIZE = 128  # Augmenté pour plus de capacité
        self.N_LAYERS = 3

        # Nouveaux paramètres pour les obstacles
        self.MIN_OBSTACLES = 1  # Nombre minimum d'obstacles
        self.MAX_OBSTACLES = 3  # Nombre maximum d'obstacles
        self.MIN_OBSTACLE_SIZE = 2  # Taille minimale d'un obstacle (carré NxN)
        self.MAX_OBSTACLE_SIZE = 4  # Taille maximale d'un obstacle

def parse_arguments():
    """
    Parse les arguments de ligne de commande pour la configuration.

    Returns:
        Namespace avec les arguments parsés
    """
    parser = argparse.ArgumentParser(
        description='Neural Cellular Automaton - Diffusion de chaleur avec obstacles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Graine aléatoire pour la reproductibilité des expériences (entraînement)'
    )

    parser.add_argument(
        '--vis-seed',
        type=int,
        default=3333,
        help='Graine aléatoire pour la séquence de visualisation (indépendante de l\'entraînement)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Nombre d\'époques d\'entraînement'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=16,
        help='Taille de la grille de simulation'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Taille des batches d\'entraînement'
    )

    return parser.parse_args()

# Parse des arguments au début
args = parse_arguments()

# Configuration globale avec seed personnalisable
cfg = Config(seed=args.seed)

# Mise à jour des paramètres depuis les arguments
cfg.N_EPOCHS = args.epochs
cfg.GRID_SIZE = args.grid_size
cfg.BATCH_SIZE = args.batch_size

# Gestion du backend matplotlib pour l'affichage interactif
def setup_matplotlib():
    """
    Configure matplotlib pour l'affichage interactif ou la sauvegarde.
    Détecte automatiquement si l'environnement supporte l'interactivité.
    """
    try:
        # Teste si on peut utiliser un backend interactif
        matplotlib.use('Qt5Agg')
        plt.ion()
        # Test simple pour vérifier que l'affichage fonctionne
        fig, ax = plt.subplots()
        plt.close(fig)
        print("✅ Mode interactif activé")
        return True
    except:
        try:
            matplotlib.use('TkAgg')
            plt.ion()
            fig, ax = plt.subplots()
            plt.close(fig)
            print("✅ Mode interactif activé (TkAgg)")
            return True
        except:
            print("⚠️  Mode non-interactif détecté - les animations seront sauvegardées")
            matplotlib.use('Agg')
            return False

# Initialisation
interactive_mode = setup_matplotlib()
if os.name == 'nt':
    interactive_mode = False  # not manage for windows
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

# Création du dossier de sortie avec seed dans le nom pour différencier
cfg.OUTPUT_DIR = f"6__nca_outputs_heat_diffuse_then_refuse_seed_{cfg.SEED}"
if cfg.SAVE_ANIMATIONS:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(f"Device: {cfg.DEVICE}")
print(f"Seed: {cfg.SEED}")
print(f"Mode interactif: {interactive_mode}")
print(f"Répertoire de sortie: {cfg.OUTPUT_DIR}")

# =============================================================================
# Simulation physique cible (diffusion de chaleur avec obstacles)
# =============================================================================

class DiffusionSimulator:
    """
    Simulateur de diffusion de chaleur basé sur convolution avec obstacles.
    Représente le processus physique que le NCA doit apprendre à reproduire.
    Les obstacles bloquent complètement la diffusion de chaleur.
    """
    def __init__(self, device: str = cfg.DEVICE):
        # Kernel de diffusion : moyenne des 8 voisins + centre
        # Simule l'équation de diffusion de chaleur discrétisée
        self.kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
        self.device = device

    def generate_obstacles(self, size: int, source_pos: Tuple[int, int], seed: Optional[int] = None) -> torch.Tensor:
        """
        Génère des obstacles aléatoirement placés dans la grille.

        Args:
            size: Taille de la grille
            source_pos: Position de la source de chaleur (i, j) à éviter
            seed: Graine pour la reproductibilité

        Returns:
            Masque des obstacles [H, W] (True = obstacle)
        """
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
        else:
            g = None

        obstacle_mask = torch.zeros((size, size), dtype=torch.bool, device=self.device)

        # Nombre d'obstacles aléatoire
        n_obstacles = torch.randint(cfg.MIN_OBSTACLES, cfg.MAX_OBSTACLES + 1, (1,),
                                   generator=g, device=self.device).item()

        for _ in range(n_obstacles):
            # Taille de l'obstacle aléatoire
            obstacle_size = torch.randint(cfg.MIN_OBSTACLE_SIZE, cfg.MAX_OBSTACLE_SIZE + 1, (1,),
                                        generator=g, device=self.device).item()

            # Position aléatoire pour l'obstacle (en évitant les bords)
            max_pos = size - obstacle_size
            if max_pos <= 1:
                continue  # Obstacle trop grand pour la grille

            for attempt in range(50):  # Maximum 50 tentatives pour placer l'obstacle
                i = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()
                j = torch.randint(1, max_pos, (1,), generator=g, device=self.device).item()

                # Vérifier que l'obstacle ne chevauche pas avec la source
                source_i, source_j = source_pos
                if not (i <= source_i < i + obstacle_size and j <= source_j < j + obstacle_size):
                    # Placer l'obstacle
                    obstacle_mask[i:i+obstacle_size, j:j+obstacle_size] = True
                    break

        return obstacle_mask

    def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor) -> torch.Tensor:
        """
        Un pas de diffusion de chaleur avec conditions aux bords, sources fixes et obstacles.

        Args:
            grid: Grille de température [H, W]
            source_mask: Masque des sources de chaleur [H, W]
            obstacle_mask: Masque des obstacles [H, W]

        Returns:
            Nouvelle grille après diffusion de chaleur
        """
        # Ajout des dimensions batch et channel pour la convolution
        x = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Convolution avec padding pour conserver la taille
        new_grid = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)

        # Les obstacles restent à 0 (pas de diffusion de chaleur)
        new_grid[obstacle_mask] = 0.0

        # Les sources de chaleur restent à intensité constante (condition de Dirichlet)
        new_grid[source_mask] = grid[source_mask]

        return new_grid

    def generate_sequence(self, n_steps: int, size: int, seed: Optional[int] = None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Génère une séquence complète de diffusion de chaleur avec source et obstacles aléatoires.

        Args:
            n_steps: Nombre d'étapes de simulation
            size: Taille de la grille
            seed: Graine pour la reproductibilité

        Returns:
            Liste des états de la grille de température, masque de la source de chaleur, masque des obstacles
        """
        # Positionnement aléatoire de la source (évite les bords)
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
            i0 = torch.randint(2, size-2, (1,), generator=g, device=self.device).item()
            j0 = torch.randint(2, size-2, (1,), generator=g, device=self.device).item()
        else:
            i0 = torch.randint(2, size-2, (1,), device=self.device).item()
            j0 = torch.randint(2, size-2, (1,), device=self.device).item()

        # Génération des obstacles en évitant la source
        obstacle_mask = self.generate_obstacles(size, (i0, j0), seed)

        # Initialisation de la grille et du masque source
        grid = torch.zeros((size, size), device=self.device)
        grid[i0, j0] = cfg.SOURCE_INTENSITY

        source_mask = torch.zeros_like(grid, dtype=torch.bool)
        source_mask[i0, j0] = True

        # S'assurer que la source n'est pas dans un obstacle
        if obstacle_mask[i0, j0]:
            obstacle_mask[i0, j0] = False

        # Simulation temporelle
        sequence = [grid.clone()]
        for _ in range(n_steps):
            grid = self.step(grid, source_mask, obstacle_mask)
            sequence.append(grid.clone())

        return sequence, source_mask, obstacle_mask

# Instance globale du simulateur
simulator = DiffusionSimulator()

# =============================================================================
# Modèle Neural Cellular Automaton
# =============================================================================

class ImprovedNCA(nn.Module):
    """
    Neural Cellular Automaton amélioré avec architecture plus robuste.

    Améliorations par rapport à la version originale :
    - Architecture plus profonde et flexible
    - Normalisation par batch pour la stabilité
    - Dropout pour la régularisation
    - Activation plus stable (ReLU au lieu de tanh)
    - Gestion du gradient clipping intégrée
    - Support pour les obstacles
    """
    def __init__(self, input_size: int = 11, hidden_size: int = cfg.HIDDEN_SIZE, n_layers: int = cfg.N_LAYERS):  # 11 au lieu de 10 pour inclure l'info obstacle
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Construction dynamique du réseau
        layers = []
        current_size = input_size

        for i in range(n_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Régularisation légère
            current_size = hidden_size

        # Couche de sortie : produit un delta dans [-0.1, 0.1] pour la stabilité
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

        # Facteur d'échelle pour limiter les changements brutaux
        self.delta_scale = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du réseau.

        Args:
            x: Patch d'entrée [B, input_size]

        Returns:
            Delta à appliquer [B, 1]
        """
        delta = self.network(x)
        return delta * self.delta_scale

class NCAUpdater:
    """
    Classe responsable de l'application du modèle NCA sur une grille avec obstacles.
    Sépare la logique de mise à jour de l'architecture du modèle.
    """
    def __init__(self, model: ImprovedNCA, device: str = cfg.DEVICE):
        self.model = model
        self.device = device

    def step(self, grid: torch.Tensor, source_mask: torch.Tensor, obstacle_mask: torch.Tensor) -> torch.Tensor:
        """
        Application du NCA sur toute la grille avec prise en compte des obstacles.

        Args:
            grid: Grille courante [H, W]
            source_mask: Masque des sources [H, W]
            obstacle_mask: Masque des obstacles [H, W]

        Returns:
            Nouvelle grille après application du NCA
        """
        H, W = grid.shape
        new_grid = grid.clone()

        # Collecte de tous les patches pour traitement par batch
        patches = []
        positions = []

        for i in range(1, H-1):
            for j in range(1, W-1):
                # Skip si c'est un obstacle
                if obstacle_mask[i, j]:
                    continue

                # Extraction du patch 3x3 autour de (i,j)
                patch = grid[i-1:i+2, j-1:j+2].reshape(-1)  # 9 éléments
                # Ajout de l'information "est source" comme feature
                is_source = source_mask[i, j].float()
                # Ajout de l'information "est obstacle" comme feature
                is_obstacle = obstacle_mask[i, j].float()
                full_patch = torch.cat([patch, is_source.unsqueeze(0), is_obstacle.unsqueeze(0)])  # 11 éléments

                patches.append(full_patch)
                positions.append((i, j))

        if patches:
            # Traitement par batch pour l'efficacité
            patches_tensor = torch.stack(patches)  # [N, 11]
            deltas = self.model(patches_tensor)  # [N, 1]

            # Application des deltas
            for idx, (i, j) in enumerate(positions):
                new_value = grid[i, j] + deltas[idx].squeeze()
                new_grid[i, j] = torch.clamp(new_value, 0.0, 1.0)

        # Les obstacles restent à 0
        new_grid[obstacle_mask] = 0.0

        # Les sources restent fixes
        new_grid[source_mask] = grid[source_mask]

        return new_grid

# =============================================================================
# Système d'entraînement
# =============================================================================

class NCATrainer:
    """
    Système d'entraînement pour le NCA avec fonctionnalités avancées et obstacles.

    Features :
    - Entraînement par batch
    - Gradient clipping automatique
    - Scheduling du learning rate
    - Métriques détaillées
    - Sauvegarde automatique
    - Support pour les obstacles
    """
    def __init__(self, model: ImprovedNCA, device: str = cfg.DEVICE):
        self.model = model
        self.device = device
        self.updater = NCAUpdater(model, device)

        # Optimiseur avec weight decay pour la régularisation
        self.optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-4)

        # Scheduler pour réduire le learning rate progressivement
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.N_EPOCHS)

        # Fonction de perte
        self.loss_fn = nn.MSELoss()

        # Historiques
        self.loss_history = []
        self.lr_history = []

    def train_step(self, target_sequence: List[torch.Tensor], source_mask: torch.Tensor, obstacle_mask: torch.Tensor) -> float:
        """
        Un pas d'entraînement sur une séquence cible avec obstacles.

        Args:
            target_sequence: Séquence cible [T+1, H, W]
            source_mask: Masque des sources [H, W]
            obstacle_mask: Masque des obstacles [H, W]

        Returns:
            Perte moyenne sur la séquence
        """
        self.optimizer.zero_grad()

        # Initialisation : grille vide avec sources
        grid_pred = torch.zeros_like(target_sequence[0])
        grid_pred[source_mask] = cfg.SOURCE_INTENSITY

        total_loss = torch.tensor(0.0, device=self.device)

        # Déroulement temporel avec calcul de perte à chaque étape
        for t_step in range(cfg.NCA_STEPS):
            target = target_sequence[t_step + 1]

            # Application du NCA
            grid_pred = self.updater.step(grid_pred, source_mask, obstacle_mask)

            # Accumulation de la perte
            step_loss = self.loss_fn(grid_pred, target)
            total_loss = total_loss + step_loss

        # Perte moyenne sur la séquence
        avg_loss = total_loss / cfg.NCA_STEPS

        # Backpropagation avec gradient clipping
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return avg_loss.item()

    def train(self) -> None:
        """
        Boucle d'entraînement principale avec batch training et obstacles.
        """
        print("🚀 Début de l'entraînement avec obstacles...")
        self.model.train()

        for epoch in range(cfg.N_EPOCHS):
            print(f'Epoch: {epoch}')
            epoch_losses = []

            # Entraînement par batch pour plus de stabilité
            for batch_idx in range(cfg.BATCH_SIZE):
                # Génération d'une nouvelle séquence cible aléatoire avec obstacles
                target_seq, source_mask, obstacle_mask = simulator.generate_sequence(
                    n_steps=cfg.NCA_STEPS,
                    size=cfg.GRID_SIZE
                )

                # Un pas d'entraînement
                loss = self.train_step(target_seq, source_mask, obstacle_mask)
                epoch_losses.append(loss)

            # Statistiques de l'époque
            avg_epoch_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_epoch_loss)
            self.lr_history.append(self.scheduler.get_last_lr()[0])

            # Mise à jour du learning rate
            self.scheduler.step()

            # Affichage périodique
            if epoch % 20 == 0 or epoch == cfg.N_EPOCHS - 1:
                print(f"Epoch {epoch:3d}/{cfg.N_EPOCHS-1} | "
                      f"Loss: {avg_epoch_loss:.6f} | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.2e}")

        print("✅ Entraînement terminé!")

# =============================================================================
# Système de visualisation avancé
# =============================================================================

class NCAVisualizer:
    """
    Système de visualisation avancé avec support pour mode interactif et sauvegarde avec obstacles.
    """
    def __init__(self, interactive: bool = interactive_mode):
        self.interactive = interactive
        self.frame_data = []  # Pour sauvegarder les animations

    def animate_comparison(self, updater: NCAUpdater, target_sequence: List[torch.Tensor],
                          source_mask: torch.Tensor, obstacle_mask: torch.Tensor, title_prefix: str, n_steps: int) -> None:
        """
        Animation comparative entre NCA et simulation cible avec obstacles.

        Args:
            updater: Système de mise à jour NCA
            target_sequence: Séquence de référence
            source_mask: Masque des sources
            obstacle_mask: Masque des obstacles
            title_prefix: Préfixe pour le titre
            n_steps: Nombre d'étapes à animer
        """
        # Initialisation de la grille NCA
        grid_nca = torch.zeros_like(target_sequence[0])
        grid_nca[source_mask] = cfg.SOURCE_INTENSITY

        if self.interactive:
            # Mode interactif
            plt.ion()
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            im_nca = axes[0].imshow(grid_nca.cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
            axes[0].set_title("NCA")
            axes[0].set_xlabel("Position X")
            axes[0].set_ylabel("Position Y")

            target_final = target_sequence[min(len(target_sequence)-1, n_steps)]
            im_target = axes[1].imshow(target_final.cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
            axes[1].set_title("Cible (état final)")
            axes[1].set_xlabel("Position X")
            axes[1].set_ylabel("Position Y")

            # Visualisation des obstacles
            obstacle_vis = obstacle_mask.cpu().numpy().astype(float)
            obstacle_vis[obstacle_vis == 0] = np.nan  # Transparent pour les zones libres
            im_obstacles = axes[2].imshow(obstacle_vis, cmap="Greys", alpha=0.8)
            axes[2].set_title("Obstacles")
            axes[2].set_xlabel("Position X")
            axes[2].set_ylabel("Position Y")

            # Ajout de barres de couleur
            fig.colorbar(im_nca, ax=axes[0], label="Intensité")
            fig.colorbar(im_target, ax=axes[1], label="Intensité")

            plt.tight_layout()

            for step in range(n_steps):
                grid_nca = updater.step(grid_nca, source_mask, obstacle_mask)
                im_nca.set_data(grid_nca.cpu().numpy())
                plt.suptitle(f"{title_prefix} — Étape {step+1}/{n_steps}")
                if self.interactive:
                    plt.draw()
                    plt.pause(0.05)

            if self.interactive:
                plt.pause(1.0)
            plt.close(fig)
            plt.ioff()

        else:
            # Mode sauvegarde
            frames = []
            target_final = target_sequence[min(len(target_sequence)-1, n_steps)]

            for step in range(n_steps):
                grid_nca = updater.step(grid_nca, source_mask, obstacle_mask)

                # Sauvegarde des données pour chaque frame
                frame_data = {
                    'step': step,
                    'nca_grid': grid_nca.cpu().numpy().copy(),
                    'target_grid': target_final.cpu().numpy().copy(),
                    'obstacle_mask': obstacle_mask.cpu().numpy().copy(),
                    'source_mask': source_mask.cpu().numpy().copy(),
                    'title': f"{title_prefix} — Étape {step+1}/{n_steps}"
                }
                frames.append(frame_data)

            # Sauvegarde des frames
            filename = f"{cfg.OUTPUT_DIR}/animation_{title_prefix.lower().replace(' ', '_')}.npy"
            np.save(filename, frames)
            print(f"💾 Animation sauvegardée: {filename}")

    def plot_training_metrics(self, trainer: NCATrainer) -> None:
        """
        Graphiques des métriques d'entraînement.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Courbe de perte
        ax1.plot(trainer.loss_history, linewidth=2)
        ax1.set_xlabel("Époque")
        ax1.set_ylabel("Perte MSE")
        ax1.set_title("Évolution de la perte")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Courbe du learning rate
        ax2.plot(trainer.lr_history, linewidth=2, color='orange')
        ax2.set_xlabel("Époque")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Évolution du Learning Rate")
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        plt.tight_layout()

        if self.interactive:
            plt.show()
        else:
            plt.savefig(f"{cfg.OUTPUT_DIR}/training_metrics.png", dpi=150, bbox_inches='tight')
            print(f"💾 Métriques sauvegardées: {cfg.OUTPUT_DIR}/training_metrics.png")

        plt.close(fig)

# =============================================================================
# Exécution principale
# =============================================================================

def main():
    """
    Fonction principale orchestrant tout le processus avec obstacles.
    """
    print("=" * 60)
    print("🧠 Neural Cellular Automaton - Diffusion avec obstacles")
    print("=" * 60)

    # Création du modèle et des systèmes associés
    model = ImprovedNCA().to(cfg.DEVICE)
    trainer = NCATrainer(model)
    visualizer = NCAVisualizer()
    updater = NCAUpdater(model)

    print(f"📊 Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")

    # Génération d'une séquence fixe pour la visualisation comparative
    # Utilise la vis_seed fournie en argument pour contrôler la configuration de test
    vis_seed = args.vis_seed
    target_sequence_vis, source_mask_vis, obstacle_mask_vis = simulator.generate_sequence(
        n_steps=max(cfg.PREVIS_STEPS, cfg.POSTVIS_STEPS),
        size=cfg.GRID_SIZE,
        seed=vis_seed
    )

    print(f"🎯 Seed de visualisation: {vis_seed}")
    print(f"🧱 Obstacles générés: {obstacle_mask_vis.sum().item()} cellules")

    # Visualisation avant entraînement
    print("\n🎬 Animation pré-entraînement...")
    with torch.no_grad():
        model.eval()
        visualizer.animate_comparison(
            updater, target_sequence_vis, source_mask_vis, obstacle_mask_vis,
            "Avant entraînement", cfg.PREVIS_STEPS
        )

    # Entraînement
    print("\n🎯 Phase d'entraînement...")
    trainer.train()

    # Visualisation après entraînement
    print("\n🎬 Animation post-entraînement...")
    with torch.no_grad():
        model.eval()
        visualizer.animate_comparison(
            updater, target_sequence_vis, source_mask_vis, obstacle_mask_vis,
            "Après entraînement", cfg.POSTVIS_STEPS
        )

    # Affichage des métriques
    print("\n📈 Génération des graphiques de métriques...")
    visualizer.plot_training_metrics(trainer)

    # Sauvegarde du modèle avec informations sur les seeds
    if cfg.SAVE_ANIMATIONS:
        model_path = f"{cfg.OUTPUT_DIR}/nca_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg,
            'training_seed': cfg.SEED,
            'visualization_seed': vis_seed,
            'loss_history': trainer.loss_history,
            'lr_history': trainer.lr_history
        }, model_path)
        print(f"💾 Modèle sauvegardé: {model_path}")

    print("\n✨ Processus terminé avec succès!")
    print(f"📝 Résumé des seeds utilisées:")
    print(f"   - Entraînement: {cfg.SEED}")
    print(f"   - Visualisation: {vis_seed}")

if __name__ == "__main__":
    main()
