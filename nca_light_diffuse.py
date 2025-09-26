import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Tuple, List, Optional
import os

# =============================================================================
# Configuration et initialisation
# =============================================================================

class Config:
    """
    Configuration centralisée pour tous les paramètres du modèle.
    Facilite les expérimentations et la reproductibilité.
    """
    # Paramètres matériels
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42  # Pour la reproductibilité

    # Paramètres de grille
    GRID_SIZE = 16
    SOURCE_INTENSITY = 1.0

    # Paramètres d'entraînement
    N_EPOCHS = 100  # Augmenté pour un meilleur apprentissage
    NCA_STEPS = 20  # Horizon temporel pour l'apprentissage multi-step
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 4  # Entraînement par batch pour plus de stabilité

    # Paramètres de visualisation
    PREVIS_STEPS = 30
    POSTVIS_STEPS = 50
    SAVE_ANIMATIONS = True  # Sauvegarde des animations si pas d'affichage interactif
    OUTPUT_DIR = "nca_outputs"

    # Paramètres du modèle
    HIDDEN_SIZE = 128  # Augmenté pour plus de capacité
    N_LAYERS = 3

# Configuration globale
cfg = Config()

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
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

# Création du dossier de sortie
if cfg.SAVE_ANIMATIONS:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(f"Device: {cfg.DEVICE}")
print(f"Mode interactif: {interactive_mode}")

# =============================================================================
# Simulation physique cible (diffusion de chaleur)
# =============================================================================

class DiffusionSimulator:
    """
    Simulateur de diffusion de chaleur basé sur convolution.
    Représente le processus physique que le NCA doit apprendre à reproduire.
    """
    def __init__(self, device: str = cfg.DEVICE):
        # Kernel de diffusion : moyenne des 8 voisins + centre
        # Simule l'équation de diffusion discrétisée
        self.kernel = torch.ones((1, 1, 3, 3), device=device) / 9.0
        self.device = device

    def step(self, grid: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """
        Un pas de diffusion avec conditions aux bords et sources fixes.

        Args:
            grid: Grille de température [H, W]
            source_mask: Masque des sources de chaleur [H, W]

        Returns:
            Nouvelle grille après diffusion
        """
        # Ajout des dimensions batch et channel pour la convolution
        x = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Convolution avec padding pour conserver la taille
        new_grid = F.conv2d(x, self.kernel, padding=1).squeeze(0).squeeze(0)

        # Les sources restent à intensité constante (condition de Dirichlet)
        new_grid[source_mask] = grid[source_mask]

        return new_grid

    def generate_sequence(self, n_steps: int, size: int, seed: Optional[int] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Génère une séquence complète de diffusion avec source aléatoire.

        Args:
            n_steps: Nombre d'étapes de simulation
            size: Taille de la grille
            seed: Graine pour la reproductibilité

        Returns:
            Liste des états de la grille, masque de la source
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

        # Initialisation de la grille et du masque source
        grid = torch.zeros((size, size), device=self.device)
        grid[i0, j0] = cfg.SOURCE_INTENSITY

        source_mask = torch.zeros_like(grid, dtype=torch.bool)
        source_mask[i0, j0] = True

        # Simulation temporelle
        sequence = [grid.clone()]
        for _ in range(n_steps):
            grid = self.step(grid, source_mask)
            sequence.append(grid.clone())

        return sequence, source_mask

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
    """
    def __init__(self, input_size: int = 10, hidden_size: int = cfg.HIDDEN_SIZE, n_layers: int = cfg.N_LAYERS):
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
    Classe responsable de l'application du modèle NCA sur une grille.
    Sépare la logique de mise à jour de l'architecture du modèle.
    """
    def __init__(self, model: ImprovedNCA, device: str = cfg.DEVICE):
        self.model = model
        self.device = device

    def step(self, grid: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """
        Application du NCA sur toute la grille (version optimisée par batch).

        Args:
            grid: Grille courante [H, W]
            source_mask: Masque des sources [H, W]

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
                # Extraction du patch 3x3 autour de (i,j)
                patch = grid[i-1:i+2, j-1:j+2].reshape(-1)  # 9 éléments
                # Ajout de l'information "est source" comme feature
                is_source = source_mask[i, j].float()
                full_patch = torch.cat([patch, is_source.unsqueeze(0)])  # 10 éléments

                patches.append(full_patch)
                positions.append((i, j))

        if patches:
            # Traitement par batch pour l'efficacité
            patches_tensor = torch.stack(patches)  # [N, 10]
            deltas = self.model(patches_tensor)  # [N, 1]

            # Application des deltas
            for idx, (i, j) in enumerate(positions):
                new_value = grid[i, j] + deltas[idx].squeeze()
                new_grid[i, j] = torch.clamp(new_value, 0.0, 1.0)

        # Les sources restent fixes
        new_grid[source_mask] = grid[source_mask]

        return new_grid

# =============================================================================
# Système d'entraînement
# =============================================================================

class NCATrainer:
    """
    Système d'entraînement pour le NCA avec fonctionnalités avancées.

    Features :
    - Entraînement par batch
    - Gradient clipping automatique
    - Scheduling du learning rate
    - Métriques détaillées
    - Sauvegarde automatique
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

    def train_step(self, target_sequence: List[torch.Tensor], source_mask: torch.Tensor) -> float:
        """
        Un pas d'entraînement sur une séquence cible.

        Args:
            target_sequence: Séquence cible [T+1, H, W]
            source_mask: Masque des sources [H, W]

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
            grid_pred = self.updater.step(grid_pred, source_mask)

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
        Boucle d'entraînement principale avec batch training.
        """
        print("🚀 Début de l'entraînement...")
        self.model.train()

        for epoch in range(cfg.N_EPOCHS):
            epoch_losses = []

            # Entraînement par batch pour plus de stabilité
            for batch_idx in range(cfg.BATCH_SIZE):
                # Génération d'une nouvelle séquence cible aléatoire
                target_seq, source_mask = simulator.generate_sequence(
                    n_steps=cfg.NCA_STEPS,
                    size=cfg.GRID_SIZE
                )

                # Un pas d'entraînement
                loss = self.train_step(target_seq, source_mask)
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
    Système de visualisation avancé avec support pour mode interactif et sauvegarde.
    """
    def __init__(self, interactive: bool = interactive_mode):
        self.interactive = interactive
        self.frame_data = []  # Pour sauvegarder les animations

    def animate_comparison(self, updater: NCAUpdater, target_sequence: List[torch.Tensor],
                          source_mask: torch.Tensor, title_prefix: str, n_steps: int) -> None:
        """
        Animation comparative entre NCA et simulation cible.

        Args:
            updater: Système de mise à jour NCA
            target_sequence: Séquence de référence
            source_mask: Masque des sources
            title_prefix: Préfixe pour le titre
            n_steps: Nombre d'étapes à animer
        """
        # Initialisation de la grille NCA
        grid_nca = torch.zeros_like(target_sequence[0])
        grid_nca[source_mask] = cfg.SOURCE_INTENSITY

        if self.interactive:
            # Mode interactif
            plt.ion()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            im_nca = axes[0].imshow(grid_nca.cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
            axes[0].set_title("NCA")
            axes[0].set_xlabel("Position X")
            axes[0].set_ylabel("Position Y")

            target_final = target_sequence[min(len(target_sequence)-1, n_steps)]
            im_target = axes[1].imshow(target_final.cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
            axes[1].set_title("Cible (état final)")
            axes[1].set_xlabel("Position X")
            axes[1].set_ylabel("Position Y")

            # Ajout de barres de couleur
            fig.colorbar(im_nca, ax=axes[0], label="Intensité")
            fig.colorbar(im_target, ax=axes[1], label="Intensité")

            plt.tight_layout()

            for step in range(n_steps):
                grid_nca = updater.step(grid_nca, source_mask)
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
                grid_nca = updater.step(grid_nca, source_mask)

                # Sauvegarde des données pour chaque frame
                frame_data = {
                    'step': step,
                    'nca_grid': grid_nca.cpu().numpy().copy(),
                    'target_grid': target_final.cpu().numpy().copy(),
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
    Fonction principale orchestrant tout le processus.
    """
    print("=" * 60)
    print("🧠 Neural Cellular Automaton - Diffusion de chaleur")
    print("=" * 60)

    # Création du modèle et des systèmes associés
    model = ImprovedNCA().to(cfg.DEVICE)
    trainer = NCATrainer(model)
    visualizer = NCAVisualizer()
    updater = NCAUpdater(model)

    print(f"📊 Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")

    # Génération d'une séquence fixe pour la visualisation comparative
    vis_seed = 123
    target_sequence_vis, source_mask_vis = simulator.generate_sequence(
        n_steps=max(cfg.PREVIS_STEPS, cfg.POSTVIS_STEPS),
        size=cfg.GRID_SIZE,
        seed=vis_seed
    )

    # Visualisation avant entraînement
    print("\n🎬 Animation pré-entraînement...")
    with torch.no_grad():
        model.eval()
        visualizer.animate_comparison(
            updater, target_sequence_vis, source_mask_vis,
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
            updater, target_sequence_vis, source_mask_vis,
            "Après entraînement", cfg.POSTVIS_STEPS
        )

    # Affichage des métriques
    print("\n📈 Génération des graphiques de métriques...")
    visualizer.plot_training_metrics(trainer)

    # Sauvegarde du modèle
    if cfg.SAVE_ANIMATIONS:
        model_path = f"{cfg.OUTPUT_DIR}/nca_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg,
            'loss_history': trainer.loss_history,
            'lr_history': trainer.lr_history
        }, model_path)
        print(f"💾 Modèle sauvegardé: {model_path}")

    print("\n✨ Processus terminé avec succès!")

if __name__ == "__main__":
    main()
