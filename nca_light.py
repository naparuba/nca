import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# =========================================================
# CONFIGURATION GLOBALE (modifiable)
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID_SIZE = 32         # taille HxW de la grille
N_STEPS = 60           # nombre de steps utilisés pour générer la target (target_history aura N_STEPS+1 états)
SOURCE_INTENSITY = 1.0 # intensité fixe de la source centrale (canal 0)
LEARNING_STEPS = 1000  # nombre d'epochs (itérations d'entraînement)

# ---------------------------
# HYPERPARAMÈTRES DE STABILITÉ
# ---------------------------
# Solution 1 : facteur qui scale l'update (évite oscillations / updates trop brutaux)
# valeurs typiques : 0.05 - 0.3 ; 0.1 est un bon point de départ.
UPDATE_RATE = 0.1

# Solution 3 : randomisation du nombre de steps simulés à chaque epoch.
# Ici on prend des valeurs entre N_STEPS//2 et N_STEPS pour forcer la robustesse.
MIN_TRAIN_STEPS = N_STEPS // 2
MAX_TRAIN_STEPS = N_STEPS

# =========================================================
# UTILITAIRES
# =========================================================
def to_numpy(x):
    """Convertit un tenseur torch sur CPU en numpy pour imshow."""
    return x.detach().cpu().numpy()

# =========================================================
# GÉNÉRATION DE LA CIBLE 'PHYSIQUE' (diffusion simple)
# =========================================================
def generate_diffusion_target(grid_size=GRID_SIZE, n_steps=N_STEPS, source_intensity=SOURCE_INTENSITY):
    """
    Génère un historique de diffusion simple :
    - Source ponctuelle au centre, fixée (n'est pas mise à jour)
    - Règle de diffusion : chaque cellule (non-source) devient la moyenne de son voisinage 3x3
    - Gestion des bords : on calcule la moyenne seulement sur voisins existants (effet semblable à reflect)
    Retour : tensor shape (n_steps+1, 1, H, W)
    """
    grid = torch.zeros((grid_size, grid_size), device=DEVICE)
    i0, j0 = grid_size // 2, grid_size // 2
    grid[i0, j0] = source_intensity

    source_mask = torch.zeros_like(grid, dtype=torch.bool)
    source_mask[i0, j0] = True

    history = [grid.clone()]  # step 0 = init

    for _ in range(n_steps):
        new_grid = grid.clone()
        # on parcourt toutes les cellules ; on utilise les voisins existants (bord géré)
        for i in range(grid_size):
            for j in range(grid_size):
                if source_mask[i, j]:
                    # la source reste fixe
                    continue
                # bornes pour le voisinage (gestion de bord)
                i0n, i1n = max(0, i-1), min(grid_size, i+2)
                j0n, j1n = max(0, j-1), min(grid_size, j+2)
                neighbors = grid[i0n:i1n, j0n:j1n]
                new_grid[i, j] = neighbors.mean()
        grid = new_grid
        history.append(grid.clone())

    # on remet un channel singleton pour être compatible avec la sortie du NCA
    return torch.stack([h.unsqueeze(0) for h in history], dim=0)  # (steps+1, 1, H, W)

# Génération de la target (sur DEVICE)
target_history = generate_diffusion_target()

# =========================================================
# DÉFINITION DU NCA
# =========================================================
class NCA(nn.Module):
    """
    NCA simple :
    - channels : nombre de canaux internes (0 = intensité visible)
    - perception : convolution fixe 3x3 appliquée par canal (group conv)
    - update : petit MLP 1x1 (conv1 -> ReLU -> conv2) appliqué cellule-par-cellule
    Notes :
    - on applique un scaling (UPDATE_RATE) à la correction pour stabiliser la dynamique
    - on n'écrase QUE le canal 0 avec le source_mask (les autres canaux servent de mémoire)
    """
    def __init__(self, channels=16):
        super().__init__()
        self.channels = channels
        # small per-pixel MLP (1x1 convolutions)
        self.conv1 = nn.Conv2d(channels * 2, 128, 1)  # on concatène [x, perception] => 2*channels
        self.conv2 = nn.Conv2d(128, channels, 1)

        # Kernel "laplacien-like" utilisé pour la perception (non entraîné)
        kernel = torch.tensor([[0.05, 0.2, 0.05],
                               [0.2,  -1.0, 0.2],
                               [0.05, 0.2, 0.05]], dtype=torch.float32)
        # enregistrer en buffer (déplacé automatiquement sur GPU si besoin, non-trainable)
        self.register_buffer("perception_kernel", kernel.view(1, 1, 3, 3))

    def perceive(self, x):
        """
        Retourne la concaténation [x, perception(x)].
        - On applique d'abord un padding réfléchi (F.pad(..., mode='reflect')) pour traiter correctement les bords.
        - Puis une convolution groupée (une convolution 3x3 par canal).
        Remarque : F.conv2d ne supporte pas padding_mode arg, donc on pad manuellement.
        """
        # x : (B, C, H, W)
        # padding : (left, right, top, bottom) -> ici = 1 pixel tout autour
        x_padded = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")

        # kernel expand : (channels, 1, 3, 3) -> conv groupée avec groups=channels
        kernel = self.perception_kernel.expand(self.channels, -1, -1, -1)

        # conv groupée : applique le même 3x3 sur chaque canal séparément
        # entrée spatiale est H+2 x W+2, sortie est H x W (car on a paddé)
        y = F.conv2d(x_padded, kernel, bias=None, groups=self.channels)

        # concat : (B, 2*C, H, W)
        return torch.cat([x, y], dim=1)

    def step(self, x, source_mask=None):
        """
        Une étape du NCA :
        - perception
        - MLP 1x1
        - update (résiduel) * UPDATE_RATE
        - ré-imposer la valeur de la source sur le canal 0 uniquement
        """
        y = self.perceive(x)             # (B, 2C, H, W)
        y = F.relu(self.conv1(y))        # (B, 128, H, W)
        y = self.conv2(y)                # (B, C, H, W)

        # Solution 1 : scaling de l'update pour la stabilité temporelle
        x = x + UPDATE_RATE * y

        # On ne veut écraser QUE le canal 0 (intensité), pas les canaux de mémoire.
        if source_mask is not None:
            # source_mask shape attendu : (1, 1, H, W) de type bool
            # on l'applique sur le canal 0 (broadcast batch dimension)
            # ATTENTION : ne pas écraser les autres canaux internes
            x[:, 0:1] = x[:, 0:1] * (~source_mask) + source_mask.float() * SOURCE_INTENSITY

        return x

    def forward(self, x, steps, source_mask=None, return_history=False):
        """
        Applique 'steps' étapes, retourne l'état final ou l'historique si demandé.
        history[t] est un tenseur (B, C, H, W).
        """
        history = [x.clone()]
        for _ in range(steps):
            x = self.step(x, source_mask)
            history.append(x.clone())
        if return_history:
            return history
        return x

# =========================================================
# SETUP : modèle, optim, loss
# =========================================================
nca = NCA(channels=16).to(DEVICE)
optimizer = optim.Adam(nca.parameters(), lr=0.01)
# on utilise une MSE standard pour obtenir un scalaire par étape
loss_fn = nn.MSELoss(reduction="mean")

# source_mask : forme (1,1,H,W) ; utilisé pour fixer canal 0
source_mask = torch.zeros((1,1,GRID_SIZE,GRID_SIZE), dtype=torch.bool, device=DEVICE)
source_mask[0,0,GRID_SIZE//2, GRID_SIZE//2] = True

# =========================================================
# TRAINING LOOP
# - Solution 3 : on randomise le nombre de pas simulés par epoch
# - Solution 2 : loss pondérée (on met plus de poids sur les étapes tardives)
# =========================================================
plt.ion()
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

for epoch in range(LEARNING_STEPS + 1):
    # Initialisation de la grille pour ce rollout (batch size = 1)
    grid = torch.zeros((1, 16, GRID_SIZE, GRID_SIZE), device=DEVICE)
    grid[0, 0, GRID_SIZE // 2, GRID_SIZE // 2] = SOURCE_INTENSITY

    # Solution 3 : nombre de steps aléatoires (renforce robustesse temporelle)
    epoch_nb_steps_simulated = random.randint(MIN_TRAIN_STEPS, MAX_TRAIN_STEPS)

    # Rollout du NCA (on récupère l'historique)
    history = nca(grid, epoch_nb_steps_simulated, source_mask, return_history=True)

    # Solution 2 : pondération de la loss — donner plus d'importance aux steps tardifs
    # On construit des poids linéaires w_t = t+1 (t=0..T) puis on normalise pour garder une échelle stable.
    T = epoch_nb_steps_simulated
    raw_weights = torch.tensor([t + 1 for t in range(T + 1)], dtype=torch.float32, device=DEVICE)  # shape (T+1,)
    sum_w = raw_weights.sum()
    weights = raw_weights / sum_w  # somme des poids = 1

    # Calcul(s) de loss : on compare uniquement le canal 0 (intensité) aux cibles
    # target_history shape : (N_STEPS+1, 1, H, W). Ici, T <= N_STEPS donc index ok.
    total_loss = 0.0
    for t in range(T + 1):
        pred = history[t][:, 0:1]            # (1,1,H,W)
        tgt = target_history[t]              # (1,1,H,W)
        mse_t = loss_fn(pred, tgt)           # scalaire
        total_loss = total_loss + weights[t] * mse_t

    # backward + update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Affichage tous les 20 epochs pour suivre
    if epoch % 20 == 0:
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[0].imshow(to_numpy(grid[0, 0]), cmap="plasma", vmin=0, vmax=1)
        axes[0].set_title("Init")
        axes[1].imshow(to_numpy(target_history[-1, 0]), cmap="plasma", vmin=0, vmax=1)
        axes[1].set_title(f"Target (step {N_STEPS})")
        axes[2].imshow(to_numpy(history[-1][0, 0]), cmap="plasma", vmin=0, vmax=1)
        axes[2].set_title(f"Pred (epoch {epoch})\nLoss={total_loss.item():.6e}")
        plt.pause(0.01)

        # si c'est la dernière epoch => afficher toute la trajectoire du rollout
        if epoch == LEARNING_STEPS:
            plt.figure(figsize=(12, 2))
            for idx, h in enumerate(history):
                plt.subplot(1, len(history), idx + 1)
                plt.imshow(to_numpy(h[0, 0]), cmap="plasma", vmin=0, vmax=1)
                plt.axis("off")
                plt.title(f"s{idx}")
            plt.suptitle(f"Final rollout (epoch {epoch})")

plt.ioff()
plt.show()
