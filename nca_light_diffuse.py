import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

GRID_SIZE = 16
N_STEPS = 30          # nombre d'itérations d'entraînement (ajuste si long)
NCA_STEPS = 20         # horizon multi-step target pendant génération
SOURCE_INTENSITY = 1.0

PREVIS_STEPS = 30      # animation avant entraînement
POSTVIS_STEPS = 50     # animation après entraînement

print(f"Using device: {DEVICE}")

# -----------------------------
# Kernel de diffusion (target)
# -----------------------------
KERNEL = torch.ones((1, 1, 3, 3), device=DEVICE) / 9.0

def simulation_step(grid, source_mask):
    """
    Convolution 3x3 (moyenne) pour faire la diffusion cible.
    """
    x = grid.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    new_grid = F.conv2d(x, KERNEL, padding=1).squeeze(0).squeeze(0)
    new_grid[source_mask] = grid[source_mask]  # source fixe
    return new_grid

# -----------------------------
# Model NCA (MLP appliqué localement)
# -----------------------------
class SimpleNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [B, 10]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # sortie dans [-1,1]

model = SimpleNCA().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -----------------------------
# Génération d'une séquence cible (avec source positionnée aléatoirement)
# -----------------------------
def generate_target_steps(n_steps=NCA_STEPS, size=GRID_SIZE, seed=None):
    """
    Retourne les étapes [t0, t1, ..., t_n_steps] ainsi que la source mask.
    """
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        i0 = torch.randint(2, size-2, (1,), generator=g).item()
        j0 = torch.randint(2, size-2, (1,), generator=g).item()
    else:
        i0 = torch.randint(2, size-2, (1,)).item()
        j0 = torch.randint(2, size-2, (1,)).item()

    grid = torch.zeros((size, size), device=DEVICE)
    grid[i0, j0] = SOURCE_INTENSITY
    source_mask = torch.zeros_like(grid, dtype=torch.bool)
    source_mask[i0, j0] = True

    steps = [grid.clone()]
    for _ in range(n_steps):
        grid = simulation_step(grid, source_mask)
        steps.append(grid.clone())
    return steps, source_mask

# -----------------------------
# NCA update step (boucles)
# -----------------------------
def nca_step(grid, source_mask):
    """
    Applique le modèle localement (boucle sur les pixels intérieurs).
    garde la source fixe.
    """
    new_grid = grid.clone()
    H, W = grid.shape
    for i in range(1, H-1):
        for j in range(1, W-1):
            patch = grid[i-1:i+2, j-1:j+2].reshape(1, -1)  # (1,9)
            patch = torch.cat([patch, source_mask[i,j].float().reshape(1,1)], dim=1)  # (1,10)
            delta = model(patch)  # (1,1)
            new_grid[i, j] = torch.clamp(grid[i, j] + delta.squeeze(), 0.0, 1.0)
    new_grid[source_mask] = grid[source_mask]
    return new_grid

# -----------------------------
# Préparation d'une cible fixe pour visualisation (même source pré/post)
# -----------------------------
vis_seed = 123  # seed pour garder la même source avant/après entraînement
target_steps_vis, source_mask_vis = generate_target_steps(n_steps=50, size=GRID_SIZE, seed=vis_seed)

# -----------------------------
# Affichage avant entraînement (comportement du modèle non entraîné)
# -----------------------------
with torch.no_grad():
    model.eval()
    grid_vis = torch.zeros_like(target_steps_vis[0])
    grid_vis[source_mask_vis] = SOURCE_INTENSITY

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im_nca = axes[0].imshow(grid_vis.cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
    axes[0].set_title("NCA (initial)")
    im_sim = axes[1].imshow(target_steps_vis[-1].cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
    axes[1].set_title("Simulation cible (t final)")
    plt.show()

    for t in range(PREVIS_STEPS):
        grid_vis = nca_step(grid_vis, source_mask_vis)
        im_nca.set_data(grid_vis.cpu().numpy())
        plt.suptitle(f"Avant entraînement — Step {t+1}/{PREVIS_STEPS}")
        plt.pause(0.04)

    plt.pause(0.5)
    plt.close(fig)
    model.train()

# -----------------------------
# Entraînement
# -----------------------------
loss_history = []

print("Training...")
for step in range(N_STEPS):
    optimizer.zero_grad()
    # génère cibles aléatoires pendant le training pour robustesse
    target_steps, source_mask = generate_target_steps(n_steps=NCA_STEPS, size=GRID_SIZE)

    grid_pred = torch.zeros_like(target_steps[0])
    grid_pred[source_mask] = SOURCE_INTENSITY

    loss = torch.tensor(0.0, device=DEVICE)
    for t_step in range(NCA_STEPS):
        target = target_steps[t_step+1]
        new_grid = grid_pred.clone()
        for i in range(1, GRID_SIZE-1):
            for j in range(1, GRID_SIZE-1):
                patch = grid_pred[i-1:i+2, j-1:j+2].reshape(1,-1)
                patch = torch.cat([patch, source_mask[i,j].float().reshape(1,1)], dim=1)
                delta = model(patch)
                new_grid[i,j] = torch.clamp(grid_pred[i,j] + delta.squeeze(), 0.0, 1.0)
        new_grid[source_mask] = grid_pred[source_mask]
        grid_pred = new_grid
        loss = loss + loss_fn(grid_pred, target)

    loss = loss / float(NCA_STEPS)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if step % 10 == 0 or step == N_STEPS-1:
        print(f"Step {step}/{N_STEPS-1}, loss = {loss.item():.6f}")

print("✅ Training done!")

# -----------------------------
# Affichage après entraînement (même cible que pré-visualisation)
# -----------------------------
with torch.no_grad():
    model.eval()
    grid_vis = torch.zeros_like(target_steps_vis[0])
    grid_vis[source_mask_vis] = SOURCE_INTENSITY

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im_nca = axes[0].imshow(grid_vis.cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
    axes[0].set_title("NCA (après entraînement)")
    im_sim = axes[1].imshow(target_steps_vis[-1].cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
    axes[1].set_title("Simulation cible (t final)")
    plt.show()

    for t in range(POSTVIS_STEPS):
        grid_vis = nca_step(grid_vis, source_mask_vis)
        im_nca.set_data(grid_vis.cpu().numpy())
        plt.suptitle(f"Après entraînement — Step {t+1}/{POSTVIS_STEPS}")
        plt.pause(0.03)

    plt.pause(0.5)
    plt.close(fig)
    model.train()

# -----------------------------
# Tracé de la loss
# -----------------------------
plt.figure(figsize=(6,3))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss (moyenne sur horizon)")
plt.title("Historique de la loss")
plt.grid(True)
plt.show()

print("Affichages terminés.")
