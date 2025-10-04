import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import csv

# =========================================================
# CONFIGURATION GLOBALE
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRID_SIZE = 32
N_STEPS = 30
SOURCE_INTENSITY = 1.0
LEARNING_STEPS = 1000
UPDATE_RATE = 0.1
MIN_TRAIN_STEPS = N_STEPS // 2
MAX_TRAIN_STEPS = N_STEPS

# CSV output file
CSV_FILE = "nca_stability_metrics.csv"

# =========================================================
# UTILITAIRES
# =========================================================
def to_numpy(x):
    return x.detach().cpu().numpy()

# =========================================================
# GÉNÉRATION DE LA TARGET
# =========================================================
def generate_diffusion_target(grid_size=GRID_SIZE, n_steps=N_STEPS, source_intensity=SOURCE_INTENSITY):
    grid = torch.zeros((grid_size, grid_size), device=DEVICE)
    i0, j0 = grid_size // 2, grid_size // 2
    grid[i0, j0] = source_intensity
    source_mask = torch.zeros_like(grid, dtype=torch.bool)
    source_mask[i0, j0] = True
    history = [grid.clone()]
    for _ in range(n_steps):
        new_grid = grid.clone()
        for i in range(grid_size):
            for j in range(grid_size):
                if source_mask[i, j]:
                    continue
                i0n, i1n = max(0, i-1), min(grid_size, i+2)
                j0n, j1n = max(0, j-1), min(grid_size, j+2)
                neighbors = grid[i0n:i1n, j0n:j1n]
                new_grid[i, j] = neighbors.mean()
        grid = new_grid
        history.append(grid.clone())
    return torch.stack([h.unsqueeze(0) for h in history], dim=0)

target_history = generate_diffusion_target()

# =========================================================
# DÉFINITION DU NCA
# =========================================================
class NCA(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels * 2, 128, 1)
        self.conv2 = nn.Conv2d(128, channels, 1)
        kernel = torch.tensor([[0.05,0.2,0.05],[0.2,-1.0,0.2],[0.05,0.2,0.05]], dtype=torch.float32)
        self.register_buffer("perception_kernel", kernel.view(1,1,3,3))

    def perceive(self, x):
        x_padded = F.pad(x, pad=(1,1,1,1), mode="reflect")
        kernel = self.perception_kernel.expand(self.channels,-1,-1,-1)
        y = F.conv2d(x_padded, kernel, bias=None, groups=self.channels)
        return torch.cat([x, y], dim=1)

    def step(self, x, source_mask=None):
        y = self.perceive(x)
        y = F.relu(self.conv1(y))
        y = self.conv2(y)
        x = x + UPDATE_RATE * y
        if source_mask is not None:
            x[:,0:1] = x[:,0:1] * (~source_mask) + source_mask.float() * SOURCE_INTENSITY
        return x

    def forward(self, x, steps, source_mask=None, return_history=False):
        history = [x.clone()]
        for _ in range(steps):
            x = self.step(x, source_mask)
            history.append(x.clone())
        if return_history:
            return history
        return x

# =========================================================
# Fonction de mesure de stabilité
# =========================================================
def measure_stability(nca, grid, source_mask, max_steps=60):
    """
    Rollout long pour mesurer la stabilité :
    - var_time : variance temporelle moyenne par pixel
    - mean_energy : moyenne de l'énergie totale
    - energy_var : variance de l'énergie totale
    - mean_grad : gradient spatial moyen (lissage)
    """
    with torch.no_grad():
        history = nca(grid.clone(), max_steps, source_mask, return_history=True)

    frames = torch.stack([h[0,0] for h in history], dim=0)

    var_time = frames.var(dim=0).mean().item()
    energy_per_frame = frames.sum(dim=(1,2))
    mean_energy = energy_per_frame.mean().item()
    energy_var = energy_per_frame.var().item()
    dx = frames[:,1:,:] - frames[:,:-1,:]
    dy = frames[:,:,1:] - frames[:,:,:-1]
    mean_grad = (dx.abs().mean() + dy.abs().mean()).item()

    return {
        "var_time": var_time,
        "mean_energy": mean_energy,
        "energy_var": energy_var,
        "mean_grad": mean_grad,
    }

# =========================================================
# Setup modèle, optim, loss
# =========================================================
nca = NCA(channels=16).to(DEVICE)
optimizer = optim.Adam(nca.parameters(), lr=0.01)
loss_fn = nn.MSELoss(reduction="mean")

source_mask = torch.zeros((1,1,GRID_SIZE,GRID_SIZE), dtype=torch.bool, device=DEVICE)
source_mask[0,0,GRID_SIZE//2, GRID_SIZE//2] = True

# =========================================================
# Préparer CSV
# =========================================================
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch","loss","var_time","mean_energy","energy_var","mean_grad"])
    writer.writeheader()

# =========================================================
# Training loop
# =========================================================
plt.ion()
fig, axes = plt.subplots(1,3,figsize=(10,4))

for epoch in range(LEARNING_STEPS+1):
    # Initial grid
    grid = torch.zeros((1,16,GRID_SIZE,GRID_SIZE), device=DEVICE)
    grid[0,0,GRID_SIZE//2, GRID_SIZE//2] = SOURCE_INTENSITY

    # Steps aléatoires (solution 3)
    epoch_nb_steps_simulated = random.randint(MIN_TRAIN_STEPS, MAX_TRAIN_STEPS)

    # Rollout court pour training
    history = nca(grid, epoch_nb_steps_simulated, source_mask, return_history=True)

    # Loss multi-step
    total_loss = 0
    for t in range(epoch_nb_steps_simulated+1):
        total_loss += loss_fn(history[t][:,0:1], target_history[t])
    total_loss /= (epoch_nb_steps_simulated+1)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # --- Mesure de stabilité + affichage live ---
    if epoch % 20 == 0:
        metrics = measure_stability(nca, grid, source_mask, max_steps=N_STEPS*3)

        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[0].imshow(to_numpy(grid[0,0]), cmap="plasma", vmin=0, vmax=1)
        axes[0].set_title("Init")
        axes[1].imshow(to_numpy(target_history[-1,0]), cmap="plasma", vmin=0, vmax=1)
        axes[1].set_title(f"Target (step {N_STEPS})")
        axes[2].imshow(to_numpy(history[-1][0,0]), cmap="plasma", vmin=0, vmax=1)
        axes[2].set_title(
            f"Pred (epoch {epoch})\nLoss={total_loss.item():.6f}\n"
            f"VarT={metrics['var_time']:.4f}, Evar={metrics['energy_var']:.4f}"
        )
        plt.pause(0.01)

        # Sauvegarde CSV
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch","loss","var_time","mean_energy","energy_var","mean_grad"])
            writer.writerow({
                "epoch": epoch,
                "loss": total_loss.item(),
                "var_time": metrics["var_time"],
                "mean_energy": metrics["mean_energy"],
                "energy_var": metrics["energy_var"],
                "mean_grad": metrics["mean_grad"]
            })

plt.ioff()
plt.show()
