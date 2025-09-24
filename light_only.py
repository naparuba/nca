import torch
import matplotlib.pyplot as plt

# --- Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 16
N_STEPS = 10
DECAY = 0.1  # fraction of light lost in neighbors
SOURCE_INTENSITY = 1.0

# --- Initialize grid
grid = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE)

# --- Place a source at the center
i0, j0 = GRID_SIZE // 2, GRID_SIZE // 2
grid[i0, j0] = SOURCE_INTENSITY

# --- Mask to keep source fixed
source_mask = torch.zeros_like(grid, dtype=torch.bool)
source_mask[i0, j0] = True

# --- Function: diffusion step
def diffusion_step(grid, source_mask):
    new_grid = grid.clone()
    for i in range(1, GRID_SIZE-1):
        for j in range(1, GRID_SIZE-1):
            if source_mask[i,j]:
                continue  # source stays fixed
            neighbors = grid[i-1:i+2, j-1:j+2]
            new_value = neighbors.mean()  # simple diffusion
            new_grid[i,j] = max(0.0, min(new_value, 1.0))  # clamp [0,1]
    return new_grid

# --- Visualization
plt.ion()
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(grid.cpu().numpy(), cmap="plasma", vmin=0, vmax=1)
ax.set_title("Diffusion Simulation")

for step in range(N_STEPS):
    grid = diffusion_step(grid, source_mask)
    im.set_data(grid.cpu().numpy())
    plt.suptitle(f"Step {step+1}")
    plt.pause(0.1)

plt.ioff()
plt.show()
