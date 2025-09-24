import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import random

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID_SIZE = 32
N_STEPS = 20
SOURCE_INTENSITY = 1.0

LEARNING_STEPS = 1000

# --- Helper pour matplotlib ---
def to_numpy(x):
    return x.detach().cpu().numpy()

# --- Génération de la target diffusante ---
def generate_diffusion_target(grid_size=GRID_SIZE, n_steps=N_STEPS, source_intensity=SOURCE_INTENSITY):
    grid = torch.zeros((grid_size, grid_size), device=DEVICE)
    i0, j0 = grid_size // 2, grid_size // 2
    grid[i0, j0] = source_intensity
    source_mask = torch.zeros_like(grid, dtype=torch.bool)
    source_mask[i0, j0] = True
    history = [grid.clone()]
    for _ in range(n_steps):
        new_grid = grid.clone()
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                if source_mask[i,j]:
                    continue
                neighbors = grid[i-1:i+2, j-1:j+2]
                new_grid[i,j] = neighbors.mean()
        grid = new_grid
        history.append(grid.clone())
    # output shape: [steps+1, 1, H, W]
    return torch.stack([h.unsqueeze(0) for h in history], dim=0)

target_history = generate_diffusion_target()
# target_history.shape = (N_STEPS+1,1,H,W)

# --- NCA ---
class NCA(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels*2, 128, 1)
        self.conv2 = nn.Conv2d(128, channels, 1)

    def perceive(self, x):
        # convolution 3x3 pour voisinage
        kernel = torch.tensor([[0.05, 0.2, 0.05],
                               [0.2, -1.0, 0.2],
                               [0.05, 0.2, 0.05]], dtype=torch.float32, device=x.device)
        kernel = kernel.view(1,1,3,3)
        y = F.conv2d(x, kernel.expand(self.channels,-1,-1,-1), padding=1, groups=self.channels)
        return torch.cat([x, y], dim=1)

    def step(self, x, source_mask=None):
        y = self.perceive(x)
        y = F.relu(self.conv1(y))
        y = self.conv2(y)
        x = x + y
        if source_mask is not None:
            x = x * (~source_mask) + source_mask.float()
        return x

    def forward(self, x, steps, source_mask=None, return_history=False):
        history = [x.clone()]
        for _ in range(steps):
            x = self.step(x, source_mask)
            history.append(x.clone())
        if return_history:
            return history
        return x

# --- Setup NCA et optimizer ---
nca = NCA(channels=16).to(DEVICE)
optimizer = optim.Adam(nca.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# --- Source mask ---
source_mask = torch.zeros((1,16,GRID_SIZE,GRID_SIZE), dtype=torch.bool, device=DEVICE)
source_mask[0,0,GRID_SIZE//2, GRID_SIZE//2] = True

# --- Training loop ---
plt.ion()
fig, axes = plt.subplots(1,3,figsize=(9,3))

for epoch in range(LEARNING_STEPS+1):
    # initial grid
    grid = torch.zeros((1,16,GRID_SIZE,GRID_SIZE), device=DEVICE)
    grid[0,0,GRID_SIZE//2, GRID_SIZE//2] = SOURCE_INTENSITY
    
    #epoch_nb_steps_simulated = random.randint(N_STEPS // 2, N_STEPS-1)
    epoch_nb_steps_simulated = N_STEPS
    # rollout NCA
    history = nca(grid, epoch_nb_steps_simulated, source_mask, return_history=True)
    
    # multistep loss
    loss = 0
    for t in range(epoch_nb_steps_simulated+1):
        loss += loss_fn(history[t][:,0:1], target_history[t])
    loss /= (epoch_nb_steps_simulated+1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        # affichage statique
        axes[0].imshow(to_numpy(grid[0,0]), cmap="plasma", vmin=0, vmax=1)
        axes[0].set_title("Init")
        axes[1].imshow(to_numpy(target_history[-1,0]), cmap="plasma", vmin=0, vmax=1)
        axes[1].set_title("Target")
        axes[2].imshow(to_numpy(history[-1][0,0]), cmap="plasma", vmin=0, vmax=1)
        axes[2].set_title(f"Pred (epoch {epoch})\nLoss={loss.item():.8f}")
        plt.pause(0.01)

        # animation de la propagation
        if epoch == LEARNING_STEPS:
            plt.figure(figsize=(10,2))
            for step, h in enumerate(history):
                plt.subplot(1,N_STEPS+1,step+1)
                plt.imshow(to_numpy(h[0,0]), cmap="plasma", vmin=0, vmax=1)
                plt.axis("off")
                plt.title(f"s{step}")
            plt.suptitle(f"Epoch {epoch} rollout")
            #plt.pause(0.1)
            
            #plt.close()

plt.ioff()
plt.show()
