import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------
# Paramètres
# -------------------
H, W = 32, 32
channels = 3   # [density, vx, vy]
steps = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Génération de la trajectoire "oracle"
# -------------------
def physics_step(state):
    density, vx, vy = state  # ATTENTION: ce sont des vues
    density = density.clone()
    vx = vx.clone()
    vy = vy.clone()

    H, W = density.shape

    # masques de rebond
    mask_x = (density[0, :] > 0) | (density[-1, :] > 0)
    mask_y = (density[:, 0] > 0) | (density[:, -1] > 0)

    # mises à jour out-of-place
    vx = torch.where(mask_x.unsqueeze(0).expand_as(vx), -vx, vx)
    vy = torch.where(mask_y.unsqueeze(1).expand_as(vy), -vy, vy)

    return torch.stack([density, vx, vy])






def generate_oracle(steps):
    density = torch.zeros(H, W)
    density[H//2-2:H//2+2, W//2-2:W//2+2] = 1.0
    vx = 0.8 * torch.ones(H, W)
    vy = 0.5 * torch.ones(H, W)
    state = torch.stack([density, vx, vy])
    seq = [state.clone()]
    for _ in range(steps-1):
        state = physics_step(state)
        seq.append(state.clone())
    return torch.stack(seq)

oracle_seq = generate_oracle(steps).to(device)

# -------------------
# Modèle NCA
# -------------------
class StableParticleNCA(nn.Module):
    def __init__(self, channels=3, hidden=32):
        super().__init__()
        self.perception = nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False)
        self.update = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        dx = self.perception(x)
        dx = torch.relu(dx)
        dx = self.update(dx)
        # clip pour éviter explosion
        dx = torch.clamp(dx, -1.0, 1.0)
        return x + dx

nca = StableParticleNCA().to(device)
optimizer = optim.Adam(nca.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -------------------
# Entraînement
# -------------------
epochs = 200
for epoch in range(epochs):
    state = oracle_seq[0].unsqueeze(0).to(device)
    preds = [state]
    for t in range(steps-1):
        state = nca(state)
        # applique rebonds pour stabiliser
        density, vx, vy = state[0]
        mask_x = (density[0,:] == 1.0) | (density[-1,:] == 1.0)
        mask_y = (density[:,0] == 1.0) | (density[:,-1] == 1.0)
        vx[mask_x] *= -1
        vy[mask_y] *= -1
        state[0] = torch.stack([density, vx, vy])
        preds.append(state)

    preds = torch.cat(preds, dim=0)
    loss = loss_fn(preds[:,0], oracle_seq[:,0])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss={loss.item():.6f}")

# -------------------
# Visualisation
# -------------------
state = oracle_seq[0].unsqueeze(0).to(device)
frames = [state[0,0].detach().cpu().numpy()]
for _ in range(steps-1):
    state = nca(state)
    frames.append(state[0,0].detach().cpu().numpy())

fig, ax = plt.subplots()
im = ax.imshow(frames[0], cmap="magma", interpolation="nearest")

def update(frame):
    im.set_array(frames[frame])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200, blit=True)
plt.show()
