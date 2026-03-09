import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# --- Simple dataset of labels ---
class ShapeDataset(Dataset):
    def __init__(self, labels, n_samples=1000):
        self.labels = labels
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        label = np.random.choice(self.labels)
        return label

# --- Generator ---
class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self, img_dim, label_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim + label_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = torch.cat([img, labels], dim=1)
        return self.model(x)

# --- Training loop (simplified) ---
noise_dim = 10
label_dim = 2   # e.g. one-hot for "circle" or "square"
img_dim = 28*28  # flattened image

G = Generator(noise_dim, label_dim, img_dim)
D = Discriminator(img_dim, label_dim)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# Dummy training loop
for epoch in range(10):
    # Fake batch
    noise = torch.randn(16, noise_dim)
    labels = torch.eye(label_dim)[torch.randint(0, label_dim, (16,))]
    fake_imgs = G(noise, labels)

    # Train discriminator
    real_imgs = torch.randn(16, img_dim)  # placeholder for real data
    real_labels = labels
    D_real = D(real_imgs, real_labels)
    D_fake = D(fake_imgs.detach(), labels)

    loss_D = criterion(D_real, torch.ones_like(D_real)) + \
             criterion(D_fake, torch.zeros_like(D_fake))
    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    # Train generator
    D_fake = D(fake_imgs, labels)
    loss_G = criterion(D_fake, torch.ones_like(D_fake))
    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    print(f"Epoch {epoch}: Loss_D={loss_D.item():.4f}, Loss_G={loss_G.item():.4f}")



# After training, generate images conditioned on labels
def generate_and_plot(generator, noise_dim, label_dim, img_dim):
    generator.eval()
    with torch.no_grad():
        # Create noise
        noise = torch.randn(2, noise_dim)

        # Labels: one-hot for circle and square
        labels = torch.eye(label_dim)

        # Generate images
        fake_imgs = generator(noise, labels)

        # Reshape to 28x28
        fake_imgs = fake_imgs.view(-1, 28, 28).cpu().numpy()

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(6,3))
        for i, ax in enumerate(axes):
            ax.imshow(fake_imgs[i], cmap="gray")
            ax.set_title("Circle" if i==0 else "Square")
            ax.axis("off")
        plt.show()

# Example usage after training loop
generate_and_plot(G, noise_dim, label_dim, img_dim)