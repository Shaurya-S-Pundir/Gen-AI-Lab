# ============================
# GAN on MNIST (PyTorch)
# Single-file, end-to-end
# ============================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import numpy as np

# ======================
# USER CONFIGURATION
# ======================
dataset_choice = "mnist"   # 'mnist' only here
epochs = 50
batch_size = 128
noise_dim = 100
learning_rate = 0.0002
save_interval = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# CREATE OUTPUT FOLDERS
# ======================
os.makedirs("generated_samples", exist_ok=True)
os.makedirs("final_generated_images", exist_ok=True)

# ======================
# DATASET (MNIST)
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
img_shape = (1, 28, 28)

# ======================
# GENERATOR
# ======================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), *img_shape)

# ======================
# DISCRIMINATOR
# ======================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# ======================
# INITIALIZE MODELS
# ======================
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# ======================
# TRAINING LOOP
# ======================
for epoch in range(1, epochs + 1):
    D_loss_epoch = 0.0
    G_loss_epoch = 0.0
    correct = 0
    total = 0

    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        curr_batch = imgs.size(0)

        real_labels = torch.ones(curr_batch, 1, device=device)
        fake_labels = torch.zeros(curr_batch, 1, device=device)

        # ---- Train Discriminator ----
        optimizer_D.zero_grad()

        real_preds = D(imgs)
        d_real_loss = criterion(real_preds, real_labels)

        z = torch.randn(curr_batch, noise_dim, device=device)
        fake_imgs = G(z)
        fake_preds = D(fake_imgs.detach())
        d_fake_loss = criterion(fake_preds, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Accuracy
        correct += (real_preds > 0.5).sum().item()
        correct += (fake_preds < 0.5).sum().item()
        total += 2 * curr_batch

        # ---- Train Generator ----
        optimizer_G.zero_grad()
        gen_preds = D(fake_imgs)
        g_loss = criterion(gen_preds, real_labels)
        g_loss.backward()
        optimizer_G.step()

        D_loss_epoch += d_loss.item()
        G_loss_epoch += g_loss.item()

    D_acc = 100.0 * correct / total

    print(
        f"Epoch {epoch}/{epochs} | "
        f"D_loss: {D_loss_epoch/len(dataloader):.2f} | "
        f"D_acc: {D_acc:.2f}% | "
        f"G_loss: {G_loss_epoch/len(dataloader):.2f}"
    )

    # ---- Save Samples ----
    if epoch % save_interval == 0:
        z = torch.randn(25, noise_dim, device=device)
        samples = G(z)
        utils.save_image(
            samples,
            f"generated_samples/epoch_{epoch:02d}.png",
            nrow=5,
            normalize=True
        )

# ======================
# SAVE FINAL 100 IMAGES
# ======================
z = torch.randn(100, noise_dim, device=device)
final_images = G(z)

utils.save_image(
    final_images,
    "final_generated_images/final_100.png",
    nrow=10,
    normalize=True
)

# ======================
# SIMPLE MNIST CLASSIFIER
# ======================
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

classifier = Classifier().to(device)
classifier.eval()

# ======================
# LABEL PREDICTION
# ======================
with torch.no_grad():
    outputs = classifier(final_images)
    preds = torch.argmax(outputs, dim=1)

unique, counts = np.unique(preds.cpu().numpy(), return_counts=True)

print("\nLabel distribution of generated images:")
for u, c in zip(unique, counts):
    print(f"Digit {u}: {c}")