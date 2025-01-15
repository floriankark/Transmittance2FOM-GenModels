import torch
from dataset import create_dataloader
from config.path import STORAGE, VERVET_DATA
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, Tile_size):
        super(Generator, self).__init__()
        self.Tile_size = Tile_size
        self.img_channels = img_channels
        self.init_size = Tile_size // 4  # Tile size has to be divisible by 4
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256 * self.init_size * self.init_size),
            nn.BatchNorm1d(256 * self.init_size * self.init_size),
            nn.ReLU(),
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Data is normalized between 0 and 1
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        img = self.conv_layers(out)
        return img


class Discriminator_GAN(nn.Module):
    def __init__(self, img_channels, Tile_size):
        super(Discriminator_GAN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear((Tile_size // 8) * (Tile_size // 8) * 256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)


class Discriminator_WGAN(nn.Module):
    def __init__(self, img_channels, Tile_size):
        super(Discriminator_WGAN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear((Tile_size // 8) * (Tile_size // 8) * 256, 1),
        )

    def forward(self, img):
        return self.model(img)


def train_GAN(epochs=30, batch_size=8, Tile_size=64, lr_d=0.0001, lr_g=0.0002):
    latent_dim = 100
    img_channels = 3
    device = torch.device("mps:0" if torch.cuda.is_available() else "cpu")

    generator = Generator(latent_dim, img_channels, Tile_size).to(device)
    discriminator = Discriminator_GAN(img_channels, Tile_size).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)
    )

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.5)

    adversarial_loss = nn.BCELoss()
    dataloader = create_dataloader(
        brain="Vervet1818", map_type="FOM", patch_size=Tile_size, batch_size=batch_size
    )

    for epoch in range(epochs):
        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(device)

            # Add Gaussian noise to make it harder for the discriminator to classify real/fake images
            noisy_real_images = real_images + 0.05 * torch.randn_like(real_images)

            optimizer_D.zero_grad()

            # Create smoothed real labels to improve training stability
            real_labels = torch.full((real_images.size(0), 1), 0.9, device=device)
            fake_labels = torch.zeros((real_images.size(0), 1), device=device)

            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(z)

            # Add Gaussian noise to make it harder for the discriminator to classify real/fake images
            noisy_fake_images = fake_images + 0.05 * torch.randn_like(fake_images)

            real_loss = adversarial_loss(discriminator(noisy_real_images), real_labels)
            fake_loss = adversarial_loss(
                discriminator(noisy_fake_images.detach()), fake_labels
            )
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            gen_labels = torch.ones(real_images.size(0), 1).to(device)
            g_loss = adversarial_loss(discriminator(fake_images), gen_labels)
            g_loss.backward()
            optimizer_G.step()

        scheduler_G.step()
        scheduler_D.step()

        print(
            f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}"
        )
    return generator, discriminator


def train_WGAN(epochs=30, batch_size=8, Tile_size=64, lr_d=0.0001, lr_g=0.0001):
    latent_dim = 100
    img_channels = 3
    n_critic = 5
    clip_value = 0.01

    device = torch.device("mps:0" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim, img_channels, Tile_size).to(device)
    critic = Discriminator_WGAN(img_channels, Tile_size).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(critic.parameters(), lr=lr_d, betas=(0.5, 0.999))

    dataloader = create_dataloader(
        brain="Vervet1818", map_type="FOM", patch_size=Tile_size, batch_size=8
    )

    for epoch in range(epochs):
        for batch_idx, real_images in enumerate(dataloader):
            real_images = real_images.to(device)

            noisy_real_images = real_images + 0.05 * torch.randn_like(real_images)

            optimizer_D.zero_grad()

            z = torch.randn(real_images.size(0), latent_dim).to(device)
            fake_images = generator(z)

            noisy_fake_images = fake_images + 0.05 * torch.randn_like(fake_images)

            real_loss = -torch.mean(critic(noisy_real_images))
            fake_loss = torch.mean(critic(noisy_fake_images.detach()))
            c_loss = real_loss + fake_loss
            c_loss.backward()
            optimizer_D.step()

            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if batch_idx % n_critic == 0:
                optimizer_G.zero_grad()
                g_loss = -torch.mean(critic(fake_images))
                g_loss.backward()
                optimizer_G.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Critic Loss: {c_loss.item():.4f} | Generator Loss: {g_loss.item():.4f}"
        )
    return generator, critic


def generate_images(generator):
    device = torch.device("mps:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 100
    with torch.no_grad():
        z = torch.randn(16, latent_dim).to(device)
        generated_images = generator(z)
        grid = torchvision.utils.make_grid(generated_images, nrow=4)

        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title("Generated Images After Training")
        plt.show()


if __name__ == "__main__":
    # train GAN and WGAN
    generator_GAN, discriminator_GAN = train_GAN(
        epochs=30, batch_size=8, Tile_size=64, lr_d=0.0001, lr_g=0.0002
    )
    # Save GAN model
    torch.save(generator_GAN.state_dict(), STORAGE / "GAN" / "generator_GAN.pth")
    generator_WGAN, critic_WGAN = train_WGAN(
        epochs=30, batch_size=8, Tile_size=64, lr_d=0.0001, lr_g=0.0001
    )
    # Save WGAN model
    torch.save(generator_WGAN.state_dict(), STORAGE / "WGAN" / "generator_WGAN.pth")
