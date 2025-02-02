import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 定义VAE的编码器和解码器
class Encoder(nn.Module):
    def __init__(self, latent_dim_any):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim_any)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim_any)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 7 * 7)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def forward(self, z):
        z = torch.relu(self.fc(z)).view(-1, 64, 7, 7)
        z = torch.relu(self.deconv1(z))
        z = torch.sigmoid(self.deconv2(z))  # 使用sigmoid将值限制在 [0, 1]
        return z


# 定义VAE
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# 定义损失函数 (重构误差 + KL散度)
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss


# 准备数据
transform = transforms.Compose([
    transforms.ToTensor()  # 保持在 [0, 1] 范围
])

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_data, batch_size=128, shuffle=True)

# 初始化模型、优化器
latent_dim = 20  # 潜在空间维度
vae = VAE(latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {train_loss / len(data_loader.dataset):.4f}")

# 生成新图像
vae.eval()
with torch.no_grad():
    # 生成随机潜在向量
    z = torch.randn(64, latent_dim)
    generated_images = vae.decoder(z)

# 显示生成的图像
generated_images = generated_images.view(-1, 1, 28, 28)
grid = torch.cat([generated_images[i] for i in range(64)], dim=1).squeeze()
plt.imshow(grid, cmap='gray')
plt.show()
