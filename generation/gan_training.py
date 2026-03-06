from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os


# ========== 生成器 ==========
class Generator(torch.nn.Module):
    def __init__(self, noise_dim=100):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(noise_dim, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 3 * 32 * 32),
            torch.nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, z):
        x = self.model(z)
        x = x.view(z.size(0), 3, 32, 32)
        return x


# ========== 判别器 ==========
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3 * 32 * 32, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


def preprocess_images(example):
    # GAN 一般把图片归一化到 [-1, 1]
    image_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = image_transform(image)
    return {
        "pixel_values": pixel_values
    }


class DataCollator:
    def __call__(self, features):
        pixel_values = torch.stack(
            [torch.tensor(f["pixel_values"], dtype=torch.float32) for f in features],
            dim=0
        )
        return {
            "pixel_values": pixel_values
        }


if __name__ == "__main__":
    train_file = "../datasets/cifar10/train.csv"

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 超参数
    noise_dim = 100
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.0002

    os.makedirs("./gan_samples", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)

    # 读取数据
    datasets = load_dataset(
        "csv",
        data_files={"train": train_file}
    )
    datasets = datasets.map(preprocess_images, num_proc=4)

    # DataCollator
    data_collator = DataCollator()

    # DataLoader
    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4
    )

    # 定义模型
    generator = Generator(noise_dim=noise_dim).to(device)
    discriminator = Discriminator().to(device)

    # 损失函数
    criterion = torch.nn.BCELoss()

    # 优化器
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # 固定噪声，用来观察生成效果
    fixed_noise = torch.randn(64, noise_dim).to(device)

    # 开始训练
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            real_images = batch["pixel_values"].to(device)
            batch_size_now = real_images.size(0)

            # 真实图片标签=1，假图片标签=0
            real_labels = torch.ones(batch_size_now, 1).to(device)
            fake_labels = torch.zeros(batch_size_now, 1).to(device)

            # =========================
            # 1. 训练判别器 D
            # =========================
            optimizer_D.zero_grad()

            # 判别真实图片
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_labels)

            # 生成假图片
            z = torch.randn(batch_size_now, noise_dim).to(device)
            fake_images = generator(z)

            # 判别假图片
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)

            # 判别器总损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # =========================
            # 2. 训练生成器 G
            # =========================
            optimizer_G.zero_grad()

            z = torch.randn(batch_size_now, noise_dim).to(device)
            fake_images = generator(z)
            outputs = discriminator(fake_images)

            # 生成器希望判别器把假图当成真图，所以目标是1
            g_loss = criterion(outputs, real_labels)

            g_loss.backward()
            optimizer_G.step()

            if step % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{step}/{len(train_dataloader)}] "
                    f"D Loss: {d_loss.item():.4f} "
                    f"G Loss: {g_loss.item():.4f}"
                )

        # 每个 epoch 保存一次生成结果
        with torch.no_grad():
            fake_images = generator(fixed_noise)
            # 还原到 [0,1] 方便保存
            fake_images = (fake_images + 1) / 2
            save_image(fake_images, f"./gan_samples/epoch_{epoch+1}.png", nrow=8)

        # 保存模型
        torch.save(generator.state_dict(), f"./checkpoints/generator_epoch_{epoch+1}.pt")
        torch.save(discriminator.state_dict(), f"./checkpoints/discriminator_epoch_{epoch+1}.pt")