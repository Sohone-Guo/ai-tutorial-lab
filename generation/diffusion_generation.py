import os
import torch
from PIL import Image
from torchvision.utils import save_image


class SimpleDiffusion(torch.nn.Module):
    def __init__(self, num_timesteps=1000):
        super(SimpleDiffusion, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(32 * 8 * 8, 128)
        self.fc2 = torch.nn.Linear(128, 3 * 32 * 32)

        self.relu = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

        self.t_embed = torch.nn.Embedding(num_timesteps, 32 * 8 * 8)

        self.num_timesteps = num_timesteps
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def predict_noise(self, xt, t):
        batch_size = xt.size(0)

        x = self.pool(self.relu(self.conv1(xt)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(batch_size, -1)

        x = x + self.t_embed(t)

        x = self.relu(self.fc1(x))
        pred_noise = self.fc2(x)
        pred_noise = pred_noise.view(batch_size, 3, 32, 32)
        return pred_noise

    def forward(self, pixel_values):
        x0 = pixel_values
        batch_size = x0.size(0)
        device = x0.device

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(batch_size, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

        pred_noise = self.predict_noise(xt, t)
        loss = self.criterion(pred_noise, noise)

        return {
            "loss": loss,
            "pred_noise": pred_noise
        }


@torch.no_grad()
def sample(model, num_images=1, image_size=32, device="cuda"):
    model.eval()

    # 1. 从纯噪声开始
    x = torch.randn(num_images, 3, image_size, image_size).to(device)

    # 2. 从 T-1 逐步去噪到 0
    for t in reversed(range(model.num_timesteps)):
        t_batch = torch.full((num_images,), t, device=device, dtype=torch.long)

        beta_t = model.betas[t]
        alpha_t = model.alphas[t]
        alpha_bar_t = model.alpha_bars[t]

        pred_noise = model.predict_noise(x, t_batch)

        # DDPM 反向采样公式
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        x = (
            1 / torch.sqrt(alpha_t)
        ) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise
        ) + torch.sqrt(beta_t) * z

    # 3. 把 [-1, 1] 转回 [0, 1]
    x = (x.clamp(-1, 1) + 1) / 2
    return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleDiffusion(num_timesteps=1000).to(device)

    # 你的训练结果目录，按实际情况改
    ckpt_path = "./checkpoints/checkpoint-2000/pytorch_model.bin"

    # 如果 Trainer 保存的是 state_dict
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    # 生成 8 张图片
    images = sample(model, num_images=8, image_size=32, device=device)

    os.makedirs("./generated", exist_ok=True)

    # 保存成网格图
    save_image(images, "./generated/sample_grid.png", nrow=4)

    # 也可以单张保存
    for i, img in enumerate(images):
        save_image(img, f"./generated/sample_{i}.png")

    print("生成完成，图片保存在 ./generated/")