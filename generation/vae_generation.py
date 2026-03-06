import os
import torch
from PIL import Image
from safetensors.torch import load_file


class SimpleVAE(torch.nn.Module):
    # 和训练脚本里的模型保持一致
    def __init__(self):
        super(SimpleVAE, self).__init__()
        # ===== Encoder =====
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_mu = torch.nn.Linear(32 * 8 * 8, 128)
        self.fc_logvar = torch.nn.Linear(32 * 8 * 8, 128)
        # ===== Decoder =====
        self.fc_decode = torch.nn.Linear(128, 32 * 8 * 8)
        self.deconv1 = torch.nn.ConvTranspose2d(
            in_channels=32, out_channels=16, kernel_size=2, stride=2
        )
        self.deconv2 = torch.nn.ConvTranspose2d(
            in_channels=16, out_channels=3, kernel_size=2, stride=2
        )
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.relu(self.fc_decode(z))
        x = x.view(x.size(0), 32, 8, 8)
        x = self.relu(self.deconv1(x))
        x = self.sigmoid(self.deconv2(x))
        return x

    def forward(self, pixel_values, targets=None):
        mu, logvar = self.encode(pixel_values)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        recon_loss = torch.nn.functional.mse_loss(recon, pixel_values, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        return {
            "loss": loss,
            "reconstructions": recon,
            "mu": mu,
            "logvar": logvar
        }


def save_tensor_as_image(tensor, save_path):
    # tensor: [3, 32, 32], 值范围 [0,1]
    tensor = tensor.detach().cpu().clamp(0, 1)
    array = (tensor * 255).byte().permute(1, 2, 0).numpy()
    image = Image.fromarray(array)
    image.save(save_path)


def load_model(model, checkpoint_path, device):
    # 兼容 pytorch_model.bin
    bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    # 兼容 model.safetensors
    safe_path = os.path.join(checkpoint_path, "model.safetensors")

    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"加载模型成功: {bin_path}")
    elif os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path)
        model.load_state_dict(state_dict)
        print(f"加载模型成功: {safe_path}")
    else:
        raise FileNotFoundError(
            f"在 {checkpoint_path} 下没有找到 pytorch_model.bin 或 model.safetensors"
        )


if __name__ == "__main__":
    # =========================
    # 1. 参数配置
    # =========================
    checkpoint_path = "./checkpoints/checkpoint-11000/pytorch_model.bin"  # 改成你的模型目录
    output_dir = "./generated_images"
    num_images = 8          # 生成多少张图
    latent_dim = 128        # 要和训练时的潜变量维度一致

    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # 2. 加载模型
    # =========================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleVAE().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # =========================
    # 3. 随机采样并生成图片
    # =========================
    with torch.no_grad():
        for i in range(num_images):
            # 从标准正态分布采样 z
            z = torch.randn(1, latent_dim).to(device)
            # 用 decoder 生成图片
            generated = model.decode(z)   # [1, 3, 32, 32]
            # 取出第1张图保存
            image_tensor = generated[0]
            save_path = os.path.join(output_dir, f"generated_{i}.png")
            save_tensor_as_image(image_tensor, save_path)
            print(f"已保存: {save_path}")

    print("图片生成完成。")