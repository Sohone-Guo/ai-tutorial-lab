from PIL import Image
from torchvision import transforms
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


class SimpleDiffusion(torch.nn.Module):
    # 最简单的 Diffusion: 训练模型预测噪声
    def __init__(self, num_timesteps=1000):
        super(SimpleDiffusion, self).__init__()

        # ===== 1. 噪声预测网络（尽量保留你原来的 CNN 结构）=====
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(32 * 8 * 8, 128)
        # 原来 fc2 输出 10 类；现在改成输出 3*32*32，表示预测整张图片上的噪声
        self.fc2 = torch.nn.Linear(128, 3 * 32 * 32)

        self.relu = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

        # ===== 2. 最简单时间步嵌入 =====
        self.t_embed = torch.nn.Embedding(num_timesteps, 32 * 8 * 8)

        # ===== 3. DDPM 的 beta / alpha 调度 =====
        self.num_timesteps = num_timesteps
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(self, pixel_values):
        x0 = pixel_values
        batch_size = x0.size(0)
        device = x0.device

        # ===== 4. 随机采样时间步 t =====
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # ===== 5. 给原图加噪，构造 x_t =====
        noise = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bars[t].view(batch_size, 1, 1, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

        # ===== 6. 用 CNN 预测噪声 =====
        x = self.pool(self.relu(self.conv1(xt)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(batch_size, -1)

        # 加入时间步信息
        x = x + self.t_embed(t)

        x = self.relu(self.fc1(x))
        pred_noise = self.fc2(x)
        pred_noise = pred_noise.view(batch_size, 3, 32, 32)

        # ===== 7. Diffusion 训练目标：预测真实噪声 =====
        loss = self.criterion(pred_noise, noise)

        return {
            "loss": loss,
            "pred_noise": pred_noise
        }


def preprocess_images(example):
    image_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # diffusion 通常更常见的是归一化到 [-1, 1]
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = image_transform(image)

    # 不再需要分类标签
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
    test_file = "../datasets/cifar10/test.csv"

    datasets = load_dataset(
        "csv", data_files={"train": train_file, "test": test_file}
    )
    datasets = datasets.map(preprocess_images, num_proc=4)

    data_collator = DataCollator()

    # 改成 Diffusion 模型
    model = SimpleDiffusion()

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        learning_rate=1e-3,   # diffusion 一般学习率小一点更稳
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=10,
        fp16=False,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        # label_names=["targets"],  # 不再需要
        # deepspeed="",             # 没有配置文件就别写
        # local_rank=-1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()