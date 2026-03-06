from PIL import Image
from torchvision import transforms
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


class SimpleVAE(torch.nn.Module):
    # 定义一个最简单的 VAE
    def __init__(self):
        super(SimpleVAE, self).__init__()
        # =======================
        # Encoder（编码器）
        # 输入图片大小: 3 x 32 x 32
        # =======================
        # 第1层卷积: 输入3通道(RGB), 输出16通道
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        # 第2层卷积: 输入16通道, 输出32通道
        self.conv2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        # 池化层: 每次把宽高缩小一半
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # CIFAR-10 图片大小是 3x32x32
        # 第1次卷积后 -> 16x32x32，再池化 -> 16x16x16
        # 第2次卷积后 -> 32x16x16，再池化 -> 32x8x8
        # 展平后大小 = 32*8*8
        # VAE 不再输出分类结果，而是输出潜变量分布的两个参数:
        # mu: 均值
        # logvar: 对数方差
        self.fc_mu = torch.nn.Linear(32 * 8 * 8, 128)
        self.fc_logvar = torch.nn.Linear(32 * 8 * 8, 128)
        # =======================
        # Decoder（解码器）
        # 把潜变量 z 还原成图片
        # =======================
        # 先把潜变量 z 映射回特征图大小
        self.fc_decode = torch.nn.Linear(128, 32 * 8 * 8)
        # 反卷积 / 上采样
        # 32x8x8 -> 16x16x16
        self.deconv1 = torch.nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2
        )
        # 16x16x16 -> 3x32x32
        self.deconv2 = torch.nn.ConvTranspose2d(
            in_channels=16,
            out_channels=3,
            kernel_size=2,
            stride=2
        )
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def encode(self, x):
        # 编码过程:
        # 输入图片 -> 卷积提取特征 -> 输出 mu 和 logvar
        # 第1层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv1(x)))
        # 第2层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv2(x)))
        # 展平，准备接全连接层
        x = x.view(x.size(0), -1)
        # 输出潜变量分布参数
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # 重参数化技巧:
        # 直接采样 z ~ N(mu, sigma^2) 不方便反向传播
        # 所以改写成:
        # z = mu + std * eps
        # 其中 eps ~ N(0,1)
        std = torch.exp(0.5 * logvar)   # 标准差
        eps = torch.randn_like(std)     # 采样标准正态噪声
        z = mu + eps * std              # 得到潜变量 z
        return z

    def decode(self, z):
        # 解码过程:
        # 潜变量 z -> 特征图 -> 重建图片
        # 全连接恢复到卷积特征图维度
        x = self.relu(self.fc_decode(z))
        # reshape 回 4维张量: [batch, channel, height, width]
        x = x.view(x.size(0), 32, 8, 8)
        # 第1层反卷积: 32x8x8 -> 16x16x16
        x = self.relu(self.deconv1(x))
        # 第2层反卷积: 16x16x16 -> 3x32x32
        # 用 sigmoid 把输出压到 [0, 1]
        x = self.sigmoid(self.deconv2(x))
        return x

    def forward(self, pixel_values, targets=None):
        # VAE 的整体流程:
        # 1. 编码，得到 mu 和 logvar
        # 2. 重参数化采样，得到 z
        # 3. 解码，得到重建图像 recon
        # 4. 计算 VAE Loss
        # 编码
        mu, logvar = self.encode(pixel_values)
        # 采样
        z = self.reparameterize(mu, logvar)
        # 解码重建
        recon = self.decode(z)
        # =======================
        # 计算 VAE 的损失
        # =======================
        # 1. 重建损失
        # 让 recon 尽量接近原图 pixel_values
        # 这里为了简单，使用 MSE Loss
        recon_loss = torch.nn.functional.mse_loss(
            recon, pixel_values, reduction="mean"
        )
        # 2. KL 散度损失
        # 让学习到的潜变量分布尽量接近标准正态分布 N(0,1)
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        # 3. 总损失
        loss = recon_loss + kl_loss
        # Trainer 训练时，最关键的是返回 loss
        return {
            "loss": loss,
            "reconstructions": recon,
            "mu": mu,
            "logvar": logvar
        }


def preprocess_images(example):
    # 定义 PIL -> Tensor 的预处理流程
    image_transform = transforms.Compose([
        transforms.Resize((32, 32)),   # CIFAR10 本身就是 32x32，这里保留
        transforms.ToTensor(),         # 转成 Tensor，并归一化到 [0,1]
    ])
    # 读取图片
    image = Image.open(example["image_path"]).convert("RGB")
    # 图像预处理
    pixel_values = image_transform(image)
    # 这里已经不做分类了，所以 label 实际不参与训练
    # 但为了尽量少改 Trainer / DataCollator 结构，保留一个 targets 字段
    return {
        "pixel_values": pixel_values,
        "targets": 0
    }


class DataCollator:

    def __call__(self, features):
        # 将多条样本拼成一个 batch
        # 拼接图片 Tensor
        pixel_values = torch.stack(
            [torch.tensor(f["pixel_values"], dtype=torch.float32) for f in features],
            dim=0
        )
        # targets 保留，但 VAE 实际不会用它
        targets = torch.tensor(
            [f["targets"] for f in features],
            dtype=torch.long
        )
        # 返回给模型 forward(self, pixel_values, targets=None)
        return {
            "pixel_values": pixel_values,
            "targets": targets
        }


if __name__ == "__main__":
    # 训练集和测试集路径
    train_file = "../datasets/cifar10/train.csv"
    test_file = "../datasets/cifar10/test.csv"
    # 读取 csv 数据集
    datasets = load_dataset(
        "csv",
        data_files={
            "train": train_file,
            "test": test_file
        }
    )
    # 数据预处理
    # num_proc=4 表示用4个进程并行处理
    datasets = datasets.map(preprocess_images, num_proc=4)
    # 定义 DataCollator
    data_collator = DataCollator()
    # 定义模型
    model = SimpleVAE()
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./checkpoints",      # 模型保存路径
        evaluation_strategy="steps",     # 按 step 做评估
        eval_steps=1000,                 # 每1000步评估一次
        logging_strategy="steps",        # 按 step 记录日志
        logging_steps=10,                # 每10步记录一次日志
        save_strategy="steps",           # 按 step 保存模型
        save_steps=1000,                 # 每1000步保存一次模型
        learning_rate=0.001,             # VAE 常用稍小一点学习率，更稳定
        per_device_train_batch_size=20,  # 每个设备上的训练 batch size
        per_device_eval_batch_size=20,   # 每个设备上的评估 batch size
        weight_decay=0.0,                # VAE 这里可以先不用 weight decay
        save_total_limit=2,              # 最多保留2个 checkpoint
        num_train_epochs=10,             # 训练10个 epoch
        fp16=False,
        load_best_model_at_end=True,     # 训练结束后加载最优模型
        metric_for_best_model="eval_loss",
        greater_is_better=False,         # eval_loss 越小越好
        remove_unused_columns=False,
        dataloader_num_workers=4,
        label_names=["targets"],         # 保留这个字段，虽然训练其实没用它
        # deepspeed="",                  # 如果没配置 deepspeed，建议注释掉
        local_rank=-1,                   # 单机单卡时一般保持 -1
    )
    # 定义 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
    )
    # 开始训练
    trainer.train()