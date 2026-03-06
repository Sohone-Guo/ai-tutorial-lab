from PIL import Image
from torchvision import transforms
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 第1层卷积: 3 -> 16
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # 第2层卷积: 16 -> 32
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 第3层卷积: 32 -> 64
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 池化层
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        # 输入 3x32x32
        # conv1 + pool -> 16x16x16
        # conv2 + pool -> 32x8x8
        # conv3 + pool -> 64x4x4
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 128)
        self.fc2 = torch.nn.Linear(128, 10)

        self.relu = torch.nn.ReLU()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, pixel_values, targets=None):
        x = pixel_values
        # 第1层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv1(x)))
        # 第2层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv2(x)))
        # 第3层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv3(x)))
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        # 计算损失
        loss = None
        if targets is not None:
            loss = self.criterion(logits, targets)

        return {
            "loss": loss,
            "logits": logits
        }


def preprocess_images(example):
    # 定义一个PLT到Tensor的转化流程
    image_transform = transforms.Compose([
        transforms.Resize((32, 32)),   # CIFAR10 本身就是 32x32，这里写上更稳
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    # 每条数据预处理模块
    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = image_transform(image)
    return {
        "pixel_values": pixel_values,
        "targets": int(example["label"])
    }


class DataCollator:

    def __call__(self, features):
        # 将数据处理成Batch
        pixel_values = torch.stack(
            [torch.tensor(f["pixel_values"], dtype=torch.float32) for f in features],
            dim=0
        )
        # 和模型的def forward(self, pixel_values, targets): 中的参数对应
        targets = torch.tensor([f["targets"] for f in features], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "targets": targets
        }


if __name__ == "__main__":
    train_file = "../datasets/cifar10/train.csv"
    test_file = "../datasets/cifar10/test.csv"
    # 定义模型
    datasets = load_dataset(
        "csv", data_files={"train": train_file, "test": test_file})
    datasets = datasets.map(preprocess_images, num_proc=4) # 多进程处理设置4个进程。
    # 定义DataCollator
    data_collator = DataCollator()
    # 定义模型
    model = SimpleCNN()
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./checkpoints", # 模型保存的地址
        evaluation_strategy="steps",
        eval_steps=1000, # 每10步评估一次
        logging_strategy="steps",
        logging_steps=10, # 每10步记录一次日志
        save_strategy="steps",
        save_steps=1000, # 每100步保存一次模型
        learning_rate=0.001,
        per_device_train_batch_size=20, # 每个设备上的训练批量大小
        per_device_eval_batch_size=20, # 每个设备上的评估批量大小
        weight_decay=0.01,
        save_total_limit=2, # 最多保存2个模型
        num_train_epochs=10, # 训练10个epoch
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        label_names=["targets"],
        deepspeed="", # 多GPU服务器配置
        local_rank=-1, # 多GPU服务器配置
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
    )
    # 开始训练
    trainer.train()