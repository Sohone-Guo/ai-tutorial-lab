from PIL import Image
from torchvision import transforms
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


class SimplePixelRegressor(torch.nn.Module):
    # 定义最简单的逐像素回归模型
    def __init__(self):
        super(SimplePixelRegressor, self).__init__()
        # 输入是前一个像素的 RGB(3维)，输出是当前像素的 RGB(3维)
        self.rnn = torch.nn.GRU(
            input_size=3,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.fc = torch.nn.Linear(32, 3)

        # 回归损失
        self.criterion = torch.nn.MSELoss()

    def forward(self, input_values, targets=None):
        # input_values: [batch, 1024, 3]
        x, _ = self.rnn(input_values)
        preds = self.fc(x)  # [batch, 1024, 3]

        loss = None
        if targets is not None:
            loss = self.criterion(preds, targets)

        return {
            "loss": loss,
            "logits": preds
        }


def preprocess_images(example):
    image_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # [3, 32, 32], 数值范围 [0, 1]
    ])

    image = Image.open(example["image_path"]).convert("RGB")
    pixel_values = image_transform(image)  # [3, 32, 32]

    # 改成按“左上 -> 右下”的像素顺序展开
    # 先从 [3, 32, 32] 变成 [32, 32, 3]
    pixel_values = pixel_values.permute(1, 2, 0).contiguous()

    # 再展平成 [1024, 3]
    pixel_values = pixel_values.view(-1, 3)

    # input_values 的第 0 个位置用 0 作为起始输入
    # 第 t 个位置输入前一个像素，预测当前像素
    input_values = torch.zeros_like(pixel_values)
    input_values[1:] = pixel_values[:-1]

    targets = pixel_values

    return {
        "input_values": input_values,
        "targets": targets
    }


class DataCollator:

    def __call__(self, features):
        input_values = torch.stack(
            [torch.tensor(f["input_values"], dtype=torch.float32) for f in features],
            dim=0
        )  # [batch, 1024, 3]

        targets = torch.stack(
            [torch.tensor(f["targets"], dtype=torch.float32) for f in features],
            dim=0
        )  # [batch, 1024, 3]

        return {
            "input_values": input_values,
            "targets": targets
        }


if __name__ == "__main__":
    train_file = "../datasets/cifar10/train.csv"
    test_file = "../datasets/cifar10/test.csv"

    datasets = load_dataset(
        "csv", data_files={"train": train_file, "test": test_file}
    )
    datasets = datasets.map(preprocess_images, num_proc=4)

    data_collator = DataCollator()

    model = SimplePixelRegressor()

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
        learning_rate=0.001,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        weight_decay=0.0,
        save_total_limit=2,
        num_train_epochs=10,
        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        label_names=["targets"],
        deepspeed=None,
        local_rank=-1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()