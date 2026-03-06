import torchvision
from PIL import Image
import os

# 创建输出目录
# 数据路径
data_dir = r"."
output_dir = "cifar10_samples"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

# 从本地加载 CIFAR-10
dataset = torchvision.datasets.CIFAR10(
    root=data_dir,
    train=True,
    download=False  # 不下载，直接使用本地数据
)

# 提取 100 个样本
with open(f"{output_dir}/labels.txt", "w") as f:
    f.write("filename,label\n")
    for i in range(100):
        img, label = dataset[i]
        img.save(f"{output_dir}/images/{i:03d}.png")
        f.write(f"{i:03d}.png,{label}\n")

print("完成！100 个样本已提取")
