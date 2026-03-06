import pickle
import numpy as np
from PIL import Image
import os

# 数据路径
data_dir = r"cifar-10-batches-py"
output_dir = "cifar10_samples"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/images", exist_ok=True)

# 加载类别名称
with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
    meta = pickle.load(f, encoding='bytes')
    classes = [name.decode('utf-8') for name in meta[b'label_names']]

# 加载第一个数据批次
with open(os.path.join(data_dir, 'data_batch_1'), 'rb') as f:
    batch = pickle.load(f, encoding='bytes')
    images = batch[b'data']
    labels = batch[b'labels']

# 提取 100 个样本
labels_file = open(f"{output_dir}/labels.txt", "w", encoding="utf-8")
labels_file.write("filename,label_index,label_name\n")

for i in range(100):
    # CIFAR-10 图像是 10000 x 3072 (32x32x3)
    # 每 1024 个值分别对应 R, G, B 通道
    img_data = images[i]
    r = img_data[0:1024].reshape(32, 32)
    g = img_data[1024:2048].reshape(32, 32)
    b = img_data[2048:3072].reshape(32, 32)

    # 合并 RGB 通道
    img_array = np.dstack((r, g, b))

    # 转换为 PIL Image 并保存
    image = Image.fromarray(img_array)
    image_path = f"{output_dir}/images/sample_{i:03d}.png"
    image.save(image_path)

    # 写入标签
    label = labels[i]
    labels_file.write(f"sample_{i:03d}.png,{label},{classes[label]}\n")
    print(f"已保存: sample_{i:03d}.png -> {classes[label]}")

labels_file.close()
print(f"\n完成！100 个样本已保存到 {output_dir}/ 目录")
print(f"图片保存在: {output_dir}/images/")
print(f"标签文件: {output_dir}/labels.txt")
