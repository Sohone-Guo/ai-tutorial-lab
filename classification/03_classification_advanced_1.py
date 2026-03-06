# 目的将Cifar-10的数据集提取成常见数据集模型，此部分由AI直接生成目的就是提取数据。
# data/cifar10/
# ├── train/
# │   ├── 000001.png
# │   ├── 000002.png
# │   └── ...
# ├── test/
# │   ├── 000001.png
# │   ├── 000002.png
# │   └── ...
# ├── train.csv
# ├── test.csv
# └── label_names.txt

import os
import csv
import pickle
import numpy as np
from PIL import Image


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def load_label_names(meta_file):
    meta = load_pickle(meta_file)
    label_names = meta[b'label_names']
    label_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in label_names]
    return label_names


def save_batch_to_images_and_csv(batch_file, output_dir, csv_writer, start_index=0):
    """
    读取单个 CIFAR-10 batch 文件，保存图片并写入 CSV。
    """
    batch = load_pickle(batch_file)

    images = batch[b'data']          # shape: (N, 3072)
    labels = batch[b'labels']        # length: N
    filenames = batch.get(b'filenames', None)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(images)):
        img = images[i].reshape(3, 32, 32).transpose(1, 2, 0)   # -> (32, 32, 3)
        label = labels[i]

        if filenames is not None:
            filename = filenames[i]
            if isinstance(filename, bytes):
                filename = filename.decode('utf-8')
        else:
            filename = f"{start_index + i:06d}.png"

        # 保险起见，统一改成 png 名字
        filename_no_ext = os.path.splitext(filename)[0]
        save_name = f"{filename_no_ext}_{start_index + i:06d}.png"
        save_path = os.path.join(output_dir, save_name)

        Image.fromarray(img).save(save_path)

        csv_writer.writerow([save_path, label])

    return start_index + len(images)


def export_cifar10_from_batches_py(cifar_root, output_root):
    """
    从官方 cifar-10-batches-py 提取图片和 CSV
    """
    os.makedirs(output_root, exist_ok=True)

    train_img_dir = os.path.join(output_root, "train")
    test_img_dir = os.path.join(output_root, "test")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)

    train_csv_path = os.path.join(output_root, "train.csv")
    test_csv_path = os.path.join(output_root, "test.csv")
    labels_txt_path = os.path.join(output_root, "label_names.txt")

    # 类别名
    meta_file = os.path.join(cifar_root, "batches.meta")
    label_names = load_label_names(meta_file)
    with open(labels_txt_path, "w", encoding="utf-8") as f:
        for idx, name in enumerate(label_names):
            f.write(f"{idx},{name}\n")

    # 导出训练集
    train_batch_files = [
        os.path.join(cifar_root, f"data_batch_{i}") for i in range(1, 6)
    ]

    with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        start_index = 0
        for batch_file in train_batch_files:
            start_index = save_batch_to_images_and_csv(
                batch_file=batch_file,
                output_dir=train_img_dir,
                csv_writer=writer,
                start_index=start_index
            )

    # 导出测试集
    test_batch_file = os.path.join(cifar_root, "test_batch")
    with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        save_batch_to_images_and_csv(
            batch_file=test_batch_file,
            output_dir=test_img_dir,
            csv_writer=writer,
            start_index=0
        )

    print("导出完成：")
    print("train csv:", train_csv_path)
    print("test csv :", test_csv_path)
    print("labels   :", labels_txt_path)
    print("train img:", train_img_dir)
    print("test img :", test_img_dir)


if __name__ == "__main__":
    cifar_root = "../datasets/cifar-10-batches-py"     # 官方解压目录
    output_root = "../datasets/cifar10"           # 输出目录
    export_cifar10_from_batches_py(cifar_root, output_root)