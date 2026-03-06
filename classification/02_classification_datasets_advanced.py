import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


if __name__ == "__main__":
    # 01 数据预处理处理
    # 将类别转成Index数值表示。
    labels_to_idx = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }
    # 将Index数值表示转换为类别名称。
    idx_to_labels = {v: k for k, v in labels_to_idx.items()}
    # 本地数据路径
    DATA_DIR = "../datasets"
    # 读取 CIFAR-10 训练集, 如果是真实照片，等价于使用PIL.Image.open() 打开图片和获取图片的label，并将Label转换为对应的Index，数值表示
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
    )
    # 打印出来数据样式
    for i in range(10):
        # image 目前是通过PIL读取的格式，不是torch.Tensor格式。
        img, label = train_dataset[i]
        print(f"数据样例{i}, 图片：{img.size}, 标签：{idx_to_labels[label]}, 数值表示：{label}") # 一般是先有标签，才能映射数值，这里因为官方数据已经处理好了。
    # 将所有数据转成Tensor格式，一个数据一个数据转化。
    train_dataset_tensors = []
    for i in range(len(train_dataset)):
        print(f"数据Tensor化，当前数据索引：{i}/{len(train_dataset)}", end="\r")
        img, label = train_dataset[i]
        img = transforms.ToTensor()(img)
        train_dataset_tensors.append((img, label))
    # 02 定义模型
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")
    # 模型
    # 2层卷积神经网络
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # 第1层卷积: 输入3通道(RGB), 输出16通道
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
            # 第2层卷积: 输入16通道, 输出32通道
            self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
            # 池化层
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            # 全连接层
            # CIFAR-10 图片大小是 3x32x32
            # 第1次卷积后还是 16x32x32，再池化 -> 16x16x16
            # 第2次卷积后还是 32x16x16，再池化 -> 32x8x8
            self.fc1 = torch.nn.Linear(32 * 8 * 8, 128)
            self.fc2 = torch.nn.Linear(128, 10)

            self.relu = torch.nn.ReLU()

        def forward(self, x):
            # 第1层: 卷积 -> ReLU -> 池化
            x = self.pool(self.relu(self.conv1(x)))
            # 第2层: 卷积 -> ReLU -> 池化
            x = self.pool(self.relu(self.conv2(x)))
            # 展平
            x = x.view(x.size(0), -1)
            # 全连接
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    model = SimpleCNN().to(device)
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 03 开始模型训练
    epoch = 1 # 训练几轮
    for e in range(epoch):
        print(f"第{e}轮训练开始")
        batch_size = 10
        # DataLoader
        train_loader = DataLoader(train_dataset_tensors, batch_size=batch_size, shuffle=True)
        for i, (images, labels) in enumerate(train_loader):
            images_batch = images.to(device)
            labels_batch = labels.to(device)
            # 前向传播
            outputs = model(images_batch)
            # 计算损失
            loss = criterion(outputs, labels_batch)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 打印当前损失, 每10个批次打印一次
            if (i // batch_size) % 10 == 0:
                print(f"当前批次损失: {loss.item():.4f}")
        # 测试正确率, 每轮训练结束后测试一次。
        model.eval()
        with torch.no_grad():
            # 加载测试集
            test_dataset = datasets.CIFAR10(
                root=DATA_DIR,
                train=False,
                download=False,
            )
            # 将所有数据转成Tensor格式，一个数据一个数据转化。
            test_dataset_tensors = []
            for i in range(len(test_dataset)):
                print(f"测试数据Tensor化，当前数据索引：{i}/{len(test_dataset)}", end="\r")
                img, label = test_dataset[i]
                img = transforms.ToTensor()(img)
                test_dataset_tensors.append((img, label))
            # 测试集损失和正确率
            test_loss = 0.0
            correct = 0
            total = 0
            test_loader = DataLoader(test_dataset_tensors, batch_size=batch_size, shuffle=True)
            for i, (images, labels) in enumerate(test_loader):
                images_batch = images.to(device)
                labels_batch = labels.to(device)
                # 前向传播
                outputs = model(images_batch)
                # 计算损失
                loss = criterion(outputs, labels_batch)
                test_loss += loss.item()
                # 计算正确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
            print(f"\n测试集损失: {test_loss / len(test_loader):.4f}, 测试集正确率: {100 * correct / total:.2f}%")


    
