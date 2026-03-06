import torch
from PIL import Image
import numpy as np


class SimplePixelRegressor(torch.nn.Module):
    def __init__(self):
        super(SimplePixelRegressor, self).__init__()
        self.rnn = torch.nn.GRU(
            input_size=3,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.fc = torch.nn.Linear(32, 3)

    def forward(self, input_values, targets=None):
        x, _ = self.rnn(input_values)
        preds = self.fc(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.mse_loss(preds, targets)

        return {
            "loss": loss,
            "logits": preds
        }


def generate_image(model, image_size=32, device="cpu"):
    model.eval()
    model.to(device)

    seq_len = image_size * image_size

    # 初始输入：第一个像素前面没有像素，所以用 0 向量
    current_input = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)

    generated_pixels = []

    hidden = None

    with torch.no_grad():
        for _ in range(seq_len):
            # GRU 单步生成
            output, hidden = model.rnn(current_input, hidden)   # [1, 1, 32]
            pred_pixel = model.fc(output)                       # [1, 1, 3]

            # 取出当前预测像素
            pixel = pred_pixel[:, -1, :]                        # [1, 3]

            # 限制到 [0, 1]
            pixel = torch.clamp(pixel, 0.0, 1.0)

            generated_pixels.append(pixel.squeeze(0).cpu())

            # 当前预测作为下一步输入
            current_input = pixel.unsqueeze(1)                 # [1, 1, 3]

    # [1024, 3]
    generated_pixels = torch.stack(generated_pixels, dim=0)

    # 还原成 [32, 32, 3]
    image_array = generated_pixels.view(image_size, image_size, 3).numpy()

    # 转成 0~255
    image_array = (image_array * 255.0).astype(np.uint8)

    image = Image.fromarray(image_array)
    return image


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 载入模型
    model = SimplePixelRegressor()

    # 把这里改成你训练后保存的模型路径
    checkpoint_path = "./checkpoints/checkpoint-1000/pytorch_model.bin"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 生成图片
    image = generate_image(model, image_size=32, device=device)

    # 保存结果
    image.save("generated.png")
    print("图片已保存到 generated.png")