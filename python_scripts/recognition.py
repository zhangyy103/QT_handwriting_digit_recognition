import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms

class CNN(nn.Module):
    def __init__(self):
        """搭建神经网络各层"""
        super(CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Tanh(),  # 第一层 卷积层
            nn.AvgPool2d(kernel_size=2, stride=2),  # 第二层 池化层
            nn.Conv2d(6, 16, kernel_size=5), nn.Tanh(),  # 第三层 卷积层
            nn.AvgPool2d(kernel_size=2, stride=2),  # 第四层 池化层
            nn.Conv2d(16, 120, kernel_size=5), nn.Tanh(),  # 第五层 卷积层
            nn.Flatten(),  # 展平
            nn.Linear(120, 84), nn.Tanh(),  # 第六层 全连接层
            nn.Linear(84, 10)  # 第七层 输出层
        )

    def forward(self, x):
        """前向传播"""
        y = self.net(x)
        return y

# 注册 CNN 类以确保在反序列化时可用
def register_cnn_class():
    import sys
    sys.modules['__main__'].CNN = CNN

register_cnn_class()

def predict_digit():
    # 加载模型
    model = torch.load('../../python_scripts/LetNet-5_mnist_model.pth', map_location=torch.device('cuda'))
    model.eval()  # 设置模型为评估模式

    # 处理图像
    image = Image.open('../../assets/saved_canvas.png')
    pil_image = image.convert("L").resize((28, 28))  # 转换为灰度图并调整大小
    pil_image = Image.fromarray(255 - np.array(pil_image))  # 检查图像是否反转
    img_tensor = transforms.ToTensor()(pil_image).unsqueeze(0).to('cuda')
    img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)  # 正确归一化

    # 进行预测
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    return predicted.item()

if __name__ == "__main__":
    print(predict_digit())