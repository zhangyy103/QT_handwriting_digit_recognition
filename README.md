# QT小程序: 手数字识别

### 简介
本项目实现了简单的0-9的手写数字识别，程序界面使用QT开发，数字识别算法采用卷积神经网络  

### 项目环境
| 环境 | 版本 |
| ----------- | ----------- |
| QT      | Qt_6_7_2 |
| c++编译器 | MSVC2019_64bit | 
| 运行平台 | windows11 |
| python解释器 | 3.9.19 |
| torch  | 1.12.0+cu113 |
| torchvision | 0.13.0+cu113 |

### 搭建思路
1. QT界面开发 简单的界面绘制左右键操作
2. 数字识别算法  
    采用LetNet_5模型  
    训练数据集是经典的的mnist数据集  
    网络结构如下
    ```python
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
    ```
    模型的训练等详细信息[请移步](https://github.com/zhangyy103/deep_learning)
3. CPython  
    在C++(qt)中调用python写的识别算法  
    环境配置一言难尽