# 步骤1 定义模型
# 下一步步骤2为数据处理，dataloader.py
import torch.nn as nn
import torch



"""_summary_

# 本项目的核心在于讲解和注释如何去操作, 不对为何这样操作做解释。 
# 本文件暂不考虑预训练、模型初始化等细节实现

定义一个模型的基本步骤:
    1. 定义一个类名
    2.在__init__定义各种模块
    3.利用定义好的模块重写forward方法,实现前向传播数据流
    
class 模型名(nn.Moudle):
    def __init__():
        # 定义各种模块，目的是实现各种操作
    
    def forward(x):
        # 前向传播数据流,利用init中定义好的模块进行各种前向传播计算(数据流向)，基本上每个模块定义完成后就有其单独的左右，比如螺栓零件，有不同的尺寸，必须放在合适的位置
"""

# 步骤1.1 实现模型
# 实现方案1
class AlexNet(nn.Module):
    def __init__(self,num_classes, dropout=0.5) -> None:
        super().__init__()
        """_summary_
        x (_type_): _description_ 模型输出的类别数
        """
        # 用于初始化，相当于搭建一个积木模型时，提前准备好各种小零件
        # 模块（零件的定义不分前后），相同参数的模块可以公用，比如ReLU()
        # 卷积过程是一个下采样过程，图像通道数增加，宽高减少。通道数的变化可以通过模块定义直接看出来
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)  # 这里3是输入图像通道数，默认使用彩色图像，如果是黑白图像则应该为1，这里意味着输入一个3通道的图像，进行卷积操作，输出一个64通道的图像。
        # self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        # self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        # self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.relu5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        
            # nn.Dropout(p=dropout)
        self.line1 = nn.Linear(256 * 6 * 6, 4096)
            # nn.ReLU(inplace=True)
            # nn.Dropout(p=dropout)
        self.line2 = nn.Linear(4096, 4096)
            # nn.ReLU(inplace=True)
        self.line3 = nn.Linear(4096, num_classes)
        self.drop= nn.Dropout(p=dropout)
        

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_ 模型的输入
            关于模型我们还可以将其理解为一个机器，我们目的就是用各种零件搭建成一个机器(豆浆机), 输入x(豆子),经过各种模块操作(去皮、研磨、粉碎、加热(假设这四步必须严格遵循先后顺序))后,变成了豆浆(output)
            x的输入尺寸是[B,C,H,W]:
                B是训练时的超参数batch大小,比如我有10000斤豆子要磨成粉,但我的机器不能一次性全部加工,可以每次加工50斤,这里50就是batch大小,对于大批量图像同理,batch越大计算资源占用越大,习惯性设置为2的n次幂。
                C是输入图像通道数,黑白图像是1通道,彩色图像是3通道
        """
        #    前向传播数据流 
        
        # 特征提取CA(卷积模块、激活函数)模块组合是常见的组合
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        
        # 平均池化
        x = self.avgpool(x)
        
        # 展平特征图方便进行特征连接
        x = x.view(x.size(0),-1)
        
        # 全连接层进行分类
        x = self.drop(x)
        x = self.line1(x)
        x = self.relu(x)
        x = self.drop(x)     
        x = self.line2(x)
        x = self.relu(x)
        out = self.line3(x)   
        return out
    
# 实现方案2, pytorch官方实现版
#         """_summary_
#         方案2与方案1的不同在于,方案二采用nn.Sequential进行了实现,优势是简化了forward操作,劣势是如果想对某一层进行不如方案1灵活。
#         方案2相当于在搭建一个机器时,先分两大部分分别去搭建,再将两个模块组合到一起。个人建议方案1
# 
#         """
"""     
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

"""


# 步骤1.2 测试模型
if __name__=="__main__":
    # 实例化模型
    model = AlexNet(num_classes=1000)
    print(model)

    # 输入一个随机张量来查看模型输出
    input = torch.randn(1, 3, 227, 227)
    output = model(input)
    print(output)
