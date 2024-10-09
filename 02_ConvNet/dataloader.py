# 步骤2，数据加载
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


"""_summary_
数据加载及处理, 发送给模型训练的数据,需要是模型能够处理的。(机器加工豆子,放芝麻就无法加工，带着皮也无法加工)。
torch官方已经实现了基本的数据加载器,我们只需继承并重写部分方法就ok。
关于数据集格式(仅针对图像分类数据集而言):
    # 无论哪种数据集,只是原始形式不同，有公开数据集，有私有数据集，最终的目的都是处理成模型能够处理的格式
    1. 经典公开数据集:mnist、cifar10、cifar100、ImageNet
        1.1 mnist、cifar10数据集已经可以在线下载,快速加载(通过torchvision几行代码就能实现)。
        1.2 cifar100、ImageNet等公开数据集属于已经定义好了图像类别、格式等等，需要自行加载处理。
    2. 自定义数据集：猫狗分类、花卉分类等等
"""

class OnlineDatasets():
    def __init__(self) -> None:
        """
        对于图像预处理，比如统一尺寸、归一化、转换为tensor数据格式等等，torchvision已经实现了基本的方法，可以直接调用，表示数据变换方式
        对于一些数据增强方案也已经实现，大家可以查看torchvision.transformers的方法实现
        """
        transform = transforms.Compose([
            transforms.Resize((227, 227)),  # 将图片大小调整为227x227，以适应AlexNet输入
            transforms.ToTensor(),           # 将图片转换为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
            ])
        # 加载训练数据集
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # 加载测试数据集
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # num_workers参数为加载数据使用的线程数，windows系统下建议设置为0，linux系统下可以设置为cpu核数*2,太大了反而速度会变慢
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        return train_loader, test_loader

# 自定义数据集加载
class CustomDatasets(Dataset):
    def __init__(self, imgdir) -> None:
        super().__init__()