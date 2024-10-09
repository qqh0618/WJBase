# 数据集划分文件
# 本文件属于dataloader.py的一部分，在加载数据时，应该已经划分好数据集，一般分为训练集、验证集、测试集

"""_summary_
本文件使用前需要你提供如何格式放置的数据
root_imgdir
|
|----class1
|----class2
|----class3
|----class4
|----class5
|----classm
|----classn

"""

root_imgpath = "path_to/root_imgdir"  # 一直到root_imgdir的绝对路径

def split_data(root_path):
    pass

split_data(root_path=root_imgpath)