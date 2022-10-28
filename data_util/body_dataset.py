# *_*coding:utf-8 *_*
import warnings
import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import drop_point

warnings.filterwarnings('ignore')


def torch_pc_normalize(pc: torch.Tensor):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
    pc = pc / m
    return pc


class body_dataset(Dataset):
    def __init__(self,
                 point_size=1024,
                 train: bool = True,
                 drop_num=None
                 ):

        train_file = h5.File("data/trainset.h5")
        # test_file = h5.File( "data/testset.h5")
        if train:
            dataset = train_file["xyz2"]
        else:
            dataset = train_file["xyz2"][7::50]

        # 标志是否打乱数据集
        self.disorder = True
        pair = []
        self.drop_num = drop_num
        self.point_size = point_size

        # TODO:还没有实现drop num
        if drop_num is not None:
            self.disorder = False
            print("drop {} points".format(drop_num))

        total_data = dataset
        total_data = total_data[:, :point_size, :]

        for idx in range(0, total_data.shape[0]-1, 2):
            pair.append([total_data[idx], total_data[idx + 1]])

        self.pair = np.array(pair)

    def __getitem__(self, index):
        t = torch.tensor(self.pair[index])
        # t = torch_pc_normalize(t)

        # TODO:drop num 还没有实现
        if self.drop_num is not None:  # 有问题
            s1, s2 = drop_point(t[0], t[1], self.point_size, self.drop_num)
            return s1, s2, s2
        return t[0], t[1], t[1]  # 为了保持各个数据集的读取方式一致所以加了第三个数据

    def __len__(self):
        return len(self.pair)


if __name__ == '__main__':
    dataset = cloth_dataset()
    x, y, z = dataset.__getitem__(33)
    print(dataset.__len__())
    print(x)
    print(x.shape)
