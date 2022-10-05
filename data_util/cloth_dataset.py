# *_*coding:utf-8 *_*
import random
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from data_util.TPS3d_dataset import gen_unoise
from utils import exclude_query_ball,drop_point

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class cloth_dataset(Dataset):
    def __init__(self,
                 point_size=512,
                 sample_size=500,
                 use_rgb=False,
                 train: bool = True,
                 drop_num=None,
                 out_liner_num=None,
                 unoise=False,
                 noise=False
                 ):

        # dataset = np.load("data/Lc_left_edge.npy")
        # dataset = np.load("data/hoody_Ll_front.npy")
        dataset = np.load("data/paper_Lc.npy")
        # dataset = np.load("data/tshirt_Ld_front_big.npy")

        pair = []
        self.drop_num = drop_num
        self.point_size = point_size
        # sample_size表示使用数据集中的多少个数据
        self.sample_size = sample_size
        self.use_rgb = use_rgb
        self.train = train
        self.noise = noise
        self.unoise = unoise
        self.out_liner_num = out_liner_num

        total_size = dataset.shape[0]
        if sample_size > total_size:
            sample_size = total_size

        if train:
            total_data = dataset[:sample_size, ...]
        else:
            total_data = dataset[0 - sample_size:, ...]

        if not use_rgb:
            total_data = total_data[..., :3]
        if noise:
            print("adding normal noise ")
        if unoise:
            print("adding uniform noise ")
        if drop_num is not None:
            self.disorder=False
            print("drop {} points".format(drop_num))
        if out_liner_num is not None:
            assert out_liner_num<=92
            self.ball=np.loadtxt("./data/ball.txt",dtype=np.float32)[:out_liner_num,:3]
            self.ball=torch.Tensor(self.ball)/10
            self.disorder=False
            print("outline {} points".format(out_liner_num))

        total_data = total_data[:, :point_size, :]
        for i in range(sample_size):
            total_data[i] = pc_normalize(total_data[i])

        for idx in range(sample_size):
            for idx2 in range(idx + 1, sample_size):
                pair.append([total_data[idx], total_data[idx2]])

        self.pair = np.array(pair)

    def __getitem__(self, index):
        t = torch.tensor(self.pair[index])
        target=t[0]
        source=t[1]
        theta=t[1]      #误
        if self.drop_num is not None:       #有问题
            s1, s2 = drop_point(t[0].clone(), t[1].clone(),self.point_size,self.drop_num)
            return s1, s2, s2
        elif self.out_liner_num is not None:
            target,source=self.outline(target.clone(),source.clone())
            return target,source, theta
        elif self.unoise:
            target=torch.cat([target,target[:50,]],dim=0)
            source=torch.cat([source,gen_unoise()],dim=0)
            return target,source,theta
        elif self.noise:
            target=target+torch.randn(target.size())*0.03
            source=source+torch.randn(source.size())*0.03
            return target,source,theta
        return t[0], t[1], t[1]  # 为了保持各个数据集的读取方式一致所以加了第三个数据

    def __len__(self):
        return len(self.pair)

    def outline(self, target:torch.Tensor,source:torch.Tensor):
        x,y,z = random.randint(-500, 500),random.randint(-500, 500),random.randint(200, 500)
        x,y,z=x/1000.,y/1000.,z/1000.
        shift=self.ball+torch.Tensor([x,y,z])
        target=torch.cat([target,shift],0)
        source=torch.cat([source,source[:self.out_liner_num,:]],0)
        return target,source
