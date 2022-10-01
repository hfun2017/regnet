from data_util.cloth_dataset import cloth_dataset
from data_util.TPS3d_dataset import TPS3d_dataset
from utils import *

DATASET = TPS3d_dataset(512, 100, 0.6, 50)
data= DATASET.__getitem__(1)
pc1,pc2,_=data
print(pc1.size())
DATASET = cloth_dataset(512, 400, False, True,50)
data= DATASET.__getitem__(1)
pc1,pc2,_=data
print(pc1.size())

