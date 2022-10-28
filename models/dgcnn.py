import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(self, enc_emb_dim=128, enc_glb_dim=1024, k_nn=20):
        super(Encoder, self).__init__()
        self.k = k_nn
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.GroupNorm(32,512)
        self.bn6 = nn.GroupNorm(32,512)
        self.bn7 = nn.GroupNorm(32,256)
        self.bn8 = nn.GroupNorm(32,enc_emb_dim)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),self.bn1,nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),self.bn2,nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),self.bn3,nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),self.bn4,nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, enc_glb_dim//2, kernel_size=1, bias=False),self.bn5,nn.LeakyReLU(negative_slope=0.2))
        self.mlp = nn.Sequential(
            nn.Conv1d(64+64+128+256+enc_glb_dim, 512, 1),
            self.bn6,
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            self.bn7,
            nn.ReLU(),
            nn.Conv1d(256, enc_emb_dim, 1),
            self.bn8,
            nn.ReLU())

    @staticmethod
    def _get_graph_feature(x, k=20, idx=None):

        def knn(x, k):
            inner = -2*torch.matmul(x.transpose(2, 1), x)
            xx = torch.sum(x**2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            idx = pairwise_distance.topk(k=k, dim=-1)[1]
            return idx

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)
        device = idx.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def forward(self, transpose_xyz):
        """
        :param transpose_xyz: [batch_size, 3, 1024]
        :return: [B,128,1024],[B,1024,1]
        """
        x = transpose_xyz
        batch_size = x.size(0)
        num_points=x.size(2)
        x = self._get_graph_feature(x, self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = self._get_graph_feature(x1, self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = self._get_graph_feature(x2, self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = self._get_graph_feature(x3, self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1)
        local_concat = x
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        global_vector = x
        repeat_glb_feat = global_vector.unsqueeze(-1).expand(batch_size, global_vector.shape[1], num_points)
        x = torch.cat((local_concat, repeat_glb_feat), 1)
        embedding_feat = self.mlp(x)
        return embedding_feat #, global_vector.unsqueeze(-1)

if __name__ == '__main__':
    dgcnn = Encoder()
    t = torch.rand([8, 1024, 3])
    a,b=dgcnn(t)
    print(a.size())
    print(b.size())
