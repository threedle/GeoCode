import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, increase_network_size=False, normalize_embeddings=False):
        super(DGCNN, self).__init__()
        #### hard coded params ####
        # args.k = 20
        ###########################
        # self.args = args
        self.k = 20  # args.k
        self.increase_network_size = increase_network_size
        self.normalize_embeddings = normalize_embeddings
        if self.increase_network_size:
            self.channels_layer_1 = 64
            self.channels_layer_2 = 64
            self.channels_layer_3 = 128
            self.channels_layer_4 = 256
            self.emb_dims = 1024
        else:
            self.channels_layer_1 = 16
            self.channels_layer_2 = 16
            self.channels_layer_3 = 32
            self.channels_layer_4 = 64
            self.emb_dims = 64

        self.bn = True
        if self.bn:
            self.bn1 = nn.BatchNorm2d(self.channels_layer_1)
            self.bn2 = nn.BatchNorm2d(self.channels_layer_2)
            self.bn3 = nn.BatchNorm2d(self.channels_layer_3)
            self.bn4 = nn.BatchNorm2d(self.channels_layer_4)
        else:
            self.bn1 = nn.InstanceNorm2d(self.channels_layer_1)
            self.bn2 = nn.InstanceNorm2d(self.channels_layer_2)
            self.bn3 = nn.InstanceNorm2d(self.channels_layer_3)
            self.bn4 = nn.InstanceNorm2d(self.channels_layer_4)

        self.conv1 = nn.Sequential(nn.Conv2d(6, self.channels_layer_1, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.channels_layer_1 * 2, self.channels_layer_2, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(self.channels_layer_2 * 2, self.channels_layer_3, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(self.channels_layer_3 * 2, self.channels_layer_4, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        if self.bn:
            self.bn5 = nn.BatchNorm1d(self.emb_dims)  # args.emb_dims
        else:
            self.bn5 = nn.InstanceNorm1d(self.emb_dims) # args.emb_dims

        self.conv5 = nn.Sequential(nn.Conv1d(self.channels_layer_4 * 2, self.emb_dims, kernel_size=1, bias=False),  # args.emb_dims
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        # net.apply(init_weights)

    @staticmethod
    def initialize_weights(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        batch_size = x.size(0)
        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        enc = torch.cat((x1, x2), 1)

        if self.normalize_embeddings:
            enc = F.normalize(enc, dim=1)

        return enc
