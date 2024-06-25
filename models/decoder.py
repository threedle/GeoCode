import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, emb_dims, output_channels, increase_network_size, bn, use_regression):
        super(Decoder, self).__init__()
        self.emb_dims = emb_dims

        if increase_network_size:
            self.linear_out_features_1 = 512
            self.linear_out_features_2 = 256
        else:
            self.linear_out_features_1 = 128
            self.linear_out_features_2 = 64
        self.bn = bn

        if use_regression:
            output_channels = 2

        self.linear1 = nn.Linear(self.emb_dims * 2, self.linear_out_features_1, bias=False)  # args.emb_dims
        if self.bn:
            self.bn6 = nn.BatchNorm1d(self.linear_out_features_1)
        else:
            self.bn6 = nn.InstanceNorm1d(self.linear_out_features_1)
        self.dp1 = nn.Dropout(p=0.5)  # args.dropout
        self.linear2 = nn.Linear(self.linear_out_features_1, self.linear_out_features_2)
        if self.bn:
            self.bn7 = nn.BatchNorm1d(self.linear_out_features_2)
        else:
            self.bn7 = nn.InstanceNorm1d(self.linear_out_features_2)
        self.dp2 = nn.Dropout(p=0.5)  # args.dropout

        self.linear3 = nn.Linear(self.linear_out_features_2, output_channels)

    def forward(self, x):
        if self.bn:
            x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        else:
            x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x = self.dp1(x)
        if self.bn:
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        else:
            x = F.leaky_relu(self.linear2(x), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class DecodersNet(nn.Module):
    def __init__(self, output_channels, increase_network_size, regression_params=None):
        """
        output_channels - array containing the number of classes per parameter (including visibility label if exists)
        """
        # there is no Parallel module in torch (and there is no reason for one to exist)
        # refer to https://github.com/pytorch/pytorch/issues/36459
        super(DecodersNet, self).__init__()
        if increase_network_size:
            self.emb_dims = 1024
        else:
            self.emb_dims = 64
        self.bn = True
        fan_out_list = []
        for i, param_output_channels in enumerate(output_channels):
            use_regression = False
            if regression_params:
                use_regression = regression_params[i]
            fan_out_list.append(Decoder(self.emb_dims, param_output_channels, increase_network_size, self.bn, use_regression))
        self.fan_out = nn.ModuleList(fan_out_list)
        self.initialize_weights(self)

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

    def decode(self, embedding):
        param_outs = []
        for net in self.fan_out:
            param_outs.append(net(embedding))

        x = torch.cat(param_outs, dim=1)
        return x


# used for comparison to "Shape Synthesis from Sketches via Procedural Models and Convolutional Networks"
class DecodersNetAlex(nn.Module):
    def __init__(self, output_channels):
        """
        output_channels - array containing the number of classes per parameter (including visibility label if exists)
        """
        # there is no Parallel module in torch (and there is no reason for one to exist)
        # refer to https://github.com/pytorch/pytorch/issues/36459
        super(DecodersNetAlex, self).__init__()
        self.emb_dims = 4096
        fan_out_list = []
        for param_output_channels in output_channels:
            fan_out_list.append(nn.Linear(4096, param_output_channels))
        self.fan_out = nn.ModuleList(fan_out_list)
        self.initialize_weights(self)

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

    def decode(self, embedding):
        param_outs = []
        for net in self.fan_out:
            param_outs.append(net(embedding))

        x = torch.cat(param_outs, dim=1)
        return x
