import torch
import torch.nn as nn

from shaper.nn import MLP, SharedMLP, Conv1d, Conv2d
from shaper.models.dgcnn.functions import get_edge_feature
from shaper.models.dgcnn.modules import EdgeConvBlockV2
from shaper.nn.init import set_bn

class TNet(nn.Module):
    """Transformation Network for DGCNN

    Structure: input -> [EdgeFeature] -> [EdgeConv]s -> [EdgePool] -> features -> [MLP] -> local features
    -> [MaxPool] -> global features -> [MLP] -> [Linear] -> logits

    Args:
        conv_channels (tuple of int): the numbers of channels of edge convolution layers
        k: the number of neareast neighbours for edge feature extractor

    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 conv_channels=(64, 128),
                 local_channels=(1024,),
                 global_channels=(512, 256),
                 k=20):
        super(TNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.edge_conv = SharedMLP(2 * in_channels, conv_channels, ndim=2)
        self.mlp_local = SharedMLP(conv_channels[-1], local_channels)
        self.mlp_global = MLP(local_channels[-1], global_channels)
        self.linear = nn.Linear(global_channels[-1], self.in_channels * out_channels, bias=True)

        self.init_weights()

    def forward(self, x):
        """TNet forward

        Args:
            x (torch.Tensor): (batch_size, in_channels, num_points)

        Returns:
            torch.Tensor: (batch_size, out_channels, in_channels)

        """
        x = get_edge_feature(x, self.k)  # (batch_size, 2 * in_channels, num_points, k)
        x = self.edge_conv(x)
        x, _ = torch.max(x, 3)  # (batch_size, edge_channels[-1], num_points)
        x = self.mlp_local(x)
        x, _ = torch.max(x, 2)  # (batch_size, local_channels[-1], num_points)
        x = self.mlp_global(x)
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.in_channels)
        I = torch.eye(self.out_channels, self.in_channels, device=x.device)
        x = x.add(I)  # broadcast first dimension
        return x

    def init_weights(self):
        # set linear transform be 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class DGCNNPartSeg(nn.Module):
    """DGCNN for part segmentation
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_conv_channels=(64, 64, 64, 64, 64, 64),
                 inter_channels= 1024,
                 global_channels=(256, 256, 128),
                 k=20,
                 dropout_prob=0.6,
                 with_transform=True):
        super(DGCNNPartSeg, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.with_transform = with_transform

        #input transform
        if self.with_transform:
            self.transform_input = TNet(inchannels, inter_channels, k=k)
        
        self.mlp_edge_conv = nn.ModuleList()
        for out_channels in edge_conv_channels:
            self.mlp_edge_conv.append(EdgeConvBlockV2(in_channels, out_channels, k))
            in_channels = out_channels
        
        self.mlp_local = Conv1d(sum(edge_conv_channels)/2, inter_channels, 1)
        self.mlp_global = MLP(inter_channels, global_channels, dropout=dropout_prob)

        self.classifier = nn.Linear(global_channels[-1])

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        #specify computing process
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

if __name__ == "__main__":
    batch_size = 4
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points)
    transform = TNet()
    out = transform(data)
    print('TNet: ', out.size())

    dgcnn = DGCNNPartSeg(in_channels, num_classes, with_transform=False)
    out_dict = dgcnn({"points": data})
    for k, v in out_dict.items():
        print('DGCNN:', k, v.shape)