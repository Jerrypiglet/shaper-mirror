import torch
import torch.nn as nn

from shaper.nn import MLP, SharedMLP, Conv1d, Conv2d
from shaper.models.dgcnn.functions import get_edge_feature
from shaper.models.dgcnn.modules import EdgeConvBlock
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
                 num_seg_class,
                 edge_conv_channels=((64, 64), (64, 64), (64,)),
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
        self.num_gpu = torch.cuda.device_count()

        #input transform
        if self.with_transform:
            self.transform_input = TNet(in_channels, in_channels, k=k)
        
        self.mlp_edge_conv = nn.ModuleList()
        for out in edge_conv_channels:
            self.mlp_edge_conv.append(EdgeConvBlock(in_channels, out, k))
            in_channels = out[-1]
        

        self.lable_conv = Conv2d(16, 64, [1,1])

        self.mlp_local = Conv1d(sum([item[-1] for item in edge_conv_channels]), inter_channels, 1)

        mlp_in_channels = inter_channels + edge_conv_channels[-1][-1] + sum([item[-1] for item in edge_conv_channels])
        self.mlp_seg = SharedMLP(mlp_in_channels, global_channels[:-1], dropout=dropout_prob)
        self.conv_seg = Conv1d(global_channels[-2], global_channels[-1], 1)
        self.seg_logit = nn.Conv1d(global_channels[-1], num_seg_class, 1,bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        #specify computing process
        end_points = {}
        x = data_batch["points"]

        num_point = x.shape[2]
      
        cls_label = data_batch["cls_label"]
        batch_size = cls_label.size()[0]
        num_classes = 16
        # print("output size is {}, data shape is {}, label shape is {}\n".format(self.out_channels, x.size(), cls_label.size()))
        # print("number of gpu is {}".format(torch.cuda.device_count()))
        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input
        # print("the size of x is {}".format(x.size()))


	# edge convolution for point cloud         
        features = []
        for edge_conv in self.mlp_edge_conv:
            x = edge_conv(x)
            features.append(x)
        # print("the size of x after edge conv is {}".format(x.size()))

	# concatenate all the feature from each edge convolutional layer 
        x = torch.cat(features, dim=1)
   
        # go through local mlp 
        x = self.mlp_local(x)
        x, max_indice = torch.max(x, 2)

        end_points['key_point_inds'] = max_indice

        # use classification label
        # print("the real cls label {}".format(cls_label))
        with torch.no_grad():
            I = torch.eye(16, dtype=x.dtype, device=x.device)
            one_hot = I[cls_label]
            one_hot_expand = one_hot.view(batch_size, num_classes, 1, 1)

        one_hot_expand = self.lable_conv(one_hot_expand)
        
        # print("size of x is {}, size of one_hot_expand is {}".format(x.size(), one_hot_expand.size()))
        # concatenate information from point cloud and label
        one_hot_expand = one_hot_expand.view(batch_size, -1)
        out_max = torch.cat([x, one_hot_expand], dim=1)
        out_max = out_max.unsqueeze(2).expand(-1, -1, num_point)

        cat_features = torch.cat(features, dim=1)
        # print("size of x is {}, size of cat_features is {}".format(out_max.size(), cat_features.size()))
        x = torch.cat([out_max, cat_features], dim=1)
        x = self.mlp_seg(x)
        x = self.conv_seg(x)
        seg_logit = self.seg_logit(x)
        preds = {
            'seg_logit': seg_logit
        }
        preds.update(end_points)

        # print("seg_logit is \n {}, size is {}".format(seg_logit, seg_logit.size()))
        # print("#"*50)
        # print("label is \n {}, size is {}".format(data_batch["seg_label"], data_batch["seg_label"].size()))
        return preds
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.seg_logit.weight)
        nn.init.zeros_(self.seg_logit.bias)

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

