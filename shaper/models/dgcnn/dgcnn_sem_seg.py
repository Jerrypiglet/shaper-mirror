"""DGCNN
References:
    @article{dgcnn,
      title={Dynamic Graph CNN for Learning on Point Clouds},
      author={Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon},
      journal={arXiv preprint arXiv:1801.07829},
      year={2018}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy
from shaper.nn import MLP, SharedMLP, Conv1d, Conv2d
from shaper.models.dgcnn.functions import get_edge_feature
from shaper.models.dgcnn.modules import EdgeConvBlock
from shaper.nn.init import set_bn


# -----------------------------------------------------------------------------
# DGCNN for part segmentation
# -----------------------------------------------------------------------------
class DGCNNSemSeg(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_conv_channels=((64, 64), (64, 64), (64, 64)),
                 inter_channels=1024,
                 global_channels=(512, 256,),
                 k=20,
                 dropout_prob=0.3):
        super(DGCNNSemSeg, self).__init__()

        self.in_channels = in_channels
        self.k = k

        self.mlp_edge_conv = nn.ModuleList()
        use_centroids = False
        for out in edge_conv_channels:
            self.mlp_edge_conv.append(EdgeConvBlock(in_channels, out, k, use_centroids=use_centroids))
            in_channels = out[-1]
            use_centroids = True

        concat_channels = sum([item[-1] for item in edge_conv_channels])
        self.mlp_local = Conv1d(concat_channels, inter_channels, 1)

        self.mlp_seg = SharedMLP(inter_channels + concat_channels, global_channels)
        self.dropout = nn.Dropout(dropout_prob)
        self.seg_logit = nn.Conv1d(global_channels[-1], out_channels, 1, bias=True)

        self.init_weights()
        set_bn(self, momentum=0.01)

    def forward(self, data_batch):
        end_points = {}
        x = data_batch["points"]
        knn_ind = data_batch.get('knn_ind', None)

        num_points = x.shape[2]

        # edge convolution for point cloud
        features = []
        for edge_conv in self.mlp_edge_conv:
            x = edge_conv(x, knn_ind)
            features.append(x)

        concat_features = torch.cat(features, dim=1)

        # local mlp 
        x = self.mlp_local(concat_features)
        global_features, max_indice = torch.max(x, 2)
        # end_points['key_point_inds'] = max_indice

        global_features_expand = global_features.unsqueeze(2).expand(-1, -1, num_points)
        x = torch.cat([global_features_expand, concat_features], dim=1)

        # mlp_seg & conv_seg
        x = self.mlp_seg(x)
        x = self.dropout(x)
        seg_logit = self.seg_logit(x)
        preds = {
            'seg_logit': seg_logit,
        }
        preds.update(end_points)

        return preds

    def init_weights(self):
        nn.init.xavier_uniform_(self.seg_logit.weight)
        nn.init.zeros_(self.seg_logit.bias)


class DGCNNSegLoss(nn.Module):
    """DGCNN part segmentation loss with optional regularization loss"""

    def __init__(self):
        super(DGCNNSegLoss, self).__init__()

    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"]
        seg_loss = F.cross_entropy(seg_logit, seg_label)
        loss_dict = {
            "seg_loss": seg_loss,
        }

        return loss_dict


if __name__ == "__main__":
    batch_size = 4
    in_channels = 3
    num_points = 1024
    num_classes = 40

    data = torch.rand(batch_size, in_channels, num_points).cuda()

    dgcnn = DGCNNSemSeg(in_channels, num_classes).cuda()
    out_dict = dgcnn({"points": data})
    for k, v in out_dict.items():
        print('DGCNN:', k, v.shape)
