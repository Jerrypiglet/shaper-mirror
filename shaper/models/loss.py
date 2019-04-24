import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy
from .functions import hungarian_matching, gather_masks

class ClsLoss(nn.Module):
    """Classification loss with optional label smoothing

    Attributes:
        label_smoothing (float or 0): the parameter to smooth labels

    """

    def __init__(self, label_smoothing=0):
        super(ClsLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, preds, labels):
        cls_logit = preds["cls_logit"]
        cls_label = labels["cls_label"]
        if self.label_smoothing > 0:
            cls_loss = smooth_cross_entropy(cls_logit, cls_label, self.label_smoothing)
        else:
            cls_loss = F.cross_entropy(cls_logit, cls_label)
        loss_dict = {
            'cls_loss': cls_loss,
        }
        return loss_dict


class PartSegLoss(nn.Module):
    """Part segmentation loss"""

    def __init__(self):
        super(PartSegLoss, self).__init__()

    def forward(self, preds, labels):
        seg_logit = preds["seg_logit"]
        seg_label = labels["seg_label"]
        seg_loss = F.cross_entropy(seg_logit, seg_label)
        loss_dict = {
            "seg_loss": seg_loss
        }

        return loss_dict




class PartInsSegLoss(nn.Module):
    """Part segmentation loss"""

    def __init__(self):
        super(PartInsSegLoss, self).__init__()

    def forward(self, preds, labels, label_key='ins_seg_label', knns=None, suffix=''):
        ins_seg_logit = preds["mask_output"]
        ins_seg_logit = F.softmax(ins_seg_logit,1)
        ins_seg_label = labels[label_key]

        batch_size = ins_seg_logit.shape[0]
        matching_idx = torch.tensor(hungarian_matching(ins_seg_logit, ins_seg_label)).cuda().long() #batch_size x num_gt_ins_mask
        active = (matching_idx>=0).float()
        matching_idx[matching_idx < 0] = 0
        gathered_gt_masks = gather_masks(ins_seg_label, matching_idx)


        matching_score = torch.sum(gathered_gt_masks * ins_seg_logit,2)
        union = torch.sum(gathered_gt_masks,2) + torch.sum(ins_seg_logit,2) - matching_score


        iou = matching_score/(union+1e-12)
        iou *= active
        count_active = torch.sum(active,1)
        per_shape_iou = torch.sum(iou, 1)/(count_active+1e-12)
        ins_seg_loss = -1*torch.sum(per_shape_iou)/(torch.sum(count_active>0).float()+1e-12)
        ins_seg_loss = -1*torch.sum(iou)/(torch.sum(active).float()+1e-12)


        conf_logit = preds['global_output']
        conf_loss = F.binary_cross_entropy_with_logits(conf_logit, active)



        loss_dict = {
            "ins_seg_loss"+suffix: ins_seg_loss,
            'conf_loss'+suffix:conf_loss
        }


        #for b in range(batch_size):
        #    label = ins_seg_label[b]
        #    active_label =  torch.nonzero(torch.sum(label,1))
        #    k=0
        #    for knn in knns:
        #        inside_connections=0
        #        outside_connections=0
        #        for a in active_label:
        #            mask = label[a[0]]
        #            for m in torch.nonzero(mask):
        #                neighbors = knn[b,m[0],:]
        #                for neighbor in neighbors:
        #                    if mask[neighbor]:
        #                        inside_connections+=1
        #                    else:
        #                        outside_connections+=1
        #        print(b,k, inside_connections, outside_connections, inside_connections*1.0/(1e-12+outside_connections+inside_connections))
        #        k+=1
        #    print(b, per_shape_iou[b])
        #exit(0)

        return loss_dict


