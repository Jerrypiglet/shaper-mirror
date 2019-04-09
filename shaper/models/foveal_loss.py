import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy
from .functions import hungarian_matching, gather_masks





class PartInsSegLoss(nn.Module):
    """Part segmentation loss"""

    def __init__(self):
        super(PartInsSegLoss, self).__init__()

    def forward(self, preds, labels):
        ins_seg_logit = preds["mask_output"]
        ins_seg_logit = F.softmax(ins_seg_logit,1)
        ins_seg_label = labels["ins_seg_label"]
        batch_size = ins_seg_logit.shape[0]
        matching_idx = torch.tensor(hungarian_matching(ins_seg_logit, ins_seg_label)).cuda().long() #batch_size x num_gt_ins_mask
        active = (matching_idx>=0).float()
        matching_idx[matching_idx < 0] = 0
        gathered_gt_masks = gather_masks(ins_seg_label, matching_idx)


        matching_score = torch.sum(gathered_gt_masks * ins_seg_logit,2)
        union = torch.sum(gathered_gt_masks,2) + torch.sum(ins_seg_logit,2) - matching_score


        iou = matching_score/(union+1e-12)
        iou *= active


        conf_logit = preds['global_output']
        conf_loss = F.binary_cross_entropy_with_logits(conf_logit, active)



        loss_dict = {
            "ins_seg_loss": -1*torch.sum(iou)/torch.sum(active),
            'conf_loss':conf_loss
        }

        return loss_dict




class ProposalLoss(nn.Module):
    """Part segmentation loss"""

    def __init__(self):
        super(PartInsSegLoss, self).__init__()

    def forward(self, preds, labels):
        ins_seg_logit = preds["mask_output"]
        batch_size  = ins_seg_logit.shape[0]
        num_points = ins_seg_losit.shape[2]
        ins_seg_logit = ins_seg_logit.view((batch_size, num_points))
        ins_seg_label = labels["ins_seg_label"] #B x K x N

        ins_seg_label = torch.max(ins_seg_label, 1)
        ins_seg_label /= torch.sum(ins_seg_label, dim=1, keepdim=True)

        proposal_loss = -1 *  ins_seg_label * F.logsoftmax(ins_seg_logit,1)

        finish_label = torch.max(ins_seg_label, 1)
        finish_logit = preds['global_output'].view((batch_size,))

        conf_loss = F.binary_cross_entropy_with_logits(finish_logit, finish_label)

        loss_dict = {
            "proposal_loss": torch.sum(proposal_loss)/batch_size,
            'conf_loss': conf_loss
        }

        return loss_dict




