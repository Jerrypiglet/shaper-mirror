import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy
from .functions import hungarian_matching, gather_masks






class ProposalLoss(nn.Module):
    """Part segmentation loss"""

    def __init__(self):
        super(ProposalLoss, self).__init__()

    def forward(self, preds, labels, label_key='ins_seg_loss'):
        ins_seg_logit = preds["mask_output"]
        ins_seg_logit = ins_seg_logit[:,0,:]
        batch_size  = ins_seg_logit.shape[0]
        num_points = ins_seg_logit.shape[1]
        ins_seg_logit = ins_seg_logit.view((batch_size, num_points))
        ins_seg_label = labels["ins_seg_label"] #B x K x N

        ins_seg_label,_ = torch.max(ins_seg_label, 1)
        ins_seg_label /= (torch.sum(ins_seg_label, 1, True)+1e-8)

        proposal_loss = -1 *  ins_seg_label * F.log_softmax(ins_seg_logit,1)

        finish_label,_ = torch.max(ins_seg_label, 1)
        finish_label /= (finish_label+1e-12)
        finish_logit = preds['global_output'].view((batch_size,))



        conf_loss = F.binary_cross_entropy_with_logits(finish_logit, finish_label)

        loss_dict = {
            "proposal_loss": torch.sum(proposal_loss)/batch_size,
            'finish_loss': conf_loss
        }

        return loss_dict




