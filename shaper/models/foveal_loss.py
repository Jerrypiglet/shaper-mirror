import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy, pdist
from shaper.data.datasets.geometry_utils import render_pts_with_label
from .functions import hungarian_matching, gather_masks






#class ProposalLoss(nn.Module):
#    """Part segmentation loss"""
#
#    def __init__(self):
#        super(ProposalLoss, self).__init__()
#
#    def forward(self, preds, labels, label_key='ins_seg_label', suffix=''):
#        ins_seg_logit = preds["mask_output"]
#        ins_seg_logit = ins_seg_logit[:,0,:]
#        batch_size  = ins_seg_logit.shape[0]
#        num_points = ins_seg_logit.shape[1]
#        ins_seg_logit = ins_seg_logit.view((batch_size, num_points))
#        ins_seg_label = labels[label_key] #B x K x N
#        viewed_mask = (labels['viewed_mask'] > 0).float()
#        ins_seg_label *= (1-viewed_mask)
#
#        #pts = labels['points']
#        #norms = torch.sum(pts**2, 1, keepdim=True)
#        #norms = norms * ins_seg_label
#        #maxnorms, maxnorms_indices = torch.max(norms,2, keepdim=True)
#        #ins_seg_label = ins_seg_label * (maxnorms == norms).float()
#
#        ins_seg_label,_ = torch.max(ins_seg_label, 1)
#        #ins_seg_label batch_size x num_point gt
#        ins_seg_label /= (torch.sum(ins_seg_label, 1, True)+1e-8)
#
#        #ins_seg_logit batch_size x num_point pred
#        proposal_loss = -1 *  ins_seg_label * F.log_softmax(ins_seg_logit,1)
#
#        finish_label,_ = torch.max(ins_seg_label, 1)
#        labels['finish_label']=finish_label
#        finish_label /= (finish_label+1e-12)
#        finish_logit = preds['global_output'].view((batch_size,))
#
#
#
#        conf_loss = F.binary_cross_entropy_with_logits(finish_logit, finish_label)
#
#        loss_dict = {
#            "proposal_loss"+suffix: torch.sum(proposal_loss)/batch_size,
#            'finish_loss'+suffix: conf_loss
#        }
#
#        return loss_dict




class ProposalLoss(nn.Module):
    """Part segmentation loss"""

    def __init__(self):
        super(ProposalLoss, self).__init__()

    def forward(self, preds, labels, label_key='ins_seg_label', suffix='', finish_weight=1):
        ins_seg_logit = preds["mask_output"]
        proposal_mask = ins_seg_logit[:,0,:]
        radius_mask = ins_seg_logit[:,1:4,:]
        radius_mask = torch.transpose(radius_mask, 2, 1)
        radius_mask = torch.exp(radius_mask)
        rotation_mask = ins_seg_logit[:,4:10,:]
        rotation_mask = torch.transpose(rotation_mask, 2, 1)
        batch_size  = proposal_mask.shape[0]
        num_point = proposal_mask.shape[1]
        ins_seg_label = labels[label_key] #B x K x N
        num_gt_masks = ins_seg_label.shape[1]
        #viewed_mask = (labels['viewed_mask'] > 0).float()

        #distances = pdist(labels['points'])
        #distances = distances.unsqueeze(1).expand(-1,num_gt_masks,-1,-1)
        #distances = distances * ins_seg_label.unsqueeze(2)
        #distances,_ = torch.max(distances, -1)
        #active = (torch.sum(ins_seg_label, 2, keepdim=True) > 0).float()
        #distances = (distances+1e-4)/(active+1e-8)
        #distances, _ =torch.min(distances, 1)
        distances = labels['radius']

        #ins_seg_label = ins_seg_label* (1-viewed_mask)
        ins_seg_label,_ = torch.max(ins_seg_label, 1)
        radius_loss = (radius_mask - distances.detach())**2 * ins_seg_label.unsqueeze(-1).detach()

        radius_loss =  torch.sum(radius_loss, 1) / (1e-12+torch.unsqueeze(torch.sum(ins_seg_label, 1), -1))


        rotation_loss =torch.zeros((1,))
        rotation_gt =labels['rotation']
        rotation_mask = torch.reshape(rotation_mask, (batch_size, num_point, 3,2))
        v1 = rotation_mask[:,:,:,0]
        v1 = v1/torch.sum(v1**2, -1, keepdim=True)**0.5
        eigengap = distances[:,:,0]/(distances[:,:,1]+1e-12)
        eigengap = torch.clamp(eigengap, max=10)
        rotation_loss = -torch.sum( v1 * rotation_gt[:,:,:,0] , -1) ** 2 * ins_seg_label.detach() * (eigengap.detach() - 1)

        v2 = rotation_mask[:,:,:,1]
        v2 = v2 - torch.sum(v1*v2,-1, keepdim=True) * v1
        v2 = v2 / torch.sum(v2**2, -1, keepdim=True)**0.5
        eigengap = distances[:,:,1]/(distances[:,:,2]+1e-12)
        eigengap = torch.clamp(eigengap, max=10)
        rotation_loss -= torch.sum( v2 * rotation_gt[:,:,:,1] , -1) ** 2 * ins_seg_label * (eigengap - 1)
        rotation_loss2=rotation_loss
        rotation_loss = torch.sum(rotation_loss,1) / (1e-12+torch.sum(ins_seg_label,1))


        v3 = torch.cross(v1,v2)

        rotation_mask = torch.cat([rotation_mask, v3.unsqueeze(-1)] , -1)
        labels['rotation_mask']=rotation_mask







        #ins_seg_label batch_size x num_point gt
        ins_seg_label = ins_seg_label / (torch.sum(ins_seg_label, 1, True)+1e-8)
        #ins_seg_logit batch_size x num_point pred
        proposal_loss = -1 *  ins_seg_label * F.log_softmax(proposal_mask,1)

        #ins_seg_label = labels[label_key] #B x K x N
        #finish_label, _ = torch.max(torch.sum(ins_seg_label*(1-viewed_mask),2)/(torch.sum(ins_seg_label,2)+1e-12), 1)
        #labels['finish_label']=finish_label
        #finish_label /= (finish_label+1e-12)
        #finish_logit = preds['global_output'].view((batch_size,))
        #finish_logit = torch.sigmoid(finish_logit)
        #conf_loss = (finish_logit - finish_label)**2
        #conf_loss = torch.sum(conf_loss)/batch_size



        loss_dict = {
            "proposal_loss"+suffix: 1*torch.sum(proposal_loss)/batch_size,
            'rotation_loss'+suffix: torch.sum(rotation_loss)/batch_size,
            'radius_loss'+suffix: 1.0*torch.sum(radius_loss)/batch_size,
            #'finish_loss'+suffix: finish_weight*conf_loss
        }

        return loss_dict




