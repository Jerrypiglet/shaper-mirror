import torch
import torch.nn as nn
import torch.nn.functional as F

from shaper.nn.functional import smooth_cross_entropy, pdist
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

    def forward(self, preds, labels, label_key='ins_seg_label', suffix=''):
        ins_seg_logit = preds["mask_output"]
        ins_seg_logit = F.softmax(ins_seg_logit,1)
        ins_seg_label = labels[label_key]

        batch_size = ins_seg_logit.shape[0]
        num_point = ins_seg_logit.shape[2]
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
        conf_logit = torch.sigmoid(conf_logit)
        conf_loss = torch.sum((conf_logit - iou.detach())**2)/batch_size
        #conf_loss = F.binary_cross_entropy_with_logits(conf_logit, active)


        #features = preds['features']
        #num_features = len(features)
        #features = torch.cat([torch.unsqueeze(feature,1) for feature in features], 1) #B x num_features x feature_length x N
        #feature_length = features.shape[2]
        #knns = torch.cat([torch.unsqueeze(knn,1) for knn in preds['knns']],1) #B x num_features x N x num_neighbors
        #active_label_indices = torch.nonzero(torch.sum(ins_seg_label,2)) #(batchno, maskno)
        #masks = ins_seg_label[active_label_indices[:,0], active_label_indices[:,1], :] #num_active_masks x N
        #masked_loss = torch.zeros((num_features,)).cuda()
        #unmasked_loss = torch.zeros((num_features,)).cuda()
        #for maskno in range(len(masks)):
        #        break
        #        mask=masks[maskno] #N
        #        batchno = active_label_indices[maskno,0]
        #        assert batchno >=0 and batchno < batch_size
        #        feature = features[batchno] #num_features x feature_length x N
        #        knn = knns[batchno] #num_features x N x num_neighbors

        #        active_point_indices = torch.nonzero(mask) #(pointno)
        #        num_active_points = active_point_indices.shape[0]
        #        pointsno=  active_point_indices[:,0]
        #        assert  (pointsno >=0).all() and  (pointsno < num_point).all()
        #        active_features = feature[:, :, pointsno] #num_features x feature_length x num_active_points

        #        neighbors = knn[:,pointsno,:] #num_features x num_active_points x num_neighbor neighbors include selfs
        #        num_neighbors = neighbors.shape[-1]
        #        expanded_neighbors = neighbors.view(num_features, num_active_points*num_neighbors)
        #        expanded_neighbors = expanded_neighbors.unsqueeze(1).expand(-1, feature_length, -1)
        #        assert (expanded_neighbors >=0).all() and (expanded_neighbors < num_point).all()
        #        neighbor_features = torch.gather(feature, 2, expanded_neighbors)   #num_features x feature_length x num_active_points * num_neighbors
        #        neighbor_features = neighbor_features.view(num_features, feature_length, num_active_points, num_neighbors) #num_features x feature_length x num_active_points x num_neighbors
        #        neighbor_feature_dists = torch.sum((active_features.unsqueeze(-1) - neighbor_features)**2, 1)**0.5 #num_features x num active_points x num_neighbots

        #        gathered_mask = mask.view(1,1, -1).expand(num_features, num_active_points, -1) #num_feature x num_active_points x N
        #        assert (neighbors >=0).all() and (neighbors < num_point).all()
        #        gathered_mask = torch.gather(gathered_mask, 2,  neighbors) #num_feature x num_active_points x num_neighbors

        #        smallest_gathered_unmasked, _ =  torch.min(neighbor_feature_dists/(1e-6+1-gathered_mask), 2, keepdim=True) #num_features x num_active_points x 1
        #        active_features_pdist = torch.clamp(pdist(active_features), min=0)**0.5 #num_features x num_active_points x num_active_points
        #        largest_masked, _ = torch.max(active_features_pdist,2,keepdim=True) #num_features x num_active_points x 1

        #        masked_loss += torch.mean(torch.mean(torch.clamp(active_features_pdist - smallest_gathered_unmasked, min=0),-1), -1)
        #        unmasked_loss += torch.mean(torch.mean(-1 * torch.clamp(neighbor_feature_dists - largest_masked, max=0), -1), -1)
        #feature_loss = (torch.sum(masked_loss)+torch.sum(unmasked_loss)) / (num_features*len(masks)+1e-6)        #print(unmasked_loss2, pds)


        loss_dict = {
            "ins_seg_loss"+suffix: ins_seg_loss,
            'conf_loss'+suffix:conf_loss*5,
            #'feature_loss'+suffix:feature_loss
        }

        return loss_dict


