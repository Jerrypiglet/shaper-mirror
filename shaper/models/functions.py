import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def hungarian_matching( pred_x, gt_x ):
    '''
    Perform Hungarian Matching between predicted instance masks and GT instance masks
    pred_x: B x NUM_PRED_INS_MASK x NUM_POINT
    gt_x: B x NUM_GT_INS_MASK x NUM_POINT
    return matching_idx : B x NUM_PRED_INS_MASK
    '''
    pred_x = pred_x.cpu().detach().numpy()
    gt_x = gt_x.cpu().detach().numpy()
    batch_size = gt_x.shape[0]
    num_pred_ins_mask = pred_x.shape[1]
    matching_score = np.matmul(gt_x,np.transpose(pred_x,axes=[0,2,1])) # B x num_gt_ins_mask x num_pred_ins_mask
    matching_score = 1-np.divide(matching_score, np.expand_dims(np.sum(pred_x,2),1)+np.sum(gt_x,2,keepdims=True)-matching_score+1e-8)
    matching_idx = -np.ones((batch_size, num_pred_ins_mask), dtype=np.int32)
    curnmasks = np.sum((np.sum(gt_x, 2) > 0).astype(np.int32),1 )
    for i, curnmask  in enumerate(curnmasks):
        if curnmask == 0:
            continue
        row_ind, col_ind = linear_sum_assignment(matching_score[i, :curnmask, :curnmask*2+3])
        matching_idx[i,col_ind]=row_ind
    return matching_idx


def gather_masks(masks, index):
    """Gather xyz of masks according to indices

    Args:
        masks: (batch_size, num_masks, num_points)
        index: (batch_size, num_to_gather)

    Returns:
        new_masks (torch.Tensor): (batch_size, channels, num_centroids)

    """
    batch_size = masks.size(0)
    num_points = masks.size(2)
    num_to_gather = index.size(1)
    index_expand = index.unsqueeze(2).expand(batch_size, num_to_gather, num_points)
    masks = masks.gather(1, index_expand)
    return masks
