from .dgcnn_cls import DGCNNCls
from ..loss import ClsLoss, PartInsSegLoss
from ..foveal_loss import ProposalLoss
from ..metric import ClsAccuracy, PartSegMetric

from .dgcnn_part_seg import DGCNNPartSeg, DGCNNPartSegLoss
from .dgcnn_twobranch import DGCNNTwoBranch


def build_dgcnn(cfg):
    if cfg.TASK == "classification":
        net = DGCNNCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            edge_conv_channels=cfg.MODEL.DGCNN.EDGE_CONV_CHANNELS,
            inter_channels=cfg.MODEL.DGCNN.INTER_CHANNELS,
            global_channels=cfg.MODEL.DGCNN.GLOBAL_CHANNELS,
            k=cfg.MODEL.DGCNN.K,
            dropout_prob=cfg.MODEL.DGCNN.DROPOUT_PROB,
            with_transform=cfg.MODEL.DGCNN.WITH_TRANSFORM,
        )
        loss_fn = ClsLoss(cfg.MODEL.DGCNN.LABEL_SMOOTHING)
        metric_fn = ClsAccuracy()
    elif cfg.TASK == "part_segmentation":
        net = DGCNNPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_class = cfg.DATASET.NUM_CLASSES,
            num_seg_class=cfg.DATASET.NUM_SEG_CLASSES,
            use_bn=cfg.MODEL.NORMALIZATION=='BN',
            use_gn=cfg.MODEL.NORMALIZATION=='GN'
        )
        loss_fn = DGCNNPartSegLoss(cfg.MODEL.DGCNN.REG_WEIGHT,
                               cfg.MODEL.DGCNN.CLS_LOSS_WEIGHT,
                               cfg.MODEL.DGCNN.SEG_LOSS_WEIGHT)
        metric_fn = PartSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
    elif cfg.TASK == "part_instance_segmentation":
        net = DGCNNTwoBranch(
            in_channels = cfg.INPUT.IN_CHANNELS,
            num_global_output = cfg.MODEL.NUM_INS_MASKS,
            num_mask_output = cfg.MODEL.NUM_INS_MASKS,
            use_bn=cfg.MODEL.NORMALIZATION=='BN',
            use_gn=cfg.MODEL.NORMALIZATION=='GN'
        )
        loss_fn = PartInsSegLoss()
        metric_fn = None#PartSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
    elif cfg.TASK == "foveal_part_instance_segmentation":
        proposal_net = DGCNNTwoBranch(
            in_channels = cfg.INPUT.IN_CHANNELS+2,
            num_global_output = 1,
            num_mask_output = 1+cfg.MODEL.META_DATA,
            use_bn=cfg.MODEL.NORMALIZATION=='BN',
            use_gn=cfg.MODEL.NORMALIZATION=='GN'
        )
        segmentation_net = DGCNNTwoBranch(
            in_channels = cfg.INPUT.IN_CHANNELS+cfg.MODEL.META_DATA,
            num_global_output = cfg.MODEL.NUM_INS_MASKS,
            num_mask_output = cfg.MODEL.NUM_INS_MASKS,
            use_bn=cfg.MODEL.NORMALIZATION=='BN',
            use_gn=cfg.MODEL.NORMALIZATION=='GN'
        )
        proposal_loss_fn = ProposalLoss()
        segmentation_loss_fn = PartInsSegLoss()
        metric_fn = None#PartSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
        return [proposal_net, segmentation_net], [proposal_loss_fn, segmentation_loss_fn]
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn
