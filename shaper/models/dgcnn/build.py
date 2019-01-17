from .dgcnn_cls import DGCNNCls
from ..loss import ClsLoss, DGCNNPartSegLoss 
from ..metric import ClsAccuracy, PartSegMetric

from .dgcnn_part_seg import DGCNNPartSeg 

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
        )
        loss_fn = DGCNNPartSegLoss(cfg.MODEL.DGCNN.REG_WEIGHT,
                               cfg.MODEL.DGCNN.CLS_LOSS_WEIGHT,
                               cfg.MODEL.DGCNN.SEG_LOSS_WEIGHT)
        metric_fn = PartSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
    else:
        raise NotImplementedError()
  
    return net, loss_fn, metric_fn
