from .dgcnn_cls import DGCNNCls
from ..loss import ClsLoss
from ..metric import ClsAccuracy


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
    else:
        raise NotImplementedError()

    return net, loss_fn, metric_fn
