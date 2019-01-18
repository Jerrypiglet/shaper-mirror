from .pn2_ssg_cls import PointNet2SSGCls
from .pn2_msg_cls import PointNet2MSGCls
from .pn2_ssg_part_seg import PointNet2SSGPartSeg
from .pn2_msg_part_seg import PointNet2MSGPartSeg
from .pn2_ssg_sem_seg import PointNet2SSGSemSeg
from .pn2_msg_sem_seg import PointNet2MSGSemSeg
from ..loss import ClsLoss, PartSegLoss, SemSegLoss
from ..metric import ClsAccuracy, PartSegMetric, SemSegMetric


def build_pointnet2ssg(cfg):
    if cfg.TASK == "classification":
        net = PointNet2SSGCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_centroids=cfg.MODEL.PN2SSG.NUM_CENTROIDS,
            radius=cfg.MODEL.PN2SSG.RADIUS,
            num_neighbours=cfg.MODEL.PN2SSG.NUM_NEIGHBOURS,
            sa_channels=cfg.MODEL.PN2SSG.SA_CHANNELS,
            local_channels=cfg.MODEL.PN2SSG.LOCAL_CHANNELS,
            global_channels=cfg.MODEL.PN2SSG.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.PN2SSG.DROPOUT_PROB,
            use_xyz=cfg.MODEL.PN2SSG.USE_XYZ
        )
        loss_fn = ClsLoss()
        metric_fn = ClsAccuracy()
    elif cfg.TASK == "part_segmentation":
        net = PointNet2SSGPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            num_centroids=cfg.MODEL.PN2SSG.NUM_CENTROIDS,
            radius=cfg.MODEL.PN2SSG.RADIUS,
            num_neighbours=cfg.MODEL.PN2SSG.NUM_NEIGHBOURS,
            sa_channels=cfg.MODEL.PN2SSG.SA_CHANNELS,
            local_channels=cfg.MODEL.PN2SSG.LOCAL_CHANNELS,
            fp_local_channels=cfg.MODEL.PN2SSG.FP_LOCAL_CHANNELS,
            fp_channels=cfg.MODEL.PN2SSG.FP_CHANNELS,
            num_fp_neighbours=cfg.MODEL.PN2SSG.NUM_FP_NEIGHBOURS,
            seg_channels=cfg.MODEL.PN2SSG.SEG_CHANNELS,
            dropout_prob=cfg.MODEL.PN2SSG.DROPOUT_PROB,
            use_xyz=cfg.MODEL.PN2SSG.USE_XYZ
        )
        loss_fn = PartSegLoss()
        metric_fn = PartSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
    elif cfg.TASK == "semantic_segmentation":
        net = PointNet2SSGSemSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            num_centroids=cfg.MODEL.PN2SSG.NUM_CENTROIDS,
            radius=cfg.MODEL.PN2SSG.RADIUS,
            num_neighbours=cfg.MODEL.PN2SSG.NUM_NEIGHBOURS,
            sa_channels=cfg.MODEL.PN2SSG.SA_CHANNELS,
            local_channels=cfg.MODEL.PN2SSG.LOCAL_CHANNELS,
            fp_channels=cfg.MODEL.PN2SSG.FP_CHANNELS,
            num_fp_neighbours=cfg.MODEL.PN2SSG.NUM_FP_NEIGHBOURS,
            seg_channels=cfg.MODEL.PN2SSG.SEG_CHANNELS,
            dropout_prob=cfg.MODEL.PN2SSG.DROPOUT_PROB,
            use_xyz=cfg.MODEL.PN2SSG.USE_XYZ
        )
        loss_fn = SemSegLoss()
        metric_fn = SemSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn


def build_pointnet2msg(cfg):
    if cfg.TASK == "classification":
        net = PointNet2MSGCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            num_centroids=cfg.MODEL.PN2MSG.NUM_CENTROIDS,
            radius_list=cfg.MODEL.PN2MSG.RADIUS,
            num_neighbours_list=cfg.MODEL.PN2MSG.NUM_NEIGHBOURS,
            sa_channels_list=cfg.MODEL.PN2MSG.SA_CHANNELS,
            global_channels=cfg.MODEL.PN2MSG.GLOBAL_CHANNELS,
            dropout_prob=cfg.MODEL.PN2MSG.DROPOUT_PROB,
            use_xyz=cfg.MODEL.PN2MSG.USE_XYZ
        )
        loss_fn = ClsLoss()
        metric_fn = ClsAccuracy()
    elif cfg.TASK == "part_segmentation":
        net = PointNet2MSGPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            num_centroids=cfg.MODEL.PN2MSG.NUM_CENTROIDS,
            radius_list=cfg.MODEL.PN2MSG.RADIUS_LIST,
            num_neighbours_list=cfg.MODEL.PN2MSG.NUM_NEIGHBOURS_LIST,
            sa_channels_list=cfg.MODEL.PN2MSG.SA_CHANNELS_LIST,
            local_channels=cfg.MODEL.PN2MSG.LOCAL_CHANNELS,
            fp_local_channels=cfg.MODEL.PN2MSG.FP_LOCAL_CHANNELS,
            fp_channels=cfg.MODEL.PN2MSG.FP_CHANNELS,
            num_fp_neighbours=cfg.MODEL.PN2MSG.NUM_FP_NEIGHBOURS,
            seg_channels=cfg.MODEL.PN2MSG.SEG_CHANNELS,
            dropout_prob=cfg.MODEL.PN2MSG.DROPOUT_PROB,
            use_xyz=cfg.MODEL.PN2MSG.USE_XYZ
        )
        loss_fn = PartSegLoss()
        metric_fn = PartSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
    elif cfg.TASK == "semantic_segmentation":
        net = PointNet2MSGSemSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            num_centroids=cfg.MODEL.PN2MSG.NUM_CENTROIDS,
            radius_list=cfg.MODEL.PN2MSG.RADIUS_LIST,
            num_neighbours_list=cfg.MODEL.PN2MSG.NUM_NEIGHBOURS_LIST,
            sa_channels_list=cfg.MODEL.PN2MSG.SA_CHANNELS_LIST,
            local_channels=cfg.MODEL.PN2MSG.LOCAL_CHANNELS,
            fp_channels=cfg.MODEL.PN2MSG.FP_CHANNELS,
            num_fp_neighbours=cfg.MODEL.PN2MSG.NUM_FP_NEIGHBOURS,
            seg_channels=cfg.MODEL.PN2MSG.SEG_CHANNELS,
            dropout_prob=cfg.MODEL.PN2MSG.DROPOUT_PROB,
            use_xyz=cfg.MODEL.PN2MSG.USE_XYZ
        )
        loss_fn = PartSegLoss()
        metric_fn = PartSegMetric(cfg.DATASET.NUM_SEG_CLASSES)
    else:
        raise NotImplementedError

    return net, loss_fn, metric_fn
