"""Classification experiments configuration"""

from .base import CN, _C

# public alias
cfg = _C

_C.TASK = "part_segmentation"

_C.TRAIN.VAL_METRIC = "seg_acc"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET.NUM_SEG_CLASSES = 0

# Data augmentation for part segmentation.
_C.TRAIN.SEG_AUGMENTATION = ()

# -----------------------------------------------------------------------------
# PointNet options
# -----------------------------------------------------------------------------
_C.MODEL.POINTNET = CN()

_C.MODEL.POINTNET.STEM_CHANNELS = (64, 128, 128)
_C.MODEL.POINTNET.LOCAL_CHANNELS = (512, 2048)
_C.MODEL.POINTNET.CLS_CHANNELS = (256, 256)
_C.MODEL.POINTNET.SEG_CHANNELS = (256, 256, 128)

_C.MODEL.POINTNET.DROPOUT_PROB_CLS = 0.3
_C.MODEL.POINTNET.DROPOUT_PROB_SEG = 0.2
_C.MODEL.POINTNET.WITH_TRANSFORM = True

_C.MODEL.POINTNET.REG_WEIGHT = 0.032
_C.MODEL.POINTNET.CLS_LOSS_WEIGHT = 0.0
_C.MODEL.POINTNET.SEG_LOSS_WEIGHT = 1.0

# -----------------------------------------------------------------------------
# PN2SSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2SSG = CN()

_C.MODEL.PN2SSG.NUM_CENTROIDS = (512, 128)
_C.MODEL.PN2SSG.RADIUS = (0.2, 0.4)
_C.MODEL.PN2SSG.NUM_NEIGHBOURS = (32, 64)
_C.MODEL.PN2SSG.SA_CHANNELS = ((64, 64, 128), (128, 128, 256))
_C.MODEL.PN2SSG.LOCAL_CHANNELS = (256, 512, 1024)
_C.MODEL.PN2SSG.FP_CHANNELS = ((256, 256), (256, 128), (128, 128, 128))
_C.MODEL.PN2SSG.NUM_FP_NEIGHBOURS = (3, 3, 3)
_C.MODEL.PN2SSG.SEG_CHANNELS = (128,)
_C.MODEL.PN2SSG.DROPOUT_PROB = 0.5
_C.MODEL.PN2SSG.USE_XYZ = True
