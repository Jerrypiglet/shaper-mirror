"""Classification experiments configuration"""

from .base import CN, _C

_C.TASK = "part_segmentation"

_C.TRAIN.VAL_METRIC = "mIOU"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET.NUM_SEG_CLASSES = 0

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
