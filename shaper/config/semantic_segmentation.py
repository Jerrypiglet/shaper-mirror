"""Semantic Segmentation experiments configuration"""

from .base import CN, _C

# public alias
cfg = _C

_C.TASK = "semantic_segmentation"

_C.TRAIN.VAL_METRIC = "seg_acc"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET.NUM_SEG_CLASSES = 0

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
# Data augmentation for semantic segmentation.
_C.TRAIN.SEG_AUGMENTATION = ()

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
# Visualize failure cases. Path to visualize point clouds
_C.TEST.VIS_DIR = ""

# ---------------------------------------------------------------------------- #
# Test-time augmentations for point cloud semantic segmentation
# Now only support multi-view voting
# ---------------------------------------------------------------------------- #
_C.TEST.VOTE = CN()

_C.TEST.VOTE.ENABLE = False

# The axis along which to rotate
_C.TEST.VOTE.AXIS = "y"
# The number of views to vote
_C.TEST.VOTE.NUM_VIEW = 12
# Whether to shuffle points from different views (especially for PointNet++)
_C.TEST.VOTE.SHUFFLE = False

# -----------------------------------------------------------------------------
# PointNet options
# -----------------------------------------------------------------------------
_C.MODEL.POINTNET = CN()

_C.MODEL.POINTNET.STEM_CHANNELS = (64, 64)
_C.MODEL.POINTNET.LOCAL_CHANNELS = (64, 128, 1024)
_C.MODEL.POINTNET.GLOBAL_CHANNELS = (256, 128)
_C.MODEL.POINTNET.SEG_CHANNELS = (512, 256, 128, 128)
_C.MODEL.POINTNET.DROPOUT_PROB = 0.3
_C.MODEL.POINTNET.WITH_TRANSFORM = True

# -----------------------------------------------------------------------------
# PN2SSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2SSG = CN()

_C.MODEL.PN2SSG.NUM_CENTROIDS = (1024, 256, 64, 16)
_C.MODEL.PN2SSG.RADIUS = (0.1, 0.2, 0.4, 0.8)
_C.MODEL.PN2SSG.NUM_NEIGHBOURS = (32, 32, 32, 32)
_C.MODEL.PN2SSG.SA_CHANNELS = ((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256, 512))
_C.MODEL.PN2SSG.FP_CHANNELS = ((256, 256), (256, 256), (256, 128), (128, 128, 128))
_C.MODEL.PN2SSG.NUM_FP_NEIGHBOURS = (3, 3, 3)
_C.MODEL.PN2SSG.SEG_CHANNELS = (128,)
_C.MODEL.PN2SSG.DROPOUT_PROB = 0.5
_C.MODEL.PN2SSG.USE_XYZ = True

# -----------------------------------------------------------------------------
# PN2MSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2MSG = CN()

_C.MODEL.PN2MSG.NUM_CENTROIDS = (1024, 256, 64, 16)
_C.MODEL.PN2MSG.RADIUS_LIST = ((0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8))
_C.MODEL.PN2MSG.NUM_NEIGHBOURS_LIST = ((16, 32), (16, 32), (16, 32), (16, 32))
_C.MODEL.PN2MSG.SA_CHANNELS_LIST = (
                        ((16, 16, 32), (32, 32, 64)),
                        ((64, 64, 128), (64, 96, 128)),
                        ((128, 196, 256), (128, 196, 256)),
                        ((256, 256, 512), (256, 384, 512)))
_C.MODEL.PN2MSG.FP_CHANNELS = ((128, 128), (256, 256), (512, 512), (512, 512))
_C.MODEL.PN2MSG.NUM_FP_NEIGHBOURS = (3, 3, 3, 3)
_C.MODEL.PN2MSG.SEG_CHANNELS = (128,)
_C.MODEL.PN2MSG.DROPOUT_PROB = 0.5
_C.MODEL.PN2MSG.USE_XYZ = True
