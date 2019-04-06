"""Classification experiments configuration"""

from .base import CN, _C

# public alias
cfg = _C

_C.TASK = "part_instance_segmentation"

_C.TRAIN.VAL_METRIC = "seg_acc"

_C.MODEL.NUM_INS_MASKS = 6

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
# Data augmentation for part segmentation.
_C.TRAIN.SEG_AUGMENTATION = ()

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
# Visualize failure cases. Path to visualize point clouds
_C.TEST.VIS_DIR = ""

# ---------------------------------------------------------------------------- #
# Test-time augmentations for point cloud classification
# ---------------------------------------------------------------------------- #
_C.TEST.VOTE = CN()

_C.TEST.VOTE.NUM_VOTE = 0

_C.TEST.VOTE.TYPE = ""

# Multi-view voting
_C.TEST.VOTE.MULTI_VIEW = CN()
# The axis along which to rotate
_C.TEST.VOTE.MULTI_VIEW.AXIS = "y"

# Data augmentation, different with TEST.AUGMENTATION.
# Use for voting only
_C.TEST.VOTE.AUGMENTATION = ()

# Whether to shuffle points from different views (especially for methods like PointNet++)
_C.TEST.VOTE.SHUFFLE = False

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


#------------------------------------------------------------------------------
# DGCNN options
#------------------------------------------------------------------------------
_C.MODEL.DGCNN = CN()

_C.MODEL.DGCNN.DROPOUT_PROB_CLS = 0.3
_C.MODEL.DGCNN.DROPOUT_PROB_SEG = 0.2
_C.MODEL.DGCNN.WITH_TRANSFORM = True

#_C.MODEL.DGCNN.REG_WEIGHT = 0.032
_C.MODEL.DGCNN.REG_WEIGHT = 0
_C.MODEL.DGCNN.CLS_LOSS_WEIGHT = 0.0
_C.MODEL.DGCNN.SEG_LOSS_WEIGHT = 1.0


# -----------------------------------------------------------------------------
# PN2SSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2SSG = CN()

_C.MODEL.PN2SSG.NUM_CENTROIDS = (512, 128)
_C.MODEL.PN2SSG.RADIUS = (0.2, 0.4)
_C.MODEL.PN2SSG.NUM_NEIGHBOURS = (32, 64)
_C.MODEL.PN2SSG.SA_CHANNELS = ((64, 64, 128), (128, 128, 256))
_C.MODEL.PN2SSG.LOCAL_CHANNELS = (256, 512, 1024)
_C.MODEL.PN2SSG.FP_LOCAL_CHANNELS = (256, 256)
_C.MODEL.PN2SSG.FP_CHANNELS = ((256, 128), (128, 128, 128))
_C.MODEL.PN2SSG.NUM_FP_NEIGHBOURS = (3, 3)
_C.MODEL.PN2SSG.SEG_CHANNELS = (128,)
_C.MODEL.PN2SSG.DROPOUT_PROB = 0.5
_C.MODEL.PN2SSG.USE_XYZ = True

# -----------------------------------------------------------------------------
# PN2MSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2MSG = CN()

_C.MODEL.PN2MSG.NUM_CENTROIDS = (512, 128)
_C.MODEL.PN2MSG.RADIUS_LIST = ((0.1, 0.2, 0.4), (0.4, 0.8))
_C.MODEL.PN2MSG.NUM_NEIGHBOURS_LIST = ((32, 64, 128), (64, 128))
_C.MODEL.PN2MSG.SA_CHANNELS_LIST = (
    ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
    ((128, 128, 256), (128, 196, 256)))
_C.MODEL.PN2MSG.LOCAL_CHANNELS = (256, 512, 1024)
_C.MODEL.PN2MSG.FP_LOCAL_CHANNELS = (256, 256)
_C.MODEL.PN2MSG.FP_CHANNELS = ((256, 128), (128, 128))
_C.MODEL.PN2MSG.NUM_FP_NEIGHBOURS = (3, 3)
_C.MODEL.PN2MSG.SEG_CHANNELS = (128,)
_C.MODEL.PN2MSG.DROPOUT_PROB = 0.5
_C.MODEL.PN2MSG.USE_XYZ = True
