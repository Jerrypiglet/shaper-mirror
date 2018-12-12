"""Classification experiments configuration"""

from .base import CN, _C

_C.TASK = "classification"

# -----------------------------------------------------------------------------
# PointNet options
# -----------------------------------------------------------------------------
_C.MODEL.POINTNET = CN()

_C.MODEL.POINTNET.STEM_CHANNELS = (64, 64)
_C.MODEL.POINTNET.LOCAL_CHANNELS = (64, 128, 1024)
_C.MODEL.POINTNET.GLOBAL_CHANNELS = (512, 256)

_C.MODEL.POINTNET.DROPOUT_PROB = 0.3
_C.MODEL.POINTNET.WITH_TRANSFORM = True

_C.MODEL.POINTNET.REG_WEIGHT = 0.032

# -----------------------------------------------------------------------------
# DGCNN options
# -----------------------------------------------------------------------------
_C.MODEL.DGCNN = CN()

_C.MODEL.DGCNN.K = 20
_C.MODEL.DGCNN.EDGE_CONV_CHANNELS = (64, 64, 64, 128)
_C.MODEL.DGCNN.INTER_CHANNELS = 1024
_C.MODEL.DGCNN.GLOBAL_CHANNELS = (512, 256)

_C.MODEL.DGCNN.DROPOUT_PROB = 0.5
_C.MODEL.DGCNN.LABEL_SMOOTHING = 0.2

# -----------------------------------------------------------------------------
# PN2SSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2SSG = CN()

_C.MODEL.PN2SSG.NUM_CENTROIDS = (512, 128)
_C.MODEL.PN2SSG.RADIUS = (0.2, 0.4)
_C.MODEL.PN2SSG.NUM_NEIGHBOURS = (32, 64)
_C.MODEL.PN2SSG.SA_CHANNELS = ((64, 64, 128), (128, 128, 256))
_C.MODEL.PN2SSG.LOCAL_CHANNELS = (256, 512, 1024)
_C.MODEL.PN2SSG.GLOBAL_CHANNELS = (512, 256)
_C.MODEL.PN2SSG.DROPOUT_PROB = 0.5
_C.MODEL.PN2SSG.USE_XYZ = True

# -----------------------------------------------------------------------------
# PN2MSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2MSG = CN()

_C.MODEL.PN2MSG.NUM_CENTROIDS = (512, 128)
_C.MODEL.PN2MSG.RADIUS = ((0.1, 0.2, 0.4), (0.2, 0.4, 0.8))
_C.MODEL.PN2MSG.NUM_NEIGHBOURS = ((16, 32, 128), (32, 64, 128))
_C.MODEL.PN2MSG.SA_CHANNELS = (
    ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
    ((64, 64, 128), (128, 128, 256), (128, 128, 256)))
_C.MODEL.PN2MSG.LOCAL_CHANNELS = (256, 512, 1024)
_C.MODEL.PN2MSG.GLOBAL_CHANNELS = (512, 256)
_C.MODEL.PN2MSG.DROPOUT_PROB = 0.5
_C.MODEL.PN2MSG.USE_XYZ = True

# -----------------------------------------------------------------------------
# S2CNN options
# -----------------------------------------------------------------------------
_C.MODEL.S2CNN = CN()

_C.MODEL.S2CNN.BAND_WIDTH_IN = 30
_C.MODEL.S2CNN.FEATURE_CHANNELS = (100, 100)
_C.MODEL.S2CNN.BAND_WIDTH_LIST = (16, 10)

# -----------------------------------------------------------------------------
# DGPN2 options
# -----------------------------------------------------------------------------
_C.MODEL.DGPN2 = CN()

# Local pointnet paras
_C.MODEL.DGPN2.NUM_POINTS = 256
_C.MODEL.DGPN2.RADIUS = (0.2)
_C.MODEL.DGPN2.NUM_SAMPLES = (16, 32, 128)
_C.MODEL.DGPN2.GROUP_MLPS = ((32, 32, 64), (64, 64, 128), (64, 96, 128))

# Dynamic graph paras
_C.MODEL.DGPN2.EDGE_CONV_CHANNELS = (128, 256, 512)
_C.MODEL.DGPN2.INTER_CHANNELS = 128
_C.MODEL.DGPN2.GLOBAL_CHANNELS = (512, 256)
_C.MODEL.DGPN2.K = 20
_C.MODEL.DGPN2.TRANSFORM_XYZ = True
_C.MODEL.DGPN2.DROP_PROB = 0.5

_C.MODEL.DGPN2.LABEL_SMOOTH = 0.0
_C.MODEL.DGPN2.TRANS_REG_WEIGHT = 0.0
