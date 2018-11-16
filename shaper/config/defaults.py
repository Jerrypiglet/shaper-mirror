from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.TASK = "classification"

# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True

_C.MODEL = CN()
_C.MODEL.TYPE = ""
# Pre-trained weights
_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# INPUT (Only support point cloud now)
# -----------------------------------------------------------------------------
_C.INPUT = CN()

_C.INPUT.IN_CHANNELS = 3
_C.INPUT.NUM_POINTS = -1

# -----------------------------------------------------------------------------
# PointNet options
# -----------------------------------------------------------------------------
_C.MODEL.POINTNET = CN()

_C.MODEL.POINTNET.STEM_CHANNELS = (64, 64)
_C.MODEL.POINTNET.LOCAL_CHANNELS = (64, 128, 1024)
_C.MODEL.POINTNET.GLOBAL_CHANNELS = (512, 256)

_C.MODEL.POINTNET.DROPOUT_PROB = 0.5
_C.MODEL.POINTNET.WITH_TRANSFORM = True

_C.MODEL.POINTNET.REG_WEIGHT = 0.0

# -----------------------------------------------------------------------------
# DGCNN options
# -----------------------------------------------------------------------------
_C.MODEL.DGCNN = CN()

_C.MODEL.DGCNN.K = 20
_C.MODEL.DGCNN.EDGE_CONV_CHANNELS = (64, 64, 64, 128)
_C.MODEL.DGCNN.INTER_CHANNELS = 1024
_C.MODEL.DGCNN.GLOBAL_CHANNELS = (512, 256)
_C.MODEL.DGCNN.DROP_PROB = 0.5

_C.MODEL.DGCNN.DROPOUT_PROB = 0.5
_C.MODEL.DGCNN.LABEL_SMOOTHING = 0.0

# -----------------------------------------------------------------------------
# S2CNN options
# -----------------------------------------------------------------------------
_C.MODEL.S2CNN = CN()

_C.MODEL.S2CNN.BAND_WIDTH_IN = 30
_C.MODEL.S2CNN.FEATURE_CHANNELS = (100, 100)
_C.MODEL.S2CNN.BAND_WIDTH_LIST = (16, 10)

# -----------------------------------------------------------------------------
# PN2MSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2MSG = CN()

_C.MODEL.PN2MSG.NUM_POINTS = (512, 128)
_C.MODEL.PN2MSG.RADIUS = ((0.1, 0.2, 0.4), (0.2, 0.4, 0.8))
_C.MODEL.PN2MSG.NUM_SAMPLE = ((16, 32, 128), (32, 64, 128))
_C.MODEL.PN2MSG.GROUP_MLPS = (
    ((32, 32, 64), (64, 64, 128), (64, 96, 128)), ((64, 64, 128), (128, 128, 256), (128, 128, 256)))
_C.MODEL.PN2MSG.GLOBAL_MLPS = (256, 512, 1024)
_C.MODEL.PN2MSG.FC_CHANNELS = (512, 256)
_C.MODEL.PN2MSG.DROP_PROB = 0.5
_C.MODEL.PN2MSG.USE_XYZ = True

# -----------------------------------------------------------------------------
# PN2SSG options
# -----------------------------------------------------------------------------
_C.MODEL.PN2SSG = CN()

_C.MODEL.PN2SSG.NUM_POINTS = (512, 128)
_C.MODEL.PN2SSG.RADIUS = (0.2, 0.4)
_C.MODEL.PN2SSG.NUM_SAMPLE = (32, 64)
_C.MODEL.PN2SSG.GROUP_MLPS = ((64, 64, 128), (128, 128, 256))
_C.MODEL.PN2SSG.GLOBAL_MLPS = (256, 512, 1024)
_C.MODEL.PN2SSG.FC_CHANNELS = (512, 256)
_C.MODEL.PN2SSG.DROP_PROB = 0.5
_C.MODEL.PN2SSG.USE_XYZ = True

# -----------------------------------------------------------------------------
# DGPN2 options
# -----------------------------------------------------------------------------
_C.MODEL.DGPN2 = CN()

# Local pointnet paras
_C.MODEL.DGPN2.NUM_POINTS = 256
_C.MODEL.DGPN2.RADIUS = (0.1, 0.2, 0.4)
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

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = ""
_C.DATASET.NUM_CLASSES = 0
_C.DATASET.SHAPE_NAME_PATH = ""

# Root directory of dataset
_C.DATASET.ROOT_DIR = ""
# List of the data names for training
_C.DATASET.TRAIN = ()
# List of the data names for validation
_C.DATASET.VAL = ()
# List of the data names for testing
_C.DATASET.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver (optimizer, learning schedule)
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Type of optimizer
_C.SOLVER.TYPE = "Adam"

_C.SOLVER.MAX_EPOCH = 1

# Basic parameters of solvers
# Notice to change learning rate according to batch size
_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.WEIGHT_DECAY = 0.0

# training schedule
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = ()

# Specific parameters of solvers
_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.momentum = 0.9

_C.SOLVER.Adam = CN()
_C.SOLVER.Adam.betas = (0.9, 0.999)

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 32

_C.TRAIN.CHECKPOINT_PERIOD = 1000
_C.TRAIN.LOG_PERIOD = 10

# Validation
_C.TRAIN.VAL_PERIOD = 1
_C.TRAIN.VAL_METRIC = "acc"

# Data augmentation. The format is "method" or ("method", *args)
# For example, ("PointCloudRotate", ("PointCloudRotatePerturbation",0.1, 0.2))
_C.TRAIN.AUGMENTATION = ()

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 32

# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ""

# Data augmentation.
_C.TEST.AUGMENTATION = ()

_C.TEST.LOG_PERIOD = 10

# Visualize errors. Path to visualize point clouds
_C.TEST.VIS_DIR = ""

# ---------------------------------------------------------------------------- #
# Test-time augmentations for point cloud classification
# Now only support multi-view voting
# ---------------------------------------------------------------------------- #
_C.TEST.VOTE = CN()

_C.TEST.VOTE.ENABLE = False

# The axis along which to rotate
_C.TEST.VOTE.AXIS = "y"
# The number of views to vote
_C.TEST.VOTE.NUM_VIEW = 12
# Heuristic used to combine predicted classification scores
#   Valid options: ("logit", "softmax", "label")
_C.TEST.VOTE.SCORE_HEUR = ("logit",)

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = "@"

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
_C.RNG_SEED = 0

# GPU devices to use; all available devices by default
_C.DEVICE_IDS = ()
