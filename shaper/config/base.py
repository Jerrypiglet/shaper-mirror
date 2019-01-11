"""Basic experiments configuration

For different tasks, a specific configuration might be created by importing this basic config.

"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# Overwritten by different tasks
_C.TASK = ""

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
# Whether to use normal
_C.INPUT.USE_NORMAL = False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = ""
_C.DATASET.NUM_CLASSES = 0

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
_C.DATALOADER.NUM_WORKERS = 1
# Whether to drop last
_C.DATALOADER.DROP_LAST = True

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
# The metric for best validation performance
_C.TRAIN.VAL_METRIC = ""

# Data augmentation. The format is "method" or ("method", *args)
# For example, ("PointCloudRotate", ("PointCloudRotatePerturbation",0.1, 0.2))
_C.TRAIN.AUGMENTATION = ()

# Regex patterns of modules and/or parameters to freeze
_C.TRAIN.FROZEN_PATTERNS = ()

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

# Visualize key points in point cloud
_C.TEST.VIS_KEY_PTS = False

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
# _C.RNG_SEED = 0
