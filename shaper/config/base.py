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
#Select Normalization
_C.MODEL.NORMALIZATION = 'BN'
_C.MODEL.PROP_NORMALIZATION = 'BN'
_C.MODEL.SEG_NORMALIZATION = 'BN'
# Pre-trained weights
_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# INPUT (Specific for point cloud)
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# Input channels of point cloud
# channels = 3: (x, y, z)
# channels = 6: (x, y, z, normal_x, normal_y, normal_z)
_C.INPUT.IN_CHANNELS = 3
# -1 for all points
_C.INPUT.NUM_POINTS = -1
# Whether to use normal. Assume points[..,3:6] is normal.
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
# Whether to drop last during training
_C.DATALOADER.DROP_LAST = True

# ---------------------------------------------------------------------------- #
# Solver (optimizer)
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Type of optimizer
_C.SOLVER.TYPE = "Adam"

# Basic parameters of solvers
# Notice to change learning rate according to batch size
_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.WEIGHT_DECAY = 0.0

# Specific parameters of solvers
_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.momentum = 0.9

_C.SOLVER.Adam = CN()
_C.SOLVER.Adam.betas = (0.9, 0.999)

# ---------------------------------------------------------------------------- #
# Scheduler (learning rate schedule)
# ---------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = ""

_C.SCHEDULER.MAX_EPOCH = 1

_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 0
_C.SCHEDULER.StepLR.gamma = 0.1

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 32

# The period to save a checkpoint
_C.TRAIN.CHECKPOINT_PERIOD = 1000
_C.TRAIN.CHECKPOINT_START = 0
_C.TRAIN.LOG_PERIOD = 10

# The period to validate
_C.TRAIN.VAL_PERIOD = 1
# The metric for best validation performance
_C.TRAIN.VAL_METRIC = ""

# Data augmentation. The format is "method" or ("method", *args)
# For example, ("PointCloudRotate", ("PointCloudRotatePerturbation",0.1, 0.2))
_C.TRAIN.AUGMENTATION = ()

# Regex patterns of modules and/or parameters to freeze
# For example, ("bn",) will freeze all batch normalization layers' weight and bias;
# And ("module:bn",) will freeze all batch normalization layers' running mean and var.
_C.TRAIN.FROZEN_PATTERNS = ()

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.VISU_MODE=False

_C.TEST.BATCH_SIZE = 32

# The path of weights to be tested. "@" has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ""

# Data augmentation.
_C.TEST.AUGMENTATION = ()

_C.TEST.LOG_PERIOD = 10

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = "@"

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
# -1 means not to set explicitly.
_C.RNG_SEED = -1


