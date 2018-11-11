from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.TASK = "classification"

_C.MODEL = CN()
_C.MODEL.TYPE = ""


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.TYPE = "PointCloud"
_C.INPUT.IN_CHANNELS = 3


# -----------------------------------------------------------------------------
# PointNet options
# -----------------------------------------------------------------------------
_C.MODEL.POINTNET = CN()

_C.MODEL.POINTNET.STEM_FUNC = ""

_C.MODEL.POINTNET.STEM_CHANNELS = (64, 64)
_C.MODEL.POINTNET.LOCAL_CHANNELS = (64, 128, 1024)
_C.MODEL.POINTNET.GLOBAL_CHANNELS = (512, 256)

_C.MODEL.POINTNET.REG_WEIGHT = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = ""
_C.DATASET.NUM_CLASSES = 0

# List of the data names for training
_C.DATASET.TRAIN = ()
# List of the data names for testing (or validation)
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

_C.SOLVER.TYPE = "Adam"

_C.SOLVER.MAX_EPOCH = 2

# Basic parameters of solvers
_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.WEIGHT_DECAY = 0.0

# training schedule
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (1,)

# Specific parameters of solvers
_C.SOLVER.Adam = CN()
_C.SOLVER.Adam.betas = (0.9, 0.999)


# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""  # if set to @, the filename of config will be used

# For reproducibility...but not really because modern fast GPU libraries use
# non-deterministic op implementations
_C.RNG_SEED = 0

# GPU devices to use
_C.DEVICE_IDS = ()