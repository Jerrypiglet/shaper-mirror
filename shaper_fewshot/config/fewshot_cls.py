from shaper.config.classification import _C, CN

_C.TASK = "classification"

# Inherit models from shaper
# Penultimate layer before classifier
_C.MODEL.PENULT_CHANNELS = 0
_C.MODEL.PENULT_DROPOUT = False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET.FEWSHOT = CN()
# The number of instances per support class
_C.DATASET.FEWSHOT.K = 1
# cross validation index
_C.DATASET.FEWSHOT.CROSS_INDEX = 0

# Random seed for data loader
_C.DATALOADER.RNG_SEED = 0

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
# Modules to be frozen; support regex
_C.TRAIN.FROZEN_MODULES = ()
# Parameters to be frozen; support regex
_C.TRAIN.FROZEN_PARAMS = ()

# Learning rate scheduler with warm up
# The number of epochs to warm up. 0 for disable.
_C.SOLVER.WARMUP_STEP = 0
# Initial ratio of learning rate for warm up
_C.SOLVER.WARMUP_GAMMA = 0.1
