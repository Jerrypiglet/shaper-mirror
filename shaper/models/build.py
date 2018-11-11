"""
Build models

Notes:
    When a new model is implemented, please provide a builder to build the model with config,
    and register it in _MODEL_BUILDERS

"""

from .pointnet import build_pointnet


_MODEL_BUILDERS = {
    "PointNet": build_pointnet,
}


def build_model(cfg):
    return _MODEL_BUILDERS[cfg.MODEL.TYPE](cfg)


def register_model_builder(name, builder):
    if name in _MODEL_BUILDERS:
        raise KeyError(
            "Duplicate keys for {:s} with {} and {}."
            "Solve key conflicts first!".format(name, _MODEL_BUILDERS[name], builder))
    _MODEL_BUILDERS[name] = builder
