"""
Build models

Notes:
    When a new model is implemented, please provide a builder to build the model with config,
    and register it in _MODEL_BUILDERS

    How to implement a model:
    1. Modularize the model
    2. Try to add in_channels, out_channels to all the modules' attributes
    3. For the complete model, like PointNetCls, output a non-nested dictionary instead of single tensor or tuples
    4. Implement loss module whose inputs are preds and labels. Both of inputs are dict.
    5. Implement metric module (or use a general module in 'metric.py')

"""
from shaper.models.build import _MODEL_BUILDERS as origin_MODEL_BUILDERS
from .pointnet_fewshot import build_pointnet_fewshot
from .pn2ssg_fewshot import build_pn2ssg_fewshot
from .pn2msg_fewshot import build_pn2msg_fewshot
from .dgcnn_fewshot import build_dgcnn_fewshot
from .pn2s2cnn_fewshot import build_pns2cnn_fewshot

_MODEL_BUILDERS = {
    "PointNetFewShot": build_pointnet_fewshot,
    "PN2SSGFewShot": build_pn2ssg_fewshot,
    "PN2MSGFewShot": build_pn2msg_fewshot,
    "DGCNNFewShot": build_dgcnn_fewshot,
    "PNS2CNNFewShot": build_pns2cnn_fewshot,
}

_MODEL_BUILDERS.update(origin_MODEL_BUILDERS)


def build_model(cfg):
    return _MODEL_BUILDERS[cfg.MODEL.TYPE](cfg)


def register_model_builder(name, builder):
    if name in _MODEL_BUILDERS:
        raise KeyError(
            "Duplicate keys for {:s} with {} and {}."
            "Solve key conflicts first!".format(name, _MODEL_BUILDERS[name], builder))
    _MODEL_BUILDERS[name] = builder
