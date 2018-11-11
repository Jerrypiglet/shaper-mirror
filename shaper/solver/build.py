"""
Build optimizers and schedulers

Notes:
    Default optimizer will optimize all parameters.
    Custom optimizer should be implemented and registered in '_OPTIMIZER_BUILDERS'

"""
import torch


_OPTIMIZER_BUILDERS = {}


def build_optimizer(cfg, model):
    name = cfg.SOLVER.TYPE
    if hasattr(torch.optim, name):
        def builder(cfg, model):
            return getattr(torch.optim, name)(
                model.parameters(),
                lr=cfg.SOLVER.BASE_LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                **cfg.SOLVER[name],
            )
    elif name in _OPTIMIZER_BUILDERS:
        builder = _OPTIMIZER_BUILDERS[name]
    else:
        raise ValueError("Unsupported type of optimizer.")

    return builder(cfg, model)


def register_optimizer_builder(name, builder):
    if name in _OPTIMIZER_BUILDERS:
        raise KeyError(
            "Duplicate keys for {:s} with {} and {}."
            "Solve key conflicts first!".format(name, _OPTIMIZER_BUILDERS[name], builder))
    _OPTIMIZER_BUILDERS[name] = builder
