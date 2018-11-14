from torch import nn


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def init_uniform(module, relu=True):
    if module.weight is not None:
        nn.init.kaiming_uniform_(module.weight,
                                 nonlinearity="relu" if relu else "linear")
    if module.bias is not None:
        nn.init.zeros_(module.bias)
