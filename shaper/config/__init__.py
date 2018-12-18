"""Configuration

The current mode is to load a config.yaml first, and then dynamically load the corresponding configuration.
The alternative is to use cfg.merge_from_file(args.config_file) if there is only one task.

"""
from .utils import load_cfg_from_file
