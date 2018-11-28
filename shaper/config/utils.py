import importlib
from yacs.config import CfgNode, load_cfg


def load_cfg_from_file(cfg_filename, purge=False):
    """Load config from a file

    Notes:
        Task-specific config is automatically imported according to cfg.TASK provided in cfg_file (yaml).

    Args:
        cfg_filename (str):
        purge (bool): whether to purge unused keys

    Returns:
        CfgNode: loaded configuration

    """
    with open(cfg_filename, "r") as f:
        cfg = load_cfg(f)
    assert cfg.TASK, "Task should be provided in config."
    cfg_module = importlib.import_module('.' + cfg.TASK, "shaper.config")
    # exec("from .{} import _C as cfg_template".format(cfg.TASK))
    cfg_template = cfg_module._C
    cfg_template.merge_from_other_cfg(cfg)
    if purge:
        purge_cfg(cfg_template)
    return cfg_template


def purge_cfg(cfg):
    target_key = cfg.get("TYPE", None)
    removed_keys = []
    for k, v in cfg.items():
        if isinstance(v, CfgNode):
            if target_key is not None and k != target_key:
                removed_keys.append(k)
            else:
                purge_cfg(v)

    for k in removed_keys:
        del cfg[k]
