"""Configuration

The current mode is to load a config.yaml first, and then dynamically load the corresponding configuration.
The alternative is to use cfg.merge_from_file(args.config_file) if there is only one task.

"""

from yacs.config import CfgNode


def purge_cfg(cfg):
    """Purge configuration

    The rules to purge is:
    1. If a CfgNode has "TYPE" attribute, remove its CfgNode children the key of which do not contain "TYPE".

    Args:
        cfg (CfgNode): input config

    Returns:
        CfgNode: output config

    """
    new_cfg = cfg.clone()
    target_key = new_cfg.get("TYPE", None)
    removed_keys = []
    for k, v in new_cfg.items():
        if isinstance(v, CfgNode):
            if target_key is not None and (k not in target_key):
                removed_keys.append(k)
            else:
                purge_cfg(v)

    for k in removed_keys:
        del new_cfg[k]

    return new_cfg
