"""Configuration"""

from yacs.config import CfgNode


def purge_cfg(cfg):
    """Purge configuration

    The rules to purge is:
    1. If a CfgNode has "TYPE" attribute, remove its CfgNode children the key of which do not start with "TYPE".

    Args:
        cfg (CfgNode): input config

    """
    target_key = cfg.get("TYPE", None)
    removed_keys = []
    for k, v in cfg.items():
        if isinstance(v, CfgNode):
            if target_key is not None and (not k.startswith(target_key)):
                removed_keys.append(k)
            else:
                # Recursive purge
                purge_cfg(v)

    for k in removed_keys:
        del cfg[k]
