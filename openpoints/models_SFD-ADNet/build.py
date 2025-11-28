#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 16:44
# @Author  : wangjie
from openpoints.utils import registry
ADAPTMODELS = registry.Registry('adaptmodels')

def build_adaptpointmodels_from_cfg(cfg, **kwargs):
    """
    Build a criterion (loss function), defined by cfg.NAME.
    Args:
        cfg (eDICT):
    Returns:
        criterion: a constructed loss function specified by cfg.NAME
    """
    valid_keys = ['type', 'w_num_anchor', 'w_sigma', 'w_R_range', 'w_S_range', 'w_T_range']
    filtered_cfg = {k: v for k, v in cfg.items() if k in valid_keys}
    print("\n=== Building AdaptPointModels with config===")
    # print("Model Type: ", cfg.MODEL.NAME)
    print("All params: ", dict(cfg))
    return ADAPTMODELS.build(cfg, **kwargs)