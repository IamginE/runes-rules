# SHARED FUNCTIONALITY FOR EVALUATION SCRIPTS
import argparse
import numpy as np
import random
import torch


from data_generation import (
    CFG_DIR,
    LOGGING_DIR,
    DATA_DIR,
    REAL_DATA_DIR
)

def str2bool(v: str | bool):
    if isinstance(v, bool):
        return v  # already a bool
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
def worker_init_fn(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def create_dataset_path(synthetic: bool, dataset_name: str, regular: bool, mode: str, is_torch: bool, setting_num: int = None) -> str:
    base_name = ""
    if synthetic:
        assert setting_num is not None
        regularity_str = 'regular' if regular else 'irregular'
        base_name = f"{DATA_DIR}/synthetic/{regularity_str}_setting_{setting_num}/{dataset_name}_{mode}"
    else:
        base_name = f'{REAL_DATA_DIR}/{dataset_name}/{mode}'
        
    if is_torch:
        return base_name + "_torch.pt"
    else:
        return base_name + ".pt"
    
    
def create_logging_dir_path(synthetic: bool, dataset_name: str, regular: bool, setting_num: int = None) -> str:
    if synthetic:
        assert setting_num is not None
        regularity_str = 'regular' if regular else 'irregular'
        return f'{LOGGING_DIR}/synthetic/{regularity_str}_setting_{setting_num}/{dataset_name}'
    else:
        return f'{LOGGING_DIR}/real/{dataset_name}'
    
    
def create_cfg_file_path(synthetic: bool, dataset_name: str, model: str, regular: bool, setting_num: int = None) -> str:
    if synthetic:
        assert setting_num is not None
        regularity_str = 'regular' if regular else 'irregular'
        return f'{CFG_DIR}/synthetic/{model}/{regularity_str}_setting_{setting_num}/{dataset_name}.yaml'
    else:
        return f'{CFG_DIR}/real/{model}/{dataset_name}.yaml'