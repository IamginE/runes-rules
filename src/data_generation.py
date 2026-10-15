from itertools import product
import math
import os
import random
import torch
import yaml

from data.real_data import (
    materialize_amazon_dataset,
    materialize_chorales_dataset,
    materialize_commits_dataset,
    materialize_holidays_dataset,
    materialize_rolling_mill_dataset
)
from data.synthetic_data import (
    generate_fully_random_dataset_regular,
    generate_random_rule_set_uniform_tails,
    generate_synthetic_data_targets_regular
)
from data.utils import (
    convert_dataset_to_torch
)

RANDOM_SEED = 1445
DATASETS_PER_SYNTHETIC_SETUP = 10
# default parameters are evaluated for all settings (grid style) while 'vary' parameters are varied 1-by-1 under the default settings
# vary_i parameters are together evaluated in grid style, but are otherwise evaluated under the default settings
PARAMETERS_SYNTHETIC_DATASETS_REG = {
    'sequence_dim':                 {'default': [128],  'vary': [4, 8, 16, 32, 256, 512, 1024]   },
    'num_rules':                    {'default': [20],   'vary': [10, 40]                         },
    'min_size_head':                {'default': [1]     },
    'max_size_head':                {'default': [8]     },
    'min_size_tail':                {'default': [1]     },
    'max_size_tail':                {'default': [1]     },
    'min_delay':                    {'default': [1]     },
    'max_delay':                    {'default': [64]    },
    'min_span_int':                 {'default': [1]     },
    'max_span_int':                 {'default': [10]    },
    
    'num_samples':                  {'default': [1024], 'vary': [64, 128, 512, 2048, 4096]      },
    'min_sequence_length':          {'default': [96]    },
    'max_sequence_length':          {'default': [128]   },
    'percentage_rules':             {'default': [0.3],  'vary': [0.6]               },
    'frequency_rule_consequent':    {'default': [0.4],  'vary_1': [0.05, 0.1, 0.9]              },
    'rule_confidence':              {'default': [0.85], 'vary_1': [0.1, 0.3, 0.6, 1.0]          },
}

PARAMETERS_SYNTHETIC_DATASETS_REG_DEF = {
    'sequence_dim':                 {'default': [128]},
    'num_rules':                    {'default': [20]},
    'min_size_head':                {'default': [1] },
    'max_size_head':                {'default': [8] },
    'min_size_tail':                {'default': [1] },
    'max_size_tail':                {'default': [1] },
    'min_delay':                    {'default': [1] },
    'max_delay':                    {'default': [64]},
    'min_span_int':                 {'default': [1] },
    'max_span_int':                 {'default': [10]},
    
    'num_samples':                  {'default': [1024]},
    'min_sequence_length':          {'default': [96]  },
    'max_sequence_length':          {'default': [128] },
    'percentage_rules':             {'default': [0.3] },
    'frequency_rule_consequent':    {'default': [0.4] },
    'rule_confidence':              {'default': [0.85]},
}

CFG_DIR = 'cfgs'
DATA_DIR = 'data'
REAL_DATA_DIR = 'data/real'
# SYNTHETIC_DATA_DIR = 'data/synthetic'
LOGGING_DIR = 'logs'

def generate_setups_from_dict(setup_dict: dict[str, dict[str, list[int | float]]]) -> list[dict[str, int | float]]:
    default_settings = []
    settings = []
    # if Python dicts were not ordered, this would break completely
    for params in product(*[v['default'] for v in setup_dict.values()]):
        setting = {k: params[i] for i, k in enumerate(setup_dict.keys())}
        if setting['rule_confidence'] > setting['frequency_rule_consequent']:
            default_settings.append(setting)
            settings.append({k: params[i] for i, k in enumerate(setup_dict.keys())})
        
    track_vary_i = {}
    for k, v in setup_dict.items():
        if 'vary' in v:
            for setting in default_settings:
                for param in v['vary']:
                    new_setting = setting.copy()
                    new_setting[k] = param
                    if new_setting['rule_confidence'] > new_setting['frequency_rule_consequent']:
                        settings.append(new_setting)
        
        for k_i in v.keys():
            if k_i.startswith('vary_'):
                idx = int(k_i.split('_')[1])
                if idx in track_vary_i:
                    track_vary_i[idx].append((k, v[k_i].copy()))
                else:
                    track_vary_i[idx] = [(k, v[k_i].copy())]
                    
    
    for l in track_vary_i.values():
        keys = [x[0] for x in l]
        param_lists = [x[1] for x in l] 
        for i, k in enumerate(keys):
            param_lists[i] += setup_dict[k]['default']
        for params in product(*param_lists):
            if all([(params[i] in setup_dict[keys[i]]['default']) for i in range(len(keys))]):
                # default config
                continue
            for setting in default_settings:
                new_setting = setting.copy()
                for k, param in zip(keys, params):
                    new_setting[k] = param
                if new_setting['rule_confidence'] > new_setting['frequency_rule_consequent'] and new_setting not in settings:
                    settings.append(new_setting)
   
    return settings


def create_dataset_file_path_synthetic(data_dir: str, regular: bool, setting_num: int, dataset_num: int, mode: str, is_torch: bool) -> str:
    regularity_str = 'regular' if regular else 'irregular'
    base_name = f"{data_dir}/synthetic/{regularity_str}_setting_{setting_num}/dataset_{dataset_num}_{mode}"
    if is_torch:
        return base_name + "_torch.pt"
    else:
        return base_name + ".pt"
                
                
def create_cfg_file_path_synthetic(cfg_dir: str, model: str, regular: bool, setting_num: int, dataset_num: int) -> str:
    regularity_str = 'regular' if regular else 'irregular'
    return f"{cfg_dir}/synthetic/{model}/{regularity_str}_setting_{setting_num}/dataset_{dataset_num}.yaml"


def create_logging_dir_path_synthetic(log_dir: str, regular: bool, setting_num: int, dataset_num: int) -> str:
    regularity_str = 'regular' if regular else 'irregular'
    return f'{log_dir}/synthetic/{regularity_str}_setting_{setting_num}/dataset_{dataset_num}'



def materialize_datasets_real():
    """
    Materialize all real datasets in their list and tensor versions.
    """
    if not os.path.exists(REAL_DATA_DIR):
        os.makedirs(REAL_DATA_DIR)
    
    # datasets from the paper
    materialize_chorales_dataset(RANDOM_SEED, save_dir=REAL_DATA_DIR)
    materialize_commits_dataset(save_dir=REAL_DATA_DIR)
    materialize_holidays_dataset(save_dir=REAL_DATA_DIR)
    materialize_rolling_mill_dataset(RANDOM_SEED, save_dir=REAL_DATA_DIR)
    materialize_amazon_dataset(RANDOM_SEED, save_dir=REAL_DATA_DIR)
    
    
def materialize_cfgs_for_dataset_synthetic_reg(setting_params: dict, setting_num: int, dataset_num: int):
    
    regular_sampling = True
    save_dir_logs = create_logging_dir_path_synthetic(LOGGING_DIR, True, setting_num, dataset_num)
    dataset_type = 'synthetic_dataset_reg'
    dataset_name = f'setting_{setting_num}_dataset_{dataset_num}'
    batch_size = 64
    only_full_window = False
    max_epochs = 1000
    warmup_epochs_rules = 60
    warmup_epochs_gt = 10
    if setting_params['num_samples'] != 1024:
        ratio = 1024 / setting_params['num_samples']
        warmup_epochs_rules = math.ceil(warmup_epochs_rules * ratio)
        warmup_epochs_rules = min(warmup_epochs_rules, 1000)
        warmup_epochs_gt = math.ceil(warmup_epochs_gt * ratio)
        warmup_epochs_gt = min(warmup_epochs_gt, 100)
        max_epochs = math.ceil(max_epochs * ratio)
    
    cfg_gt_model = {
        'dataset': {
            'type': dataset_type,
            'name': dataset_name,
            'batching_strategy': 'samples',
        },
        'logging': {
            'save_dir': save_dir_logs,
            'name': 'gt_model'
        },
        'model': {
            'type': 'rules_gt',
            'sequence_dim': setting_params['sequence_dim'],
            'max_sequence_length': setting_params['max_sequence_length'],
            'max_relative_distance': setting_params['max_delay'],
            'regular_sampling': regular_sampling,
            'only_full_window': only_full_window,
        },
        'optimization': {
            'optimizer': {
                'kind': 'adam',
                'params': {'lr': 0.03},
            },
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'num_lr_decreases': 10, # early stopping stops sooner, typically
            'lr_decay': 0.3,
            'lr_patience': 10,
            'plateau_delta': 0.01,
            'warmup_epochs': warmup_epochs_gt,
        } 
    }
    cfg_file_path = create_cfg_file_path_synthetic(CFG_DIR, 'gt_model', True, setting_num, dataset_num)
    if not os.path.exists(os.path.dirname(cfg_file_path)):
        os.makedirs(os.path.dirname(cfg_file_path))
    with open(cfg_file_path, "w") as f:
        yaml.dump(cfg_gt_model, f, default_flow_style=False)
        
    
    cfg_rules = {
        'dataset': {
            'type': dataset_type,
            'name': dataset_name,
            'batching_strategy': 'samples',
        },
        'logging': {
            'save_dir': save_dir_logs,
            'name': 'rules'
        },     
        'model': {
            'type': 'rules',
            'sequence_dim': setting_params['sequence_dim'],
            'num_neurons': 300,
            'num_rules': 64,
            'max_sequence_length': setting_params['max_sequence_length'],
            'max_relative_distance': setting_params['max_delay'],
            'initial_temperature': 1.0,
            'decay_neuron_temp': 0.90,
            'decay_conjunction_temp': 0.90,
            'regular_sampling': regular_sampling,
            'only_full_window': only_full_window,
        },
        'optimization': {
            'optimizer': {
                'kind': 'adam',
                'params': {'lr': 0.03},
            },
            'batch_size': batch_size,
            'max_epochs': max_epochs,
            'num_lr_decreases': 0, # directly anneal temp. after warmup epochs
            'lr_decay': 1.0, 
            'lr_patience': 0,
            'plateau_delta': 0.0,
            'warmup_epochs': warmup_epochs_rules,
            'steps_temp_annealing': 32,
            'lambda_reg_int': 0.0,
        } 
    }
    cfg_file_path = create_cfg_file_path_synthetic(CFG_DIR, 'rules', True, setting_num, dataset_num)
    if not os.path.exists(os.path.dirname(cfg_file_path)):
        os.makedirs(os.path.dirname(cfg_file_path))
    with open(cfg_file_path, "w") as f:
        yaml.dump(cfg_rules, f, default_flow_style=False)
        
    
    cfg_transformer =  {
        'dataset': {
            'type': dataset_type,
            'name': dataset_name,
            'batching_strategy': 'samples',
        },
        'logging': {
            'save_dir': save_dir_logs,
            'name': 'transformer'
        },     
        'model': {
            'type': 'transformer',
            'sequence_dim': setting_params['sequence_dim'],
            'transformer_num_layers': 4,
            'att_dim': 128,
            'embedding_dim': 128,
            'num_heads': 4,
            'max_sequence_length': setting_params['max_sequence_length'],
            'transformer_ff_nn_dim': 256,
            'final_nn_dim': 128,
            'final_nn_num_layers': 2,
            'activation': 'gelu',
            'dropout': 0.1,
            'max_relative_distance': setting_params['max_delay'],
            'regular_sampling': regular_sampling,
            'only_full_window': only_full_window,
        },
        'optimization': {
            'optimizer': {
                'kind': 'adam',
                'params': {'lr': 0.0003},
            },
            'scheduler': {
                'kind': 'on_plateau',
                'params': {
                    'mode': 'min', 
                    'factor': 0.1, 
                    'patience': 10, 
                    'threshold': 0.01,
                    'threshold_mode': 'abs'
                },
            },
            'batch_size': batch_size,
            'max_epochs': max_epochs
        } 
    }
    cfg_file_path = create_cfg_file_path_synthetic(CFG_DIR, 'transformer', True, setting_num, dataset_num)
    if not os.path.exists(os.path.dirname(cfg_file_path)):
        os.makedirs(os.path.dirname(cfg_file_path))
    with open(cfg_file_path, "w") as f:
        yaml.dump(cfg_transformer, f, default_flow_style=False)


def materialize_datasets_and_configs_synthetic_regular(base_setting_num: int = 0):
    """
    Materialize all synthetic datasets for regular sampling in their list and tensor versions.
    Also generate the configs for the synthetic data.
    """    
    regular_sampling = True
    
    print(f"Seeding random with seed: {RANDOM_SEED}")
    random.seed(RANDOM_SEED)
    for setting_num, setting_params in enumerate(generate_setups_from_dict(PARAMETERS_SYNTHETIC_DATASETS_REG)):
        
        setting_num += base_setting_num
        print(f"Generating rules and datasets for setting {setting_num}.")
        for dataset_num in range(DATASETS_PER_SYNTHETIC_SETUP):
            # ====================== Generate the synthetic datasets ======================
            # this could be done more easily with **kwargs but paranoia from missusing args kicks in :D
            rule_set = generate_random_rule_set_uniform_tails(
                num_rules = setting_params['num_rules'],
                sequence_dim = setting_params['sequence_dim'],
                min_size_head = setting_params['min_size_head'],
                max_size_head = setting_params['max_size_head'],
                min_size_tail = setting_params['min_size_tail'],
                max_size_tail = setting_params['max_size_tail'],
                min_delay = setting_params['min_delay'],
                max_delay = setting_params['max_delay'],
                min_span_int = setting_params['min_span_int'],
                max_span_int = setting_params['max_span_int'],
                regular_sampling = regular_sampling,
                random_seed = None, # otherwise we would get the same rule set every time
            )
            
            for mode in ['train', 'val', 'test']:
                num_samples = setting_params['num_samples']
                if mode == 'val':
                    num_samples = num_samples // 2
                
                data = generate_synthetic_data_targets_regular(
                    num_samples = num_samples,
                    min_sequence_length = setting_params['min_sequence_length'],
                    max_sequence_length = setting_params['max_sequence_length'],
                    sequence_dim = setting_params['sequence_dim'],
                    rules = rule_set,
                    percentage_rules = setting_params['percentage_rules'],
                    frequency_rule_consequent = setting_params['frequency_rule_consequent'],
                    rule_confidence = setting_params['rule_confidence'],
                    uniform_random_placement = False, # for fast generation
                    ensure_disjoint = False,
                    random_seed = None, # otherwise we would get the same dataset for train, val, test
                    retries_fast_placement = 5,
                )
                
                list_dataset_file_path =  create_dataset_file_path_synthetic(
                    data_dir = DATA_DIR, 
                    regular = True, 
                    setting_num = setting_num, 
                    dataset_num = dataset_num,
                    mode = mode,
                    is_torch = False 
                )
                dir_name_dataset = os.path.dirname(list_dataset_file_path)
                if not os.path.exists(dir_name_dataset):
                    os.makedirs(dir_name_dataset)
                # summary file of the generation parameters
                if not os.path.exists(f'{dir_name_dataset}/summary.yaml') or dataset_num == 0:
                    with open(f'{dir_name_dataset}/summary.yaml', "w") as f:
                        yaml.dump(setting_params, f, default_flow_style=False)
                torch.save(data, list_dataset_file_path)
        
                convert_dataset_to_torch(
                    data, 
                    save_path = create_dataset_file_path_synthetic(
                        data_dir = DATA_DIR, 
                        regular = True, 
                        setting_num = setting_num, 
                        dataset_num = dataset_num,
                        mode = mode,
                        is_torch = True 
                    ),
                    has_padding = True
                )
            
            # ====================== Generate the associated config files. ======================  
            materialize_cfgs_for_dataset_synthetic_reg(setting_params, setting_num, dataset_num)
                
            
def materialize_datasets_and_configs_random_regular():
    """
    Materialize regularly sampled random datasets in their list and tensor versions.
    Also generate the configs for the random data.
    """    
    num_samples = 1024
    min_sequence_length = 96
    max_sequence_length = 128
    max_relative_distance = 64
    sequence_dim = 128
    regular_sampling = True
    dataset_type = 'synthetic_dataset_reg'
    batch_size = 64
    only_full_window = False
    max_epochs = 1000
    warmup_epochs_rules = 60

    
    print(f"Seeding random with seed: {RANDOM_SEED}")
    random.seed(RANDOM_SEED)
    for dataset_num in range(10):
        
        data = generate_fully_random_dataset_regular(
            num_samples = num_samples,
            min_sequence_length = min_sequence_length,
            max_sequence_length = max_sequence_length,
            sequence_dim = sequence_dim,
            random_seed = None
        )
        list_dataset_file_path = f'{DATA_DIR}/synthetic/regular_random/dataset_{dataset_num}.pt'
        dir_name_dataset = os.path.dirname(list_dataset_file_path)
        if not os.path.exists(dir_name_dataset):
            os.makedirs(dir_name_dataset)
        # summary file of the generation parameters
        if not os.path.exists(f'{dir_name_dataset}/summary.yaml') or dataset_num == 0:
            with open(f'{dir_name_dataset}/summary.yaml', "w") as f:
                yaml.dump({
                    'num_samples': num_samples,
                    'min_sequence_length': min_sequence_length,
                    'max_sequence_length': max_sequence_length,
                    'sequence_dim': sequence_dim
                }, f, default_flow_style=False)
        torch.save(data, list_dataset_file_path)
        
        convert_dataset_to_torch(
            data, 
            save_path = f'{DATA_DIR}/synthetic/regular_random/dataset_{dataset_num}_torch.pt',
            has_padding = True
        )
        
        save_dir_logs = f'{LOGGING_DIR}/synthetic/regular_random/dataset_{dataset_num}' 
        cfg_file_path = f'{CFG_DIR}/synthetic/rules/regular_random/dataset_{dataset_num}.yaml'
        dataset_name = f'random_dataset_{dataset_num}'
        cfg_rules = {
            'dataset': {
                'type': dataset_type,
                'name': dataset_name,
                'batching_strategy': 'samples',
            },
            'logging': {
                'save_dir': save_dir_logs,
                'name': 'rules'
            },     
            'model': {
                'type': 'rules',
                'sequence_dim': sequence_dim,
                'num_neurons': 300,
                'num_rules': 64,
                'max_sequence_length': max_sequence_length,
                'max_relative_distance': max_relative_distance,
                'initial_temperature': 1.0,
                'decay_neuron_temp': 0.90,
                'decay_conjunction_temp': 0.90,
                'regular_sampling': regular_sampling,
                'only_full_window': only_full_window,
            },
            'optimization': {
                'optimizer': {
                    'kind': 'adam',
                    'params': {'lr': 0.03},
                },
                'batch_size': batch_size,
                'max_epochs': max_epochs,
                'num_lr_decreases': 0, # directly anneal temp. after warmup epochs
                'lr_decay': 1.0, 
                'lr_patience': 0,
                'plateau_delta': 0.0,
                'warmup_epochs': warmup_epochs_rules,
                'steps_temp_annealing': 32,
                'lambda_reg_int': 0.0,
            } 
        }
        
        if not os.path.exists(os.path.dirname(cfg_file_path)):
            os.makedirs(os.path.dirname(cfg_file_path))
        with open(cfg_file_path, "w") as f:
            yaml.dump(cfg_rules, f, default_flow_style=False)
                                
            
def materialize_cfgs_for_dataset_synthetic_ppm_reg(setting_params: dict, setting_num: int, dataset_num: int):
    dataset_type = 'synthetic_dataset_reg'
    dataset_name = f'setting_{setting_num}_dataset_{dataset_num}'
    
    cfg_ppm_model = {
        'dataset': {
            'type': dataset_type,
            'name': dataset_name,
            'batching_strategy': 'samples',
        },
        'model': {
            'type': 'ppm_markov_model',
            'sequence_dim': setting_params['sequence_dim'],
        },
    }
    cfg_file_path = create_cfg_file_path_synthetic(CFG_DIR, 'ppm_markov_model', True, setting_num, dataset_num)
    if not os.path.exists(os.path.dirname(cfg_file_path)):
        os.makedirs(os.path.dirname(cfg_file_path))
    with open(cfg_file_path, "w") as f:
        yaml.dump(cfg_ppm_model, f, default_flow_style=False)
      
        
def materialize_configs_synthetic_regular_ppm(base_setting_num: int = 0):
    """
    Generate the PPM configs for the synthetic, regularly sampled data.
    """    
   
    for setting_num, setting_params in enumerate(generate_setups_from_dict(PARAMETERS_SYNTHETIC_DATASETS_REG)):
        setting_num += base_setting_num
        for dataset_num in range(DATASETS_PER_SYNTHETIC_SETUP):
            materialize_cfgs_for_dataset_synthetic_ppm_reg(setting_params, setting_num, dataset_num)
            

def materialize_cfgs_for_dataset_synthetic_trees_reg(setting_params: dict, setting_num: int, dataset_num: int):
    
    regular_sampling = True
    save_dir_logs = create_logging_dir_path_synthetic(LOGGING_DIR, True, setting_num, dataset_num)
    dataset_type = 'synthetic_dataset_reg'
    dataset_name = f'setting_{setting_num}_dataset_{dataset_num}'
    
    cfg_tree_models = {
        'dataset': {
            'type': dataset_type,
            'name': dataset_name,
            'batching_strategy': 'samples',
        },
        'logging': {
            'save_dir': save_dir_logs,
        },
        'model': {
            'sequence_dim': setting_params['sequence_dim'],
            'max_sequence_length': setting_params['max_sequence_length'],
            'max_relative_distance': setting_params['max_delay'],
            'regular_sampling': regular_sampling,
        },
    }
    cfg_file_path = create_cfg_file_path_synthetic(CFG_DIR, 'trees', True, setting_num, dataset_num)
    if not os.path.exists(os.path.dirname(cfg_file_path)):
        os.makedirs(os.path.dirname(cfg_file_path))
    with open(cfg_file_path, "w") as f:
        yaml.dump(cfg_tree_models, f, default_flow_style=False)
      
        
def materialize_configs_synthetic_regular_trees(base_setting_num: int = 0):
    """
    Generate the tree boosting configs for the synthetic, regularly sampled data.
    """    
   
    for setting_num, setting_params in enumerate(generate_setups_from_dict(PARAMETERS_SYNTHETIC_DATASETS_REG)):
        setting_num += base_setting_num
        for dataset_num in range(DATASETS_PER_SYNTHETIC_SETUP):
            materialize_cfgs_for_dataset_synthetic_trees_reg(setting_params, setting_num, dataset_num)
            
                
if __name__ == "__main__":
    materialize_datasets_and_configs_random_regular()
    materialize_datasets_and_configs_synthetic_regular()
    materialize_configs_synthetic_regular_ppm()
    materialize_configs_synthetic_regular_trees()
    materialize_datasets_real()