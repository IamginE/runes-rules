import argparse
import lightning as L
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
import yaml

from data_generation import (
    CFG_DIR,
    LOGGING_DIR,
    REAL_DATA_DIR
)
from data.real_data import (
    AVAILABLE_DATASETS,
    MAP_NAME_TO_KIND_DATASET
)
from data.utils import (
    SequenceDataset,
    SequenceDatasetRandomSampling,
    custom_collate_fn,
)
from eval_utils import(
    worker_init_fn
)
from models.rope_transformer import ROPETransformer
from models.runes import Runes
from models.rules import (
    Rules
)

MODELS_AND_KEYS = {
    'rules': ('ckpt_dict', 'rules'),
    'transformer': ('ckpt_dict', 'transformer'),
    'rules_processed_rules': ('ckpt_list',),
    'runes_processed_rules': ('ckpt_list',),
    'runes_ablation': ('ckpt_dict', 'transformer'),
}

# ====================== Run ====================== #
# Get mean and std results for saved models.
if __name__ == '__main__':
    
    # ====================== Setup ====================== #
    print(f"Cuda is available {torch.cuda.is_available()}")
    full_run_time = time.time()
    parser = argparse.ArgumentParser(description='Real Data Evaluation Script to get mean and standard deviations.')
    parser.add_argument('--dataset', type=str, required=True, help=f"The dataset to run the model on. Has to be one of {AVAILABLE_DATASETS}.")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0, the main process spawns the loaders.")
    
    args = parser.parse_args()
    requested_dataset = args.dataset
    num_workers_dataloader = args.num_workers_dataloader
    
    # For debugging:
    # requested_dataset = 'holidays_irregular'
    # num_workers_dataloader = 0
    
    assert requested_dataset in AVAILABLE_DATASETS, f"Requested dataset '{requested_dataset}' but the only available datasets are {AVAILABLE_DATASETS}."
    log_results = f'{LOGGING_DIR}/real/{requested_dataset}'


    path_dataset_torch_train = f'{REAL_DATA_DIR}/{requested_dataset}/train_torch.pt'
    path_dataset_torch_val   = f'{REAL_DATA_DIR}/{requested_dataset}/val_torch.pt'
    path_dataset_torch_test  = f'{REAL_DATA_DIR}/{requested_dataset}/test_torch.pt'
    
    cfg_runes = f'{CFG_DIR}/real/runes/{requested_dataset}.yaml'
    path_results_rules_transformer     = f'{log_results}/aggregated_results.yaml'
    path_results_rules_processed_rules = f'{log_results}/rules_processed_rules_aggregated_results.yaml'
    path_results_runes_processed_rules = f'{log_results}/runes_processed_rules_aggregated_results.yaml'

    for p in [
        path_dataset_torch_train, path_dataset_torch_val, path_dataset_torch_test,
        cfg_runes, path_results_rules_transformer, path_results_rules_processed_rules, 
        path_results_runes_processed_rules
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}") 
    
        
    # ====================== Evaluate the Models ====================== #
    init_fn = None
    seed_val = 1445
    L.seed_everything(seed_val)
    result_dict = {}
    
    if MAP_NAME_TO_KIND_DATASET[requested_dataset] == 'samples':
        dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
        dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == False
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == False
    elif MAP_NAME_TO_KIND_DATASET[requested_dataset] == 'random_start':
        if num_workers_dataloader > 0:
            init_fn = worker_init_fn
        dataset_train_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_train))
        dataset_test_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_test))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == True
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == True
    
    train_loader = DataLoader(
        dataset_train_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = num_workers_dataloader,
        persistent_workers = False, # ensure the same chunks when testing different models on random-sampling datasets
        worker_init_fn = init_fn    # ensure the same chunks when testing different models on random-sampling datasets 
    )
    
    test_loader = DataLoader(
        dataset_test_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = num_workers_dataloader,
        persistent_workers = False, # ensure the same chunks when testing different models on random-sampling datasets
        worker_init_fn = init_fn    # ensure the same chunks when testing different models on random-sampling datasets
    )
    
    
    for model_kind, access_tuple in MODELS_AND_KEYS.items():
        acc_train_results = []
        ce_train_results = []
        acc_test_results = []
        ce_test_results = []
        if model_kind == 'rules':
            diff_train_acc_results = []
            diff_train_ce_results = []
            diff_test_acc_results = []
            diff_test_ce_results = []
        if model_kind == 'rules' or model_kind == 'transformer':
            with open(path_results_rules_transformer, 'r') as f:
                result_logs = yaml.safe_load(f)
            ckpt_list = result_logs[access_tuple[0]][access_tuple[1]]
            ckpt_list = [entry[0] for entry in ckpt_list]
        
        elif model_kind == 'rules_processed_rules':
            with open(path_results_rules_processed_rules, 'r') as f:
                result_logs = yaml.safe_load(f)
            ckpt_list = result_logs[access_tuple[0]]
            ckpt_list = [entry[0] for entry in ckpt_list]
        
        elif model_kind == 'runes_processed_rules':
            with open(path_results_runes_processed_rules, 'r') as f:
                result_logs = yaml.safe_load(f)
            ckpt_list = result_logs[access_tuple[0]]
            ckpt_list = [entry[0] for entry in ckpt_list]
            
        elif model_kind == 'runes_ablation':
            with open(path_results_rules_transformer, 'r') as f:
                result_logs = yaml.safe_load(f)
            ckpt_list = result_logs[access_tuple[0]][access_tuple[1]]
            ckpt_list = [entry[0] for entry in ckpt_list]
                   
        # assert len(ckpt_list) == 4, f"Unexpected number of model checkpoints: {len(ckpt_list)}"
        for ckpt in ckpt_list:
            model = None
            if model_kind == 'rules':
                model = rules = Rules.load_from_checkpoint(ckpt)
                
                # soft model eval
                rules.symbolic = False
                trainer = L.Trainer(logger=False, enable_progress_bar=False)
                L.seed_everything(seed_val)
                res_train = trainer.test(rules, train_loader)[0]
                train_loss_soft = res_train['test_loss_corrected']
                train_acc_soft = res_train['test_acc_corrected']
                res_test = trainer.test(model, test_loader)[0]
                test_loss_soft = res_test['test_loss_corrected']
                test_acc_soft = res_test['test_acc_corrected']
                
                L.seed_everything(seed_val)
                rules.symbolic = True # use symbolic rule heads for evaluation
                # sparcify Rules (as done for the other evaluations)
                rules.sparcify(train_loader, threshold_min_usage=0.0, threshold_max_usage=1.0)
                model = rules
            
            elif model_kind == 'transformer':
                model = ROPETransformer.load_from_checkpoint(ckpt)
            
            elif model_kind == 'rules_processed_rules':
                model = Rules.load_from_checkpoint(ckpt)
                assert model.symbolic
            
            elif model_kind == 'runes_processed_rules':
                with open(cfg_runes, 'r') as f:
                    config = yaml.safe_load(f)
                model_type = config['model']['type']
                del config['model']['type']
                assert model_type == 'runes'
                
                with open(f"{config['rules_base_model_loc']}/rules_processed_rules_aggregated_results.yaml", 'r') as f:
                    res_dict = yaml.safe_load(f)
                ckpt_list_rules = res_dict['ckpt_list']
                rules = Rules.load_from_checkpoint(ckpt_list_rules[0][0])
                rules.symbolic = True
                config_residual_network = config['model']['transformer_residual_network']
                model = Runes.load_from_checkpoint(ckpt, transformer=ROPETransformer(**config_residual_network), rules=rules)
            
            elif model_kind == 'runes_ablation':
                transformer = ROPETransformer.load_from_checkpoint(ckpt)
                with open(f"{config['rules_base_model_loc']}/rules_processed_rules_aggregated_results.yaml", 'r') as f:
                    res_dict = yaml.safe_load(f)
                ckpt_list_rules = res_dict['ckpt_list']
                rules = Rules.load_from_checkpoint(ckpt_list_rules[0][0])
                rules.symbolic = True
                model = Runes(transformer=transformer, rules=rules)
                
            if model_kind == 'runes_processed_rules' or model_kind == 'runes_ablation':
                assert model.transformer.regular_sampling == train_loader.dataset.metadata['regular_sampling']
            else:
                assert model.regular_sampling == train_loader.dataset.metadata['regular_sampling']

        
            # ====================== Accuracy and loss for the prediction tasks ====================== #
            trainer = L.Trainer(logger=False, enable_progress_bar=False)
            L.seed_everything(seed_val)
            res_train = trainer.test(model, train_loader)[0]
            train_loss = res_train['test_loss_corrected']
            train_acc = res_train['test_acc_corrected']
            res_test = trainer.test(model, test_loader)[0]
            test_loss = res_test['test_loss_corrected']
            test_acc = res_test['test_acc_corrected']
            
            ce_train_results.append(train_loss)
            acc_train_results.append(train_acc)
            ce_test_results.append(test_loss)
            acc_test_results.append(test_acc)
            
            if model_kind == 'rules':
                diff_train_ce_results.append(train_loss - train_loss_soft)
                diff_train_acc_results.append(train_acc - train_acc_soft)
                diff_test_ce_results.append(test_loss - test_loss_soft)
                diff_test_acc_results.append(test_acc - test_acc_soft)
        
        result_dict[model_kind] = {}
        # add results
        for l, metric_name in zip(
            [ce_train_results, acc_train_results, ce_test_results, acc_test_results], 
            ['train_data_loss', 'train_data_acc', 'test_data_loss', 'test_data_acc']
        ):
            result_dict[model_kind][metric_name] = {
                    'min': min(l),
                    'max': max(l),
                    'mean': np.mean(l).item(),
                    'std': np.std(l, ddof=1).item(),
                    'list': l
                }
            
        
        if model_kind == 'rules':
            for l, metric_name in zip(
                [diff_train_ce_results, diff_train_acc_results, diff_test_ce_results, diff_test_acc_results], 
                ['diff_train_data_loss', 'diff_train_data_acc', 'diff_test_data_loss', 'diff_test_data_acc']
            ):
                l = [abs(x) for x in l]
                result_dict[model_kind][metric_name] =  {
                        'min': min(l),
                        'max': max(l),
                        'mean': np.mean(l).item(),
                        'std': np.std(l, ddof=1).item(),
                        'list': l
                    }
                
            
    with open(f'{log_results}/results_mean_std.yaml', 'w') as f:
        yaml.dump(result_dict, f, default_flow_style=False)