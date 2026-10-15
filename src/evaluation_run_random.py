import argparse
import datetime
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint
)
import lightning.pytorch.loggers as pl_loggers
import os
from pathlib import Path
import pickle
import time
import torch
from torch.utils.data import DataLoader
import yaml

from eval_utils import(
    str2bool
)
from metrics import calculate_support_confidence
from metrics_aggregation import (
    TrackerSingleDataset
)
from data_generation import (
    CFG_DIR, 
    DATA_DIR,
    LOGGING_DIR,
)
from data.utils import (
    SequenceDataset,
    custom_collate_fn
)
from models.rules import (
    Rules
)

RANDOM_SEEDS = [631, 8567, 49415, 68230]

# ====================== Run ====================== #
# Run Rules on a synthetic random dataset.
if __name__ == '__main__':
    
    # ====================== Setup ====================== #
    print(f"Cuda is available {torch.cuda.is_available()}")
    full_run_time = time.time()
    parser = argparse.ArgumentParser(description='Random Synthetic Data Evaluation Script')
    parser.add_argument('--regular', type=str2bool, required=True, help="Whether the considered synthetic dataset is regularly sampled.")
    parser.add_argument('--dataset_num', type=int, required=True, help="The number of the dataset as numbered by data_generation.materialize_datasets_and_configs_random_regular()")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0 the main process spawns loaders.")
    args = parser.parse_args()
    regular = args.regular
    dataset_num = args.dataset_num
    num_workers_dataloader = args.num_workers_dataloader
    
    persistent_workers = num_workers_dataloader > 0
    regularity_str = 'regular' if regular else  'irregular'
    log_results = f'{LOGGING_DIR}/synthetic/{regularity_str}_random/dataset_{dataset_num}'
    if not os.path.exists(log_results):
        os.makedirs(log_results)
    tracker = TrackerSingleDataset()
    
    print("Checking if all dataset and config files exists...")
    path_dataset_list_train = f'{DATA_DIR}/synthetic/{regularity_str}_random/dataset_{dataset_num}.pt'
    path_dataset_torch_train = f'{DATA_DIR}/synthetic/{regularity_str}_random/dataset_{dataset_num}_torch.pt'
    
    path_cfg_file_rules = f'{CFG_DIR}/synthetic/rules/{regularity_str}_random/dataset_{dataset_num}.yaml'
    
    for p in [path_dataset_list_train,  path_dataset_torch_train, path_cfg_file_rules]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}") 
        
    dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
    assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
        
    # ====================== Train Each Model ====================== #
    cfg_path = path_cfg_file_rules
    print(f"Loading config {cfg_path} and setting up datasets for training...")
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)    
    assert config['dataset']['batching_strategy'] == 'samples', "Batching strategy for synthetic datasets should be 'samples'." 
    model_type = config['model']['type']
    del config['model']['type']
    
    for i, random_seed in enumerate(RANDOM_SEEDS):
        print(f"Running {model_type} on seed number {i+1}/{len(RANDOM_SEEDS)}.")
        logger = pl_loggers.CSVLogger(save_dir=config['logging']['save_dir'], name=f"{config['logging']['name']}_seed_{random_seed}")
        print(f"Preparing to log to {logger.log_dir}")
        # set random seed for reproducability
        L.seed_everything(random_seed, workers=True)
        start_time = time.time()
        model = None
        model_ckpt_callback = None
        callbacks = None
        val_loader = None
        
        optimization_dict = config['optimization']
        train_loader = DataLoader(
            dataset_train_torch,
            batch_size = optimization_dict['batch_size'], 
            collate_fn = custom_collate_fn,
            shuffle = True, 
            pin_memory = True,
            num_workers = num_workers_dataloader,
            persistent_workers = persistent_workers,
        )

        if model_type == "rules":
            model = Rules(
                **config['model'],
                optimization_dict=optimization_dict,
            )
            model_ckpt_callback = ModelCheckpoint(every_n_epochs=64)
            callbacks = [
                model_ckpt_callback,
            ]
            
        else:
            raise ValueError(f"Model type {model_type} not recognized.")

        trainer = L.Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=config['optimization']['max_epochs'],
            enable_progress_bar=False # can be switched on if this is wanted
        )
        trainer.fit(model, train_loader, val_loader)
        
        training_time = time.time()-start_time  
        ckpt_path = model_ckpt_callback.best_model_path
        if model_type == 'rules':
            # since we only ckpt every 64 epochs, we need to save the final model
            if ckpt_path == '':
                ckpt_path_new = f'{logger.log_dir}/checkpoints/last.ckpt'
            else:
                ckpt_path_new = str(Path(ckpt_path).parent / "last.ckpt")
                os.remove(ckpt_path)
            ckpt_path = ckpt_path_new
            trainer.save_checkpoint(ckpt_path)
            
        metric = trainer.callback_metrics['train_loss_corrected'].item() 
        tracker.track_model(model_type, ckpt_path, logger.log_dir, metric, training_time, i)
        
        print(f"Running {model_type} on seed number {i+1}/{len(RANDOM_SEEDS)} took {datetime.timedelta(seconds=training_time)}")
    
    print(f"Done with training all models. Took {datetime.timedelta(seconds=time.time() - full_run_time)}") 
    print(f"Logging results to {log_results}...")
    # ====================== Get all metrics ====================== # 
    result_dict = {}      
    # save the tracker in case we need it again
    with open(f"{log_results}/tracker.pkl", "wb") as f:
        pickle.dump(tracker, f)
        
    print("Training for all models done, calculating metrics and aggregating results...")
    train_loader = DataLoader(
        dataset_train_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = num_workers_dataloader,
        persistent_workers = persistent_workers,
    )
    best_ckpt_dict = tracker.get_best_ckpt()
    rules = Rules.load_from_checkpoint(best_ckpt_dict['rules'][0][0])
    rules.symbolic = True # use symbolic rule heads for evaluation
    # sparcify the Rules model
    rules.sparcify(train_loader)
    
    res_train = trainer.test(model, train_loader)[0]
    result_dict['train_data_loss'] = res_train['test_loss_corrected']
    result_dict['train_data_acc'] = res_train['test_acc_corrected']
    
    list_dataset_train = torch.load(path_dataset_list_train)
    rule_set, _ = rules.get_rule_set(quantile_q = 0.95, max_min_threshold = 0.9, inclusion_threshold = 0.95)
    result_dict['size_rule_set'] = len(rule_set)
    support_conf_dict = calculate_support_confidence(list_dataset_train['samples'], rule_set, regular_sampling=regular)
    del support_conf_dict['num_rule_holds']
    result_dict.update(support_conf_dict)
    result_dict['max_confidence'] = 0.0 if len(rule_set) == 0 else max(support_conf_dict['rule_confidence_individual'])
    result_dict['max_adjusted_confidence'] = 0.0 if len(rule_set) == 0 else max(support_conf_dict['adjusted_rule_confidence_individual'])
    result_dict['max_rel_support'] = 0.0 if len(rule_set) == 0 else max(support_conf_dict['rel_rule_support_individual'])
    with open(f"{log_results}/aggregated_results.yaml", "w") as f:
        yaml.dump(result_dict, f, default_flow_style=False)
    
    print(f"Done with all results. Took {datetime.timedelta(seconds=time.time() - full_run_time)}")