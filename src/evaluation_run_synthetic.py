import argparse
import datetime
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
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

from metrics_aggregation import (
    TrackerSingleDataset,
    aggregate_results_single_dataset_synthetic
)
from data_generation import (
    CFG_DIR, 
    DATA_DIR,
    LOGGING_DIR,
    create_cfg_file_path_synthetic,
    create_dataset_file_path_synthetic,
    create_logging_dir_path_synthetic
)
from data.utils import (
    SequenceDataset,
    custom_collate_fn
)
from eval_utils import(
    str2bool
)
from models.rope_transformer import ROPETransformer
from models.rules import (
    Rules,
    initialize_from_rules
)

RANDOM_SEEDS = [631, 8567, 49415, 68230]

# ====================== Run ====================== #
# Evaluate models for a specific synthetic dataset.
if __name__ == '__main__':
    
    # ====================== Setup ====================== #
    print(f"Cuda is available {torch.cuda.is_available()}")
    full_run_time = time.time()
    parser = argparse.ArgumentParser(description='Synthetic Data Evaluation Script')
    parser.add_argument('--regular', type=str2bool, required=True, help="Whether the considered synthetic dataset is regularly sampled.")
    parser.add_argument('--setting_num', type=int, required=True, help="The setting for the dataset as numbered by data_generation.materialize_datasets_and_configs()")
    parser.add_argument('--dataset_num', type=int, required=True, help="The number of the dataset as numbered by data_generation.materialize_datasets_and_configs()")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0 the main process spawns loaders.")
    args = parser.parse_args()
    regular = args.regular
    setting_num = args.setting_num
    dataset_num = args.dataset_num
    num_workers_dataloader = args.num_workers_dataloader
    
    # For debugging:
    # regular = True
    # setting_num = 0
    # dataset_num = 0
    # num_workers_dataloader = 0
    
    persistent_workers = num_workers_dataloader > 0
    regularity_str = 'regular' if regular else  'irregular'
    log_results = create_logging_dir_path_synthetic(LOGGING_DIR, regular, setting_num, dataset_num)
    if not os.path.exists(log_results):
        os.makedirs(log_results)
    tracker = TrackerSingleDataset()
    
    print("Checking if all dataset and config files exists...")
    path_dataset_list_train = create_dataset_file_path_synthetic(data_dir=DATA_DIR, regular=regular, setting_num=setting_num, 
                                                      dataset_num=dataset_num, mode='train', is_torch=False)
    path_dataset_list_val = create_dataset_file_path_synthetic(data_dir=DATA_DIR, regular=regular, setting_num=setting_num,
                                                    dataset_num=dataset_num, mode='val', is_torch=False)
    path_dataset_list_test = create_dataset_file_path_synthetic(data_dir=DATA_DIR, regular=regular, setting_num=setting_num, 
                                                     dataset_num=dataset_num, mode='test', is_torch=False)
    
    path_dataset_torch_train = create_dataset_file_path_synthetic(data_dir=DATA_DIR, regular=regular, setting_num=setting_num, 
                                                      dataset_num=dataset_num, mode='train', is_torch=True)
    path_dataset_torch_val = create_dataset_file_path_synthetic(data_dir=DATA_DIR, regular=regular, setting_num=setting_num,
                                                    dataset_num=dataset_num, mode='val', is_torch=True)
    path_dataset_torch_test = create_dataset_file_path_synthetic(data_dir=DATA_DIR, regular=regular, setting_num=setting_num, 
                                                     dataset_num=dataset_num, mode='test', is_torch=True)
    
    path_cfg_file_gt_model = create_cfg_file_path_synthetic(CFG_DIR, 'gt_model', True, setting_num, dataset_num)
    path_cfg_file_rules = create_cfg_file_path_synthetic(CFG_DIR, 'rules', True, setting_num, dataset_num)
    path_cfg_file_transformer = create_cfg_file_path_synthetic(CFG_DIR, 'transformer', True, setting_num, dataset_num)
    
    for p in [
        path_dataset_list_train, path_dataset_list_val, path_dataset_list_test,
        path_dataset_torch_train, path_dataset_torch_val, path_dataset_torch_test,
        path_cfg_file_gt_model, path_cfg_file_rules, path_cfg_file_transformer
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}") 
        
    dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
    assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
    dataset_val_torch = SequenceDataset(torch.load(path_dataset_torch_val))
    assert dataset_val_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
        
    # ====================== Train Each Model ====================== #
    for cfg_path in [path_cfg_file_rules, path_cfg_file_gt_model, path_cfg_file_transformer]:
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
                
            elif model_type == "rules_gt":
                ground_truth_rules = dataset_train_torch.metadata['rules']
                rule_heads = [rule[0] for rule in ground_truth_rules]
                model: Rules = initialize_from_rules(
                    rule_heads = rule_heads,
                    **config['model'],
                    optimization_dict=optimization_dict,
                )
                model_ckpt_callback = ModelCheckpoint(filename='{epoch}-{train_loss_corrected:.2f}', monitor='train_loss_corrected', mode='min')
                callbacks = [
                    model_ckpt_callback,
                    EarlyStopping(monitor='train_loss_corrected', mode='min', strict=True, patience=3*optimization_dict['lr_patience']+1, min_delta=0.001) 
                ]
                
            elif model_type == "transformer":
                model = ROPETransformer(
                    **config['model'],
                    optimization_dict=optimization_dict,
                )
                model_ckpt_callback = ModelCheckpoint(filename='{epoch}-{val_loss_corrected:.2f}', monitor='val_loss_corrected', mode='min')
                callbacks = [
                    model_ckpt_callback,
                    EarlyStopping(monitor='val_loss_corrected', mode='min', strict=True, patience=3*optimization_dict['scheduler']['params']['patience']+1) 
                ]
                val_loader = DataLoader(
                    dataset_val_torch,
                    batch_size = optimization_dict['batch_size'], 
                    collate_fn = custom_collate_fn,
                    shuffle = False, 
                    pin_memory = True,
                    num_workers = num_workers_dataloader,
                    persistent_workers = persistent_workers,
                )
                
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
                
            metric = trainer.callback_metrics['train_loss_corrected'].item() if model_type == 'rules' else model_ckpt_callback.best_model_score.item()
            tracker.track_model(model_type, ckpt_path, logger.log_dir, metric, training_time, i)
            
            print(f"Running {model_type} on seed number {i+1}/{len(RANDOM_SEEDS)} took {datetime.timedelta(seconds=training_time)}")
    
    print(f"Done with training all models. Took {datetime.timedelta(seconds=time.time() - full_run_time)}") 
    print(f"Logging results to {log_results}...")
    # ====================== Get all metrics ====================== #       
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
    dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
    assert dataset_test_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
    test_loader = DataLoader(
        dataset_test_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = num_workers_dataloader,
        persistent_workers = persistent_workers,
    )
    
    aggregate_results_single_dataset_synthetic(
        tracker = tracker,
        log_dir = log_results,
        path_dataset_list_train = path_dataset_list_train,
        path_dataset_list_test = path_dataset_list_test,
        train_data_loader = train_loader,
        test_data_loader = test_loader  
    )
    
    print(f"Done with all results. Took {datetime.timedelta(seconds=time.time() - full_run_time)}")