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
    str2bool,
    worker_init_fn
)
from metrics_aggregation import (
    TrackerSingleDataset,
    aggregate_results_single_dataset_real
)
from models.rope_transformer import ROPETransformer
from models.rules import (
    Rules
)

RANDOM_SEEDS = [631, 8567, 49415, 68230]
# RANDOM_SEEDS = [631]
IRREGULAR_TO_REGULAR_MAX_DIST = {
    'holidays_irregular': 11,
    'chorales': 22, 
    'amazon_purchases':  64 - 1
}

# ====================== Run ====================== #
# Evaluate models for a specific real dataset.
if __name__ == '__main__':
    
    # ====================== Setup ====================== #
    print(f"Cuda is available {torch.cuda.is_available()}")
    full_run_time = time.time()
    parser = argparse.ArgumentParser(description='Real Data Evaluation Script')
    parser.add_argument('--dataset', type=str, required=True, help=f"The dataset to run the model on. Has to be one of {AVAILABLE_DATASETS}.")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0, the main process spawns the loaders.")
    parser.add_argument('--run_with_transformer', type=str2bool, required=True, help="Whether to run a transformer as a baseline as well.")
    parser.add_argument('--treat_irregular_as_regular', type=str2bool, required=False, default=False, help=f"Whether to treat a dataset with irregular sampling times as regularly sampled (ignoring timestamps).")
    parser.add_argument('--postfix', type=str, required=False, default='', help=f"The postfix for configs and logging.")
    parser.add_argument('--split_off_val_rules', type=str2bool, required=True, help="Whether to use a validation split for Rules models.\n"
        + "If True, a validation set is split off from the training data and used to select the best generalizing model.\n" 
        + "If False, the model is trained on the full training data (without splitting off validation data) and the model that explains the training data best (lowest training loss) is selected.\n")
    
    args = parser.parse_args()
    requested_dataset = args.dataset
    num_workers_dataloader = args.num_workers_dataloader
    run_with_transformer = args.run_with_transformer
    treat_irregular_as_regular = args.treat_irregular_as_regular
    postfix = args.postfix
    split_off_val_rules = args.split_off_val_rules
    
    # For debugging:
    # requested_dataset = 'holidays_irregular'
    # num_workers_dataloader = 0
    # postfix = ''
    # run_with_transformer = True
    # split_off_val_rules = True
    # treat_irregular_as_regular = True
    
    postfix_treat_irregular_as_regular = '_regular' if treat_irregular_as_regular else ''
    assert requested_dataset in AVAILABLE_DATASETS, f"Requested dataset '{requested_dataset}' but the only available datasets are {AVAILABLE_DATASETS}."
    persistent_workers = num_workers_dataloader > 0
    log_results = f'{LOGGING_DIR}/real/{requested_dataset}{postfix}'
    if not os.path.exists(log_results):
        os.makedirs(log_results)
    tracker = TrackerSingleDataset()
    
    path_dataset_list_train_val = f'{REAL_DATA_DIR}/{requested_dataset}/train_val.pt'
    path_dataset_list_test      = f'{REAL_DATA_DIR}/{requested_dataset}/test.pt' 
    path_dataset_list_train = None
    path_dataset_list_val   = None
    if run_with_transformer or split_off_val_rules:
        path_dataset_list_train = f'{REAL_DATA_DIR}/{requested_dataset}/train.pt'
        path_dataset_list_val   = f'{REAL_DATA_DIR}/{requested_dataset}/val.pt'
    
    path_dataset_torch_train_val = f'{REAL_DATA_DIR}/{requested_dataset}/train_val_torch.pt'
    path_dataset_torch_test      = f'{REAL_DATA_DIR}/{requested_dataset}/test_torch.pt'
    path_dataset_torch_train = None
    path_dataset_torch_val   = None
    if run_with_transformer or split_off_val_rules:
        path_dataset_torch_train = f'{REAL_DATA_DIR}/{requested_dataset}/train_torch.pt'
        path_dataset_torch_val   = f'{REAL_DATA_DIR}/{requested_dataset}/val_torch.pt'
    
    
    path_cfg_rules = f'{CFG_DIR}/real/rules/{requested_dataset}{postfix}.yaml'
    path_cfg_file_transformer = None
    if run_with_transformer:
        path_cfg_file_transformer = f'{CFG_DIR}/real/transformer/{requested_dataset}{postfix}.yaml'
    
    for p in [
        path_dataset_list_train, path_dataset_list_val, path_dataset_list_train_val, path_dataset_list_test,
        path_dataset_torch_train, path_dataset_torch_val, path_dataset_torch_train_val, path_dataset_torch_test,
        path_cfg_rules, path_cfg_file_transformer
    ]:
        if p is None:
            continue
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}") 
    
        
    # ====================== Train the Models ====================== #
    cfgs = [path_cfg_rules, path_cfg_file_transformer] if run_with_transformer else [path_cfg_rules]
    quantile_q, inclusion_threshold = None, None
    init_fn = None
    for cfg_path in cfgs:
        print(f"Loading config {cfg_path} and setting up datasets for training...")
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        model_type = config['model']['type']
        del config['model']['type']
        if treat_irregular_as_regular:
            config['model']['regular_sampling'] = True # overwrite the config
            config['model']['treat_irregular_as_regular'] = True
            if model_type == 'rules':
                config['model']['max_relative_distance_irregular_as_regular'] = IRREGULAR_TO_REGULAR_MAX_DIST[requested_dataset]
        
        train_path_torch = path_dataset_torch_train if model_type == 'transformer' or split_off_val_rules else path_dataset_torch_train_val
        if MAP_NAME_TO_KIND_DATASET[requested_dataset] == 'samples':
            dataset_train_torch = SequenceDataset(torch.load(train_path_torch))
            assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
            assert config['dataset']['batching_strategy'] == 'samples', "Batching strategy should be 'samples'." 
            if split_off_val_rules or (run_with_transformer and model_type == 'transformer'):
                dataset_val_torch = SequenceDataset(torch.load(path_dataset_torch_val))
                assert dataset_val_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."  
        
        elif MAP_NAME_TO_KIND_DATASET[requested_dataset] == 'random_start':
            dataset_train_torch = SequenceDatasetRandomSampling(torch.load(train_path_torch))
            if num_workers_dataloader > 0:
                init_fn = worker_init_fn
            assert dataset_train_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
            assert config['dataset']['batching_strategy'] == 'random_start', "Batching strategy should be 'random_start'."
            if split_off_val_rules or (run_with_transformer and model_type == 'transformer'):
                dataset_val_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_val))
                assert dataset_val_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset." 
        
        if treat_irregular_as_regular:
            assert not dataset_train_torch.metadata['regular_sampling'], "Trying to use 'treat_irregular_as_regular' with a regular dataset."  
        
        for i, random_seed in enumerate(RANDOM_SEEDS):
            print(f"Running {model_type} on seed number {i+1}/{len(RANDOM_SEEDS)}.")
            logger = pl_loggers.CSVLogger(save_dir=config['logging']['save_dir'], name=f"{config['logging']['name']}{postfix_treat_irregular_as_regular}_seed_{random_seed}")
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
                worker_init_fn = init_fn
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
                quantile_q = config['discretization']['quantile_q']
                inclusion_threshold = config['discretization']['inclusion_threshold']
            
            elif model_type == "transformer":
                model = ROPETransformer(
                    **config['model'],
                    optimization_dict=optimization_dict,
                )
                model_ckpt_callback = ModelCheckpoint(filename='{epoch}-{val_loss_corrected:.2f}', monitor='val_loss_corrected', mode='min')
                callbacks = [
                    model_ckpt_callback,
                    EarlyStopping(monitor='val_loss_corrected', mode='min', strict=True, stopping_threshold=0.005,
                                  patience=3*optimization_dict['scheduler']['params']['patience']+1) 
                ]
                val_loader = DataLoader(
                    dataset_val_torch,
                    batch_size = optimization_dict['batch_size'], 
                    collate_fn = custom_collate_fn,
                    shuffle = False, 
                    pin_memory = True,
                    num_workers = num_workers_dataloader,
                    persistent_workers = persistent_workers,
                    worker_init_fn = init_fn
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
    if split_off_val_rules:
        L.seed_everything(1445)
        trainer = L.Trainer(logger=False, enable_progress_bar=False)
        # calculate the validation losses and replace the training losses in the tracker for Rules
        val_loader = DataLoader(
            dataset_val_torch,
            batch_size = optimization_dict['batch_size'], 
            collate_fn = custom_collate_fn,
            shuffle = False, 
            pin_memory = True,
            num_workers = num_workers_dataloader,
            persistent_workers = False,
            worker_init_fn = init_fn
        )
        new_tracker_list = []
        for ckpt_path, log_dir, metric, training_time, run_id in tracker.tracking_dict['rules']:
            L.seed_everything(1445)
            rules = Rules.load_from_checkpoint(ckpt_path)
            rules.symbolic = True # use symbolic rule heads for evaluation
            res_val = trainer.test(rules, val_loader)[0]
            metric = res_val['test_loss_corrected']
            new_tracker_list.append((ckpt_path, log_dir, metric, training_time, run_id))
        tracker.tracking_dict['rules'] = new_tracker_list
        
    # save the tracker in case we need it again
    with open(f"{log_results}/tracker{postfix_treat_irregular_as_regular}.pkl", "wb") as f:
        pickle.dump(tracker, f)
        
    print("Training for all models done, calculating metrics and aggregating results...")
    # NOTE: could consider rolling window evaluation for random starting point datasets, but a fair approach would require starting the model at every single starting point
    seed_val = 1445
    L.seed_everything(seed_val)
    
    train_path_torch = path_dataset_torch_train if split_off_val_rules else path_dataset_torch_train_val
    train_path_list = path_dataset_list_train if split_off_val_rules else path_dataset_list_train_val
    if MAP_NAME_TO_KIND_DATASET[requested_dataset] == 'samples':
        dataset_train_torch = SequenceDataset(torch.load(train_path_torch))
        dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == False
    elif MAP_NAME_TO_KIND_DATASET[requested_dataset] == 'random_start':
        if num_workers_dataloader > 0:
            init_fn = worker_init_fn
        dataset_train_torch = SequenceDatasetRandomSampling(torch.load(train_path_torch))
        dataset_test_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_test))
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
    
    aggregate_results_single_dataset_real(
        tracker = tracker,
        log_dir = log_results,
        path_dataset_list_train = train_path_list,
        path_dataset_list_test = path_dataset_list_test,
        train_data_loader = train_loader,
        test_data_loader = test_loader,
        random_seed = seed_val,
        quantile_q = quantile_q,
        inclusion_threshold = inclusion_threshold,
        treat_irregular_as_regular = treat_irregular_as_regular
    )
    
    print(f"Done with all results. Took {datetime.timedelta(seconds=time.time() - full_run_time)}")