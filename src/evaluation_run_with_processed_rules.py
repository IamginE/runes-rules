import argparse
import datetime
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)
import lightning.pytorch.loggers as pl_loggers
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import time
import torch
from torch.utils.data import DataLoader
import yaml

from data.real_data import (
    AVAILABLE_DATASETS,
    MAP_NAME_TO_KIND_DATASET,
)
from data.utils import (
    SequenceDataset,
    SequenceDatasetRandomSampling,
    custom_collate_fn,
)
from eval_utils import(
    create_cfg_file_path,
    create_dataset_path,
    create_logging_dir_path,
    str2bool,
    worker_init_fn
)
from models.rules import (
    Rules,
    initialize_from_rules
)


RANDOM_SEEDS = [631, 8567, 49415, 68230]
# ====================== Run ====================== #
# Evaluate Rules with processed rules.
if __name__ == '__main__':
    
    # ====================== Setup ====================== #
    print(f"Cuda is available {torch.cuda.is_available()}")
    full_run_time = time.time()
    
    parser = argparse.ArgumentParser(description='Evaluation Script for Rules initialized with processed rules.')
    parser.add_argument('--synthetic', type=str2bool, required=True, help="Whether the considered dataset is synthetic.")
    parser.add_argument('--dataset', type=str, required=True, help="The name of the dataset to run the model on.")
    parser.add_argument('--regular', type=str2bool, required=True, help="Whether the considered dataset is regularly sampled.")
    parser.add_argument('--setting_num', type=int, required=False, default=None, help="The setting number for synthetic data.")
    parser.add_argument('--split_off_val_rules', type=str2bool, required=True, help="Whether to use a validation split for Rules models.\n"
        + "If True, a validation set is split off from the training data and used to select the best generalizing model.\n" 
        + "If False, the model is trained on the full training data (without splitting off validation data) and the model that explains the training data best (lowest training loss) is selected.\n")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0, the main process spawns the loaders.")
    
    args = parser.parse_args()
    synthetic = args.synthetic
    dataset_name = args.dataset
    regular = args.regular
    setting_num = args.setting_num
    split_off_val_rules = args.split_off_val_rules
    num_workers_dataloader = args.num_workers_dataloader
    persistent_workers = num_workers_dataloader > 0
    
    # For debugging
    # synthetic = False
    # dataset_name = 'rolling_mill'
    # regular = True
    # setting_num = 14
    # split_off_val_rules = True
    # num_workers_dataloader = 4
    # persistent_workers = num_workers_dataloader > 0
    
    if synthetic:
        assert setting_num is not None, "Setting number has to be given for synthetic data."
    else:
        assert dataset_name in AVAILABLE_DATASETS, f"Requested non-synthetic dataset '{dataset_name}' but the only available datasets are {AVAILABLE_DATASETS}."

    print("Checking if all dataset and config files exists...")
    log_results = create_logging_dir_path(synthetic, dataset_name, regular, setting_num)
    rule_file = f'{log_results}/processed_ruleset_rules.pkl'
    train_mode_str = 'train' if synthetic else 'train_val'
    path_dataset_list_train = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode=train_mode_str,
                                                  is_torch=False, setting_num=setting_num)
    path_dataset_list_test = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='test',
                                                  is_torch=False, setting_num=setting_num)
    
    path_dataset_torch_train = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode=train_mode_str,
                                                  is_torch=True, setting_num=setting_num)
    path_dataset_torch_test = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='test',
                                                  is_torch=True, setting_num=setting_num)
    cfg_path =  create_cfg_file_path(synthetic, dataset_name, 'rules_processed_rules', regular, setting_num)
    
    for p in [
        cfg_path, path_dataset_list_train, path_dataset_list_test,
        path_dataset_torch_train, path_dataset_torch_test, rule_file
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")
        
    if split_off_val_rules:
        path_dataset_list_val = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='val',
                                                  is_torch=False, setting_num=setting_num)
        path_dataset_torch_val = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='val',
                                                  is_torch=True, setting_num=setting_num)
        for p in [
            path_dataset_list_val, path_dataset_torch_val
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Expected file not found: {p}")
    
    
    with open(rule_file, 'rb') as f:
        rule_set: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]] = pickle.load(f)
    rule_heads = [rule[0] for rule in rule_set]
    init_fn = None
    print(f"Loading config {cfg_path} and setting up datasets for training...")
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    assert config['model']['type'] == 'rules_processed_rules'
    del config['model']['type']
    
    if synthetic:
        dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
        dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
        assert config['dataset']['batching_strategy'] == 'samples', "Batching strategy should be 'samples'."
        if split_off_val_rules:
            dataset_val_torch = SequenceDataset(torch.load(path_dataset_torch_val))
            assert dataset_val_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
    else:
        if MAP_NAME_TO_KIND_DATASET[dataset_name] == 'samples':
            dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
            dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
            assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
            assert dataset_test_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
            assert config['dataset']['batching_strategy'] == 'samples', "Batching strategy should be 'samples'." 
            if split_off_val_rules:
                dataset_val_torch = SequenceDataset(torch.load(path_dataset_torch_val))
                assert dataset_val_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
            
        
        elif MAP_NAME_TO_KIND_DATASET[dataset_name] == 'random_start':
            dataset_train_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_train))
            dataset_test_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_test))
            if num_workers_dataloader > 0:
                init_fn = worker_init_fn
            assert dataset_train_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
            assert dataset_test_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
            assert config['dataset']['batching_strategy'] == 'random_start', "Batching strategy should be 'random_start'."
            if split_off_val_rules:
                dataset_val_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_val))
                assert dataset_val_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
        
    
    best_ckpt_list = []
    for i, random_seed in enumerate(RANDOM_SEEDS):
        print(f"Running Rules with processed rules on seed number {i+1}/{len(RANDOM_SEEDS)}.")
        logger = pl_loggers.CSVLogger(save_dir=config['logging']['save_dir'], name=f"rules_processed_rules_seed_{random_seed}")
        print(f"Preparing to log to {logger.log_dir}")
        # set random seed for reproducability
        L.seed_everything(random_seed, workers=True)
        start_time = time.time()
        
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
        
        model: Rules = initialize_from_rules(
            rule_heads = rule_heads,
            **config['model'],
            optimization_dict=optimization_dict,
        )
        
        if split_off_val_rules:
            val_loader = DataLoader(
                dataset_val_torch,
                batch_size = optimization_dict['batch_size'], 
                collate_fn = custom_collate_fn,
                shuffle = True, 
                pin_memory = True,
                num_workers = num_workers_dataloader,
                persistent_workers = persistent_workers,
                worker_init_fn = init_fn
            )
            
            model_ckpt_callback = ModelCheckpoint(filename='{epoch}-{val_loss_corrected:.2f}', monitor='val_loss_corrected', mode='min')
            callbacks = [
                model_ckpt_callback,
                EarlyStopping(monitor='val_loss_corrected', mode='min', strict=True, patience=3*optimization_dict['lr_patience']+1, min_delta=0.001) 
            ]
        
        else:
            model_ckpt_callback = ModelCheckpoint(filename='{epoch}-{train_loss_corrected:.2f}', monitor='train_loss_corrected', mode='min')
            callbacks = [
                model_ckpt_callback,
                EarlyStopping(monitor='train_loss_corrected', mode='min', strict=True, patience=3*optimization_dict['lr_patience']+1, min_delta=0.001) 
            ]
        
        trainer = L.Trainer(
            logger=logger,
            callbacks=callbacks,
            max_epochs=config['optimization']['max_epochs'],
            enable_progress_bar=False # can be switched on if this is wanted
        )
        trainer.fit(model, train_loader, val_loader)
           
         
        training_time = time.time()-start_time  
        ckpt_path = model_ckpt_callback.best_model_path  
        metric = model_ckpt_callback.best_model_score.item()
        best_ckpt_list.append((ckpt_path, logger.log_dir, metric, training_time, i))
        
        print(f"Running Rules with processed rules on seed number {i+1}/{len(RANDOM_SEEDS)} took {datetime.timedelta(seconds=training_time)}")
     
    # =================== Log the Desired Metrics ================== 
    print(f"Done with training all models. Took {datetime.timedelta(seconds=time.time() - full_run_time)}") 
    print(f"Logging results to {log_results}...")
    model_name = 'rules_processed_rules'
    result_dict = {model_name: {}}
    # get the best model according to the training loss
    best_ckpt_list.sort(key=lambda x: x[2])
    model = Rules.load_from_checkpoint(best_ckpt_list[0][0])
    result_dict['ckpt_list'] = [list(e) for e in best_ckpt_list]
    
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
    
    # ================= Accuracy and loss for the prediction tasks =================
    trainer = L.Trainer(logger=False, enable_progress_bar=False)
    L.seed_everything(1445)
    res_train = trainer.test(model, train_loader)[0]
    result_dict[model_name]['train_data_loss'] = res_train['test_loss_corrected']
    result_dict[model_name]['train_data_acc'] = res_train['test_acc_corrected']
    result_dict[model_name]['time_training'] = best_ckpt_list[0][3]
    res_test = trainer.test(model, test_loader)[0]
    result_dict[model_name]['test_data_loss'] = res_test['test_loss_corrected']
    result_dict[model_name]['test_data_acc'] = res_test['test_acc_corrected']
        
    # trainer moves back the models to cpu after testing
    if model.device == torch.device('cpu') and torch.cuda.is_available():
        model.to(torch.device('cuda')) 
       
    
    # if synthetic and os.path.isfile(f"{log_results}/tracker.pkl"):
    #     # this only works if we have trained the GT model before
    #     with open(f"{log_results}/tracker.pkl", 'rb') as f:
    #         tracker: TrackerSingleDataset = pickle.load(f) 
    #     
    #     tracker_dict = tracker.get_best_ckpt()
    #     if len(tracker_dict['rules_gt']) > 0:
    #         rules_gt = Rules.load_from_checkpoint(tracker_dict['rules_gt'][0][0])
    #         assert rules_gt.symbolic == True # model should always be symbolic if initialized from rules
    #         assert rules_gt.gt_rule_heads == True 
    #         if torch.cuda.is_available():
    #             rules_gt.to(torch.device('cuda')) 
    #         
    #         # ================= Agreement to Rules with GT rule heads (antecedents) ================= #
    #         agreement_gt_pred = calculate_agreement_prediction_models(
    #             model_gt = rules_gt,
    #             model = model,
    #             device_gt = rules_gt.device,
    #             device_model = model.device,
    #             data_loader = train_loader
    #         )
    #         result_dict[model_name]['agreement_gt_pred'] = agreement_gt_pred['agreement_prediction']
    #         result_dict[model_name]['js_div_gt'] = agreement_gt_pred['avg_js']
    #         result_dict[model_name]['kl_gt_model'] = agreement_gt_pred['avg_kl_p_q']
    #         result_dict[model_name]['kl_model_gt'] = agreement_gt_pred['avg_kl_q_p']
            
    # ====================== Print Results to File and Loss Curve Plotting ======================
    with open(f"{log_results}/rules_processed_rules_aggregated_results.yaml", "w") as f:
        yaml.dump(result_dict, f, default_flow_style=False)
        
    metrics = {
        'train_loss_corrected': 'Train Loss',
        'train_acc_corrected': 'Train Accuracy',
    }
    if split_off_val_rules:
        metrics.update({
            'val_loss_corrected': 'Validation Loss',
            'val_acc_corrected': 'Validation Accuracy'
        })
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(16, 12))
    
    for _, metric_dir, _, _, run_id  in best_ckpt_list: 
        metric_file = os.path.join(metric_dir, 'metrics.csv')
        df = pd.read_csv(metric_file)
        df = df.groupby('epoch').last().reset_index()
        
        for idx, (metric_key, _) in enumerate(metrics.items()):
            ax = axs[idx]
            if metric_key in df.columns:
                ax.plot(
                    df['epoch'],
                    df[metric_key],
                    label=f'Rules-Proc. Rules - run {run_id}',
                    linestyle='-',
                    alpha=0.8
                )
            
    for idx, (_, metric_title) in enumerate(metrics.items()):
        ax = axs[idx]
        ax.set_title(metric_title)
        ax.set_ylabel(metric_title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
        
    fig.tight_layout()
    fig.savefig(f"{log_results}/rules_processed_rules_training_plots.png")
    plt.close()
      
    print(f"Done with all results. Took {datetime.timedelta(seconds=time.time() - full_run_time)}")