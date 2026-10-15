import argparse
import datetime
import lightning as L
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
import yaml

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
    create_cfg_file_path,
    create_dataset_path,
    create_logging_dir_path,
    str2bool,
    worker_init_fn,
)
from models.interpretable_prediction_baselines import (
    PPM_MarkovModel
)


# ====================== Run ====================== #
# Train PPM Markov Models
if __name__ == '__main__':
    
    # ====================== Setup ====================== #
    full_run_time = time.time()
    
    parser = argparse.ArgumentParser(description='PPM Evaluation Script')
    parser.add_argument('--synthetic', type=str2bool, required=True, help="Whether the considered dataset is synthetic.")
    parser.add_argument('--dataset', type=str, required=True, help="The name of the dataset to run the model on.")
    parser.add_argument('--regular', type=str2bool, required=True, help="Whether the considered dataset is regularly sampled.")
    parser.add_argument('--setting_num', type=int, required=False, default=None, help="The setting number for synthetic data.")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0, the main process spawns the loaders.")
    
    args = parser.parse_args()
    synthetic = args.synthetic
    dataset_name = args.dataset
    regular = args.regular
    setting_num = args.setting_num
    num_workers_dataloader = args.num_workers_dataloader
    persistent_workers = num_workers_dataloader > 0
    
    # For debugging
    # synthetic = True
    # dataset_name = 'dataset_1'
    # regular = True
    # setting_num = 1
    # num_workers_dataloader = 4
    # persistent_workers = num_workers_dataloader > 0
    
    if synthetic:
        assert setting_num is not None, "Setting number has to be given for synthetic data."
    else:
        assert dataset_name in AVAILABLE_DATASETS, f"Requested non-synthetic dataset '{dataset_name}' but the only available datasets are {AVAILABLE_DATASETS}."
           
    log_results = create_logging_dir_path(synthetic, dataset_name, regular, setting_num)
    if not os.path.exists(log_results):
        os.makedirs(log_results)
    
    print("Checking if all dataset and config files exists...")
    path_dataset_torch_train = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='train',
                                                  is_torch=True, setting_num=setting_num)
    path_dataset_list_train  = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='train',
                                                  is_torch=False, setting_num=setting_num)
    path_dataset_list_val    = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='val',
                                                  is_torch=False, setting_num=setting_num)
    path_dataset_torch_test  = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='test',
                                                  is_torch=True, setting_num=setting_num)
    cfg_path = create_cfg_file_path(synthetic, dataset_name, 'ppm_markov_model', regular, setting_num)
    
    for p in [
        cfg_path, path_dataset_torch_train, path_dataset_list_train, path_dataset_list_val, path_dataset_torch_test,
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")
    
    init_fn = None
    print(f"Loading config {cfg_path} and setting up datasets for training...")
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
        
    model_type = config['model']['type']
    del config['model']['type']
    assert model_type == 'ppm_markov_model'
    
    if synthetic:
        dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
        dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
        assert config['dataset']['batching_strategy'] == 'samples', "Batching strategy should be 'samples'."
    else:
        if MAP_NAME_TO_KIND_DATASET[dataset_name] == 'samples':
            dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
            dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
            assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
            assert config['dataset']['batching_strategy'] == 'samples', "Batching strategy should be 'samples'." 
        
        elif MAP_NAME_TO_KIND_DATASET[dataset_name] == 'random_start':
            dataset_train_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_train))
            dataset_test_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_test))
            if num_workers_dataloader > 0:
                init_fn = worker_init_fn
            assert dataset_train_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
            assert config['dataset']['batching_strategy'] == 'random_start', "Batching strategy should be 'random_start'."
        
    
    # =================== Fit the PPM Markov Model ===================
    models = []
    best_model_list = []
    for model_order in range(0, 6):
        print(f"Fitting PPM Markov Model of order: {model_order}.")
        start_time = time.time()
        model = PPM_MarkovModel(k=model_order, **config['model'])
        dataset_list_train = torch.load(path_dataset_list_train)
        for sample in dataset_list_train['samples']:
            x: list[int] = sample['x']
            # sampling_times: list[float] = sample['sampling_times'] sampling times cannot be modeled in a Markov Model
            target_mask: list[bool] = sample['target_mask']
            model.train_on_sequence(x, target_mask)
        training_time = time.time() - start_time
        print(f"Done with fitting the model. Took {datetime.timedelta(seconds=training_time)}")
        models.append(model) 
        
        dataset_list_val = torch.load(path_dataset_list_val)
        val_count = 0
        val_acc = 0
        val_ce_loss = 0
        for sample in dataset_list_val['samples']:
            x: list[int] = sample['x']
            target_mask: list[bool] = sample['target_mask']
            for i in range(1, len(x)):
                if target_mask is not None and not target_mask[i]:
                    continue
                val_count += 1
                probs = model.predict_probabilities(x[max(0, i-model_order):i])
                val_ce_loss += -np.log(probs[x[i]])
                if probs.argmax() == x[i]:
                    val_acc += 1
        val_ce_loss = (val_ce_loss / val_count).item()
        val_acc = val_acc / val_count
        best_model_list.append((model_order, val_ce_loss, val_acc, training_time))    
       
    # =================== Log the Desired Metrics ================== 
    print(f"Done with training all models. Took {datetime.timedelta(seconds=time.time() - full_run_time)}") 
    print(f"Logging results to {log_results}...")
    
    model_name = model_type
    result_dict = {}
    
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
    for model, stats in zip(models, best_model_list):
        L.seed_everything(1445)
        model_order = model.k
        result_dict[f'{model_name}_{model_order}'] = {}
        result_dict[f'{model_name}_{model_order}']['time_training'] = stats[-1]
        
        # here we use Pytorch data loaders to be consistent with the evaluation of the random starting point datasets
        for loader, split in zip([train_loader, test_loader], ['train', 'test']):
            total_acc = 0
            total_count = 0
            total_ce_loss = 0
            
            for batch in loader:
                x, sampling_times, input_padding_mask, target_mask, _, _ = batch 
                mask = None
                if input_padding_mask is not None:
                    mask = ~input_padding_mask # [batch_size, max_sequence_length]
                if target_mask is not None:
                    if mask is None:
                        mask = target_mask
                    else:
                        mask = mask & target_mask # [batch_size, max_sequence_length]
                
                x = x.tolist()
                if mask is not None:
                    mask = mask.tolist() 
                # TODO: keep track of the choice of skipping the first time step
                for sample_id in range(len(x)):
                    x_it = x[sample_id]
                    for i in range(1,len(x[sample_id])):
                        if mask is not None and not mask[sample_id][i]:
                            continue
                        total_count += 1
                        probs = model.predict_probabilities(x_it[max(0, i-model_order):i])
                        total_ce_loss += -np.log(probs[x_it[i]])
                        if probs.argmax() == x_it[i]:
                            total_acc += 1
                
            result_dict[f'{model_name}_{model_order}'][f'{split}_data_loss'] = (total_ce_loss / total_count).item()
            result_dict[f'{model_name}_{model_order}'][f'{split}_data_acc'] = total_acc / total_count
    
    best_model_list.sort(key=lambda x: x[1])
    result_dict['model_list'] = [list(e) for e in best_model_list] 
    # ====================== Print Results to File ======================
    with open(f"{log_results}/{model_type}_aggregated_results.yaml", "w") as f:
        yaml.dump(result_dict, f, default_flow_style=False)
      
    print(f"Done with all results. Took {datetime.timedelta(seconds=time.time() - full_run_time)}")