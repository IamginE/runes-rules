import argparse
import catboost
import datetime
import itertools
import lightning as L
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
import xgboost
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
    convert_list_dataset_to_tree_dataset,
    pad_same_length
)


RANDOM_SEEDS = [631 , 8567] #, 49415, 68230] # NOTE: Remove two seeds to bring down runtime. 
                                             # Unlike with other models, we do a full grid search with each seed.
AVAILABLE_MODELS = {'xgboost', 'catboost'}
PADDING_ID = 0
# ====================== Run ====================== #
# Train a selected tree-boosting model
if __name__ == '__main__':
    print(f"Cuda is available {torch.cuda.is_available()}") 
    
    # ====================== Setup ====================== #
    full_run_time = time.time()
    
    parser = argparse.ArgumentParser(description='Tree Boosting Evaluation Script')
    parser.add_argument('--synthetic', type=str2bool, required=True, help="Whether the considered dataset is synthetic.")
    parser.add_argument('--dataset', type=str, required=True, help="The name of the dataset to run the model on.")
    parser.add_argument('--regular', type=str2bool, required=True, help="Whether the considered dataset is regularly sampled.")
    parser.add_argument('--setting_num', type=int, required=False, default=None, help="The setting number for synthetic data.")
    parser.add_argument('--treat_irregular_as_regular', type=str2bool, required=False, default=False, help=f"Whether to treat a dataset with irregular sampling times as regularly sampled (ignoring timestamps).")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0, the main process spawns the loaders.")
    parser.add_argument('--model_type', type=str, required=True, help=f"The model type from {AVAILABLE_MODELS}.")
    
    args = parser.parse_args()
    synthetic = args.synthetic
    dataset_name = args.dataset
    regular = args.regular
    treat_irregular_as_regular = args.treat_irregular_as_regular
    setting_num = args.setting_num
    num_workers_dataloader = args.num_workers_dataloader
    model_type = args.model_type
    persistent_workers = num_workers_dataloader > 0
    
    # For debugging
    # synthetic = True
    # dataset_name = 'dataset_1'
    # regular = True
    # treat_irregular_as_regular = False
    # setting_num = 0
    # num_workers_dataloader = 4
    # persistent_workers = num_workers_dataloader > 0
    # model_type = 'catboost'
    
    postfix_treat_irregular_as_regular = '_regular' if treat_irregular_as_regular else ''
    assert model_type in AVAILABLE_MODELS, f"Request model_type '{model_type}' but the only available models are {AVAILABLE_MODELS}."
    if synthetic:
        assert setting_num is not None, "Setting number has to be given for synthetic data."
    else:
        assert dataset_name in AVAILABLE_DATASETS, f"Requested non-synthetic dataset '{dataset_name}' but the only available datasets are {AVAILABLE_DATASETS}."
           
    log_results = create_logging_dir_path(synthetic, dataset_name, regular, setting_num)
    if not os.path.exists(log_results):
        os.makedirs(log_results)
    if not synthetic and dataset_name == 'amazon_purchases': 
        # amazon purchases is way larger than the rest, reduce the number of random seeds to run on to decrease running time
        # usually the results do not change much based on random seeds
        RANDOM_SEEDS = [631]
    
    print("Checking if all dataset and config files exists...")
    path_dataset_torch_train = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='train',
                                                  is_torch=True, setting_num=setting_num)
    path_dataset_list_train  = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='train',
                                                  is_torch=False, setting_num=setting_num)
    path_dataset_list_val    = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='val',
                                                  is_torch=False, setting_num=setting_num)
    path_dataset_torch_test  = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='test',
                                                  is_torch=True, setting_num=setting_num)
    cfg_path =  create_cfg_file_path(synthetic, dataset_name, 'trees', regular, setting_num)
    
    for p in [
        cfg_path, path_dataset_torch_train, path_dataset_list_train, path_dataset_list_val, path_dataset_torch_test,
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")
    
    print(f"Loading config {cfg_path} and setting up datasets for training...")
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f) 
    if treat_irregular_as_regular:
        config['model']['regular_sampling'] = True # overwrite the config 
    assert config['model']['regular_sampling'] == regular
    max_relative_distance = config['model']['max_relative_distance']
    max_sequence_length = config['model']['max_sequence_length']
    sequence_dim = config['model']['sequence_dim']
    
    init_fn = None
    if synthetic:
        dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
        dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset."
    else:
        if MAP_NAME_TO_KIND_DATASET[dataset_name] == 'samples':
            dataset_train_torch = SequenceDataset(torch.load(path_dataset_torch_train))
            dataset_test_torch = SequenceDataset(torch.load(path_dataset_torch_test))
            assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
        
        elif MAP_NAME_TO_KIND_DATASET[dataset_name] == 'random_start':
            dataset_train_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_train))
            dataset_test_torch = SequenceDatasetRandomSampling(torch.load(path_dataset_torch_test))
            if num_workers_dataloader > 0:
                init_fn = worker_init_fn
            assert dataset_train_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."     
    
    if treat_irregular_as_regular:
        assert not dataset_train_torch.metadata['regular_sampling'], "Trying to use 'treat_irregular_as_regular' with a regular dataset." 
    # =================== Grid Search over Parameters ===================
    train_data_list = torch.load(path_dataset_list_train)
    val_data_list = torch.load(path_dataset_list_val)
    num_samples_random_left_padding = None
    # augment the data whith samples, where the full context is not present for datasets that consist of long sequences (random subsampling datasets/optimization for neural sequence models)
    # this is also what we test the models on
    if not synthetic:
        if dataset_name == 'holidays_irregular':
            num_samples_random_left_padding = 1024 * 10
        elif dataset_name == 'commits':
            num_samples_random_left_padding = 1024 * 32
        elif dataset_name == 'amazon_purchases':
            num_samples_random_left_padding = 1024 * 128
            
        
    train_features, train_time_features, train_targets = convert_list_dataset_to_tree_dataset(
        train_data_list['samples'],
        regular_sampling = regular,
        max_relative_distance = max_relative_distance,
        padding_id = PADDING_ID,
        treat_irregular_as_regular = treat_irregular_as_regular,
        ensure_max_length = max_sequence_length - 1, # ensures that we do not use more context than other models
        expected_num_classes = sequence_dim, # add 1 dummy entry for every class that is unseen in the training data to ensure that CatBoost and XGBoost can work
        num_samples_random_left_padding = num_samples_random_left_padding
    )
    val_features, val_time_features, val_targets = convert_list_dataset_to_tree_dataset(
        val_data_list['samples'],
        regular_sampling = regular,
        max_relative_distance = max_relative_distance,
        padding_id = PADDING_ID,
        treat_irregular_as_regular = treat_irregular_as_regular,
        ensure_max_length = max_sequence_length - 1, # ensures that we do not use more context than other models
        num_samples_random_left_padding = num_samples_random_left_padding // 2 if num_samples_random_left_padding is not None else None
    )
    
    train_features, val_features = pad_same_length(np.array(train_features), np.array(val_features), padding_value=PADDING_ID)
    train_time_features, val_time_features = pad_same_length(np.array(train_time_features), np.array(val_time_features), padding_value=max_relative_distance)
    train_targets = np.array(train_targets)
    val_targets = np.array(val_targets)
    
    train_len = train_features.shape[1]
    num_cat = train_features.shape[1]
    num_numeric = train_time_features.shape[1]

    model_result_list = []
    best_model = None
    best_score = None
    
    # use lists for hstack, so categorical features do not get converted to float
    train_X = [list(train_features[i]) + list(train_time_features[i]) for i in range(len(train_features))]
    val_X = [list(val_features[i]) + list(val_time_features[i]) for i in range(len(val_features))]

    # ====================== Train the Models ====================== #
    if model_type == 'catboost':
        # Perform grid search over parameters
        cat_features = list(range(num_cat))
        train_pool = catboost.Pool(train_X, train_targets, cat_features=cat_features)
        val_pool = catboost.Pool(val_X, val_targets, cat_features=cat_features)
        
        if not synthetic and dataset_name == 'amazon_purchases': # amazon purchases is way larger than the rest, reduce the search space
            catboost_grid = {
                "depth": [6],
                "l2_leaf_reg": [0, 3, 5],
                "learning_rate": [0.003, 0.03]
            }
        else:
            catboost_grid = {
                "depth": [4, 6],
                "l2_leaf_reg": [0, 1, 3, 5],
                "learning_rate": [0.03, 0.3]
            }
        param_combinations = [
            dict(zip(catboost_grid.keys(), values))
            for values in itertools.product(*catboost_grid.values())
        ]
        
        for i, random_seed in enumerate(RANDOM_SEEDS):
            print(f"Running grid search for {model_type} on seed number {i+1}/{len(RANDOM_SEEDS)}.")
            L.seed_everything(random_seed)
            
            for j, params in enumerate(param_combinations):
                print(f"Testing params: {params} ({j+1}/{len(param_combinations)})")
                if not synthetic and dataset_name == 'amazon_purchases':
                    early_stopping_rounds = 6
                    subsample = 0.1
                else:
                    early_stopping_rounds = 30
                    subsample = 0.6
                start_time = time.time()
                model = catboost.CatBoostClassifier(
                    cat_features = cat_features,
                    loss_function = "MultiClass",
                    eval_metric = "MultiClass",
                    custom_metric = "Accuracy",
                    iterations = 300,
                    early_stopping_rounds = early_stopping_rounds,
                    classes_count = sequence_dim,
                    bootstrap_type = "Bernoulli",
                    subsample = subsample,
                    random_seed = random_seed,   
                    task_type = "GPU" if torch.cuda.is_available() else "CPU",
                    train_dir = f"{config['logging']['save_dir']}/{model_type}",
                    verbose = False,
                    **params
                )
                model.fit(
                    train_pool,
                    eval_set = val_pool,
                    use_best_model = True,
                    verbose = False
                )
                
                training_time = time.time() - start_time
                score_dict = model.get_evals_result()
                best_iteration = model.get_best_iteration()
                val_loss = score_dict['validation']['MultiClass'][best_iteration]
                val_acc = score_dict['validation']['Accuracy'][best_iteration]
                model_dict = params.copy()
                model_dict['val_data_loss'] = val_loss
                model_dict['val_data_acc'] = val_acc
                model_dict['random_seed'] = random_seed
                model_dict['time_training'] = training_time
                model_result_list.append(model_dict)
                
                if best_score is None or val_loss < best_score:
                    best_model = model
                    best_score = val_loss
                
                print(f"Evaluating params: {params} ({j+1}/{len(param_combinations)}) took {datetime.timedelta(seconds=time.time() - start_time)}")
        
        save_model_dir = f"{log_results}/{model_type}"
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        best_model.save_model(f"{save_model_dir}/best{postfix_treat_irregular_as_regular}.cbm")
        best_model.save_model(f"{save_model_dir}/best{postfix_treat_irregular_as_regular}.json", format="json", pool=train_pool)
        
        
    elif model_type == 'xgboost':
        feature_types = ['c'] * num_cat + ['q'] * num_numeric

        # Grid search parameters
        if not synthetic and dataset_name == 'amazon_purchases': # amazon purchases is way larger than the rest, reduce the search space
            xgb_grid = {
                "max_depth": [6],
                "lambda": [0, 3, 5], # L2 regularization
                "eta": [0.003, 0.03], # learning rate
            }
        else:
            xgb_grid = {
                "max_depth": [4, 6],
                "lambda": [0, 1, 3, 5], # L2 regularization
                "eta": [0.03, 0.3], # learning rate
            }
        param_combinations = [
            dict(zip(xgb_grid.keys(), values))
            for values in itertools.product(*xgb_grid.values())
        ]

        for i, random_seed in enumerate(RANDOM_SEEDS):
            print(f"Running grid search for {model_type} on seed number {i+1}/{len(RANDOM_SEEDS)}.")
            L.seed_everything(random_seed)

            for j, params in enumerate(param_combinations):
                print(f"Testing params: {params} ({j+1}/{len(param_combinations)})")
                if not synthetic and dataset_name == 'amazon_purchases':
                    early_stopping_rounds = 6
                    subsample = 0.1
                else:
                    early_stopping_rounds = 30
                    subsample = 0.6
                start_time = time.time()
                
                # builds one tree per class per iteration (n_estimators)
                model = xgboost.XGBClassifier(
                    n_estimators = 300,
                    early_stopping_rounds = early_stopping_rounds,
                    device = 'cuda' if torch.cuda.is_available() else 'cpu',
                    subsample = subsample,
                    tree_method = 'hist',
                    objective = 'multi:softprob',
                    eval_metric = ['merror', 'mlogloss'], # last metric is used for early stopping
                    random_state = random_seed,
                    enable_categorical = True,
                    max_cat_to_onehot = 4,
                    max_cat_threshold = 50,
                    feature_types = feature_types,
                    num_class = sequence_dim,
                    verbosity = 0,
                    **params
                )
                
                model.fit(
                    train_X,
                    train_targets,
                    eval_set = [(val_X, val_targets)],
                    verbose = False
                )

                training_time = time.time() - start_time
                score_dict = model.evals_result()
                val_loss = score_dict['validation_0']['mlogloss'][model.best_iteration]
                val_acc = 1 - score_dict['validation_0']['merror'][model.best_iteration]
                model_dict = params.copy()
                model_dict['val_data_loss'] = val_loss
                model_dict['val_data_acc'] = val_acc
                model_dict['random_seed'] = random_seed
                model_dict['time_training'] = training_time
                model_result_list.append(model_dict)

                if best_score is None or val_loss < best_score:
                    best_model = model
                    best_score = val_loss
                
                print(f"Evaluating params: {params} ({j+1}/{len(param_combinations)}) took {datetime.timedelta(seconds=time.time() - start_time)}")

        save_model_dir = f"{log_results}/{model_type}"
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        best_model.save_model(f"{save_model_dir}/best{postfix_treat_irregular_as_regular}.json")

       
    # =================== Log the Desired Metrics ================== 
    print(f"Done with training all models. Took {datetime.timedelta(seconds=time.time() - full_run_time)}") 
    print(f"Logging results to {log_results}...")
    model_result_list.sort(key=lambda x: x['val_data_loss'])
    
    model_name = model_type
    result_dict = {
        model_name: {},
        "metadata_dataset_conversion": {
            'num_cat_features': num_cat,
            'num_time_features': num_numeric,
            'num_train_samples_random_left_padding': num_samples_random_left_padding,
            'padding_id_cat': PADDING_ID,
            'training_length_features': train_len,
        }
    }
    result_dict[model_name]['time_training'] = model_result_list[0]['time_training']
    result_dict['model_list'] = model_result_list
    
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
    L.seed_everything(1445)
    
    # here we use Pytorch data loaders to be consistent with the evaluation of the random starting point datasets
    for loader, split in zip([train_loader, test_loader], ['train', 'test']):
        total_acc = 0
        total_count = 0
        total_ce_loss = 0
        
        dataset = []
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
            
            for i in range(len(x)):
                dataset.append({
                    'x': x[i].tolist(), 
                    'sampling_times': None if sampling_times is None else sampling_times[i].tolist(),
                    'target_mask': None if mask is None else mask[i].tolist()
                })
        features, time_features, targets = convert_list_dataset_to_tree_dataset(dataset, regular, max_relative_distance, PADDING_ID, 
                                                                                treat_irregular_as_regular, ensure_length=train_len)
       
        data = [list(features[i]) + list(time_features[i]) for i in range(len(features))]
        probs = best_model.predict_proba(data)
        targets = np.array(targets)  
        result_dict[model_name][f'{split}_data_loss'] = -np.log(probs[np.arange(len(targets)), targets]).mean().item()
        result_dict[model_name][f'{split}_data_acc'] = (probs.argmax(-1) == targets).mean().item()
        
    # ====================== Print Results to File and Loss Curve Plotting ======================
    with open(f"{log_results}/{model_type}_aggregated_results{postfix_treat_irregular_as_regular}.yaml", "w") as f:
        yaml.dump(result_dict, f, default_flow_style=False)
    
    print(f"Done with all results. Took {datetime.timedelta(seconds=time.time() - full_run_time)}")