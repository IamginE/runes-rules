import argparse
import catboost
import datetime
import lightning as L
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import os
import pickle
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
from metrics import activation_histogram_rules
from models.interpretable_prediction_baselines import (
    PPM_MarkovModel,
    catboost_extract_tree_info,
    catboost_get_constraint_tree_number,
    convert_list_dataset_to_tree_dataset,
    xgboost_extract_tree_info,
    xgboost_get_constraint_tree_number
)
from models.rules import Rules


MODELS = ['rules_processed_rules', 'ppm_markov_model', 'catboost', 'xgboost']
MODEL_CFGS = ['rules_processed_rules', 'ppm_markov_model', 'trees', 'trees']
RANDOM_SEED = 1445 # same random seed as for other evaluations
TEST_CONSTRAINT = [False]

_TREE_INFO = None
_MODEL_TYPE = None

def create_shared_treeinfo(tree_info):
    data = pickle.dumps(tree_info, protocol=pickle.HIGHEST_PROTOCOL)
    shm = shared_memory.SharedMemory(create=True, size=len(data))
    shm.buf[:len(data)] = data
    return shm.name, len(data)


def init_worker_trees(shm_name, size, model_type):
    global _TREE_INFO, _MODEL_TYPE
    from multiprocessing import shared_memory
    import pickle

    shm = shared_memory.SharedMemory(name=shm_name)
    _TREE_INFO = pickle.loads(bytes(shm.buf[:size]))
    _MODEL_TYPE = model_type
    shm.close()

def count_active_rules_trees(sample_leaf_ids):
    sample_active_rules = set()
    for tree_id, leaf_id in enumerate(sample_leaf_ids):
        leaf_info = _TREE_INFO[tree_id]["leaves"][leaf_id]
        if leaf_info["depth"] == 0:
            continue
        if _MODEL_TYPE == 'catboost' and leaf_info["leaf_weight"] == 0:
            continue
        sample_active_rules.add(leaf_info["conditions"])

    return len(sample_active_rules)

# ====================== Run ====================== #
# Measure the Interpretability/Complexity of Models on a given dataset 
if __name__ == '__main__':
    print(f"Cuda is available {torch.cuda.is_available()}") 
    print(f"CPU count: {multiprocessing.cpu_count()}")
    
    # ====================== Setup ====================== #
    full_run_time = time.time()
    
    parser = argparse.ArgumentParser(description='Complexity Evaluation Script')
    parser.add_argument('--synthetic', type=str2bool, required=True, help="Whether the considered dataset is synthetic.")
    parser.add_argument('--dataset', type=str, required=True, help="The name of the dataset to run the model on.")
    parser.add_argument('--regular', type=str2bool, required=True, help="Whether the considered dataset is regularly sampled.")
    parser.add_argument('--setting_num', type=int, required=False, default=None, help="The setting number for synthetic data.")
    parser.add_argument('--treat_irregular_as_regular', type=str2bool, required=False, default=False, help=f"Whether to treat a dataset with irregular sampling times as regularly sampled (ignoring timestamps).")
    parser.add_argument('--num_workers_dataloader', type=int, required=False, default=4, help="The number of workers to use in the data loaders. If 0, the main process spawns the loaders.")
    parser.add_argument('--mp_cpu_count', type=int, required=False, default=4, help="CPU count for multiprocessing parallelization. HINT: Check whether the corresponding section is commented out.")
    parser.add_argument('--evaluate_rule_activation_and_loss', type=str2bool, required=False, default=False, help=("Whether to compute statistics about the average rule activation for prediction and the loss on each dataset." 
                                                                                                                   "HINT: On large datasets (with many samples and and many categories), enabling this option can blow up runtime, especially for the unconstraint XGBoost model."))
    
    args = parser.parse_args()
    synthetic = args.synthetic
    dataset_name = args.dataset
    regular = args.regular
    treat_irregular_as_regular = args.treat_irregular_as_regular
    setting_num = args.setting_num
    num_workers_dataloader = args.num_workers_dataloader
    persistent_workers = num_workers_dataloader > 0
    mp_cpu_count = args.mp_cpu_count
    evaluate_rule_activation_and_loss = args.evaluate_rule_activation_and_loss
    
    # For debugging
    # synthetic = False
    # dataset_name = 'holidays_irregular'
    # regular = False
    # treat_irregular_as_regular = False
    # setting_num = 14
    # num_workers_dataloader = 4
    # persistent_workers = num_workers_dataloader > 0
    # mp_cp_count = 2
    # evaluate_rule_activation_and_loss = False
    
    postfix_treat_irregular_as_regular = '_regular' if treat_irregular_as_regular else ''
    if synthetic:
        assert setting_num is not None, "Setting number has to be given for synthetic data."
    else:
        assert dataset_name in AVAILABLE_DATASETS, f"Requested non-synthetic dataset '{dataset_name}' but the only available datasets are {AVAILABLE_DATASETS}."
          
           
    log_results = create_logging_dir_path(synthetic, dataset_name, regular, setting_num)
    result_dict = {}
    rules_num_rules = None
    
    print("Checking if all dataset and config files exists...")
    path_dataset_torch_train = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='train',
                                                  is_torch=True, setting_num=setting_num)
    path_dataset_list_train  = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='train',
                                                  is_torch=False, setting_num=setting_num)
    path_dataset_torch_test  = create_dataset_path(synthetic=synthetic, dataset_name=dataset_name, regular=regular, mode='test',
                                                  is_torch=True, setting_num=setting_num)
    
    dataset_files = [path_dataset_torch_train, path_dataset_list_train, path_dataset_torch_test]
    cfg_paths = [create_cfg_file_path(synthetic, dataset_name, m, regular, setting_num) for m in MODEL_CFGS]
    result_files = [
        f"{log_results}/{m}_aggregated_results.yaml" if m == 'ppm_markov_model' else
        f"{log_results}/{m}_aggregated_results{postfix_treat_irregular_as_regular}.yaml"
        for m in MODELS
    ]
    rule_file_rules = f'{log_results}/processed_ruleset_rules{postfix_treat_irregular_as_regular}.pkl'
    
    for p in dataset_files + cfg_paths + result_files + [rule_file_rules]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")
    
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
    # =================== Setup the Models ===================
    for model_type, cfg_path, result_file in zip(MODELS, cfg_paths, result_files):
        start_time = time.time()
        print(f"Evaluating model of type {model_type}.")
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        with open(result_file, 'r') as f:
            results_model = yaml.safe_load(f)
        
        if 'type' in config['model']:
            assert model_type == config['model']['type']
            del config['model']['type']
        
        model = None
        if model_type == 'ppm_markov_model':
            # markov model is so fast to fit that saving and loading it is unnecessary
            best_model_order = results_model['model_list'][0][0]
            model = PPM_MarkovModel(k=best_model_order, **config['model'])
            dataset_list_train = torch.load(path_dataset_list_train)
            for sample in dataset_list_train['samples']:
                x: list[int] = sample['x']
                # sampling_times: list[float] = sample['sampling_times'] sampling times cannot be modeled in a Markov Model
                target_mask: list[bool] = sample['target_mask']
                model.train_on_sequence(x, target_mask)
                
        elif model_type == 'catboost':
            save_model_dir = f"{log_results}/catboost"
            model = catboost.CatBoostClassifier()
            model.load_model(f"{save_model_dir}/best{postfix_treat_irregular_as_regular}.cbm")
            
        elif model_type == 'xgboost':
            save_model_dir = f"{log_results}/xgboost"
            model = xgboost.XGBClassifier()
            model.load_model(f"{save_model_dir}/best{postfix_treat_irregular_as_regular}.json")
            
        elif model_type == 'rules_processed_rules':
            path_to_best_model = results_model['ckpt_list'][0][0]
            model = Rules.load_from_checkpoint(path_to_best_model)
            model.symbolic = True
            rules_num_rules = model.num_rules
            
            
        # ====================== Evaluate the Models ====================== # 
        train_loader = None
        test_loader = None
        if evaluate_rule_activation_and_loss:
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
        
        result_dict[model_type] = {'unconstraint': {}, 'constraint': {}}
        if model_type == 'ppm_markov_model':
            # model: PPM_MarkovModel
            
            for is_constraint in TEST_CONSTRAINT:
                if is_constraint:
                    assert rules_num_rules is not None
                    model.prune_contexts(rules_num_rules)
                constraint = 'constraint' if is_constraint else 'unconstraint'
                count_dict = model.get_counts()
                result_dict[model_type][constraint]['num_rules'] = len(count_dict) - 1 # -1 for 0-order context
                context_sizes = [len(context) for context in count_dict if len(context) > 0]
                result_dict[model_type][constraint]['mean_rule_size'] = np.mean(context_sizes).item()
                result_dict[model_type][constraint]['std_rule_size'] = np.std(context_sizes, ddof=1).item()
                
                if evaluate_rule_activation_and_loss:
                    L.seed_everything(RANDOM_SEED)
                    for loader, split in zip([train_loader, test_loader], ['train', 'test']):
                        total_acc = 0
                        total_count = 0
                        total_ce_loss = 0
                        num_rules_active = []
                        
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
                                    probs = model.predict_probabilities(x_it[:i])
                                    total_ce_loss += -np.log(probs[x_it[i]])
                                    if probs.argmax() == x_it[i]:
                                        total_acc += 1
                                    num_rules_active.append(model.count_matched_orders(x_it[:i]))
                        
                        result_dict[model_type][constraint][f'{split}_data_loss'] = (total_ce_loss / total_count).item()
                        result_dict[model_type][constraint][f'{split}_data_acc'] = total_acc / total_count
                        result_dict[model_type][constraint][f'{split}_mean_rules_active'] = np.mean(num_rules_active).item()
                        result_dict[model_type][constraint][f'{split}_std_rules_active'] = np.std(num_rules_active, ddof=1).item()
            
        
        elif model_type == 'catboost' or model_type == 'xgboost':
            # model: catboost.CatBoostClassifier | xgboost.XGBClassifier
            max_relative_distance = config['model']['max_relative_distance']
            padding_id = results_model['metadata_dataset_conversion']['padding_id_cat']
            train_len = results_model['metadata_dataset_conversion']['training_length_features']
            num_classes = config['model']['sequence_dim']
            tree_info = None
            
            for is_constraint in TEST_CONSTRAINT:
                max_tree_idx = None
                if model_type == 'catboost':
                    if tree_info is None:
                        tree_info = catboost_extract_tree_info(
                            model,
                            f"{save_model_dir}/best{postfix_treat_irregular_as_regular}.json",
                            num_classes = num_classes
                        )
                    if is_constraint:
                        assert rules_num_rules is not None
                        max_tree_idx = catboost_get_constraint_tree_number(tree_info, num_rules = rules_num_rules)
                
                elif model_type == 'xgboost':
                    if tree_info is None:
                        tree_info = xgboost_extract_tree_info(model, num_classes = num_classes)
                    if is_constraint:
                        assert rules_num_rules is not None
                        max_tree_idx = xgboost_get_constraint_tree_number(tree_info, num_rules = rules_num_rules, num_classes = num_classes)
                
                constraint = 'constraint' if is_constraint else 'unconstraint'
                rule_set = set()
                rule_lengths = []
                
                for j, tree_id in enumerate(tree_info):
                    mulitplier = 1 if model_type == 'catboost' else num_classes
                    if max_tree_idx is not None and j >= max_tree_idx * mulitplier: # constraint model, use only the first few trees
                        break
                    if tree_info[tree_id]['num_leaves'] == 1:
                        continue # tree without split, not a rule
                    
                    for leaf_info in tree_info[tree_id]['leaves'].values():
                        if model_type == 'catboost' and leaf_info["leaf_weight"] == 0: 
                            continue # leaves with 0 entries do not contribute (0 weights for all classes), thus do not count toward rules
                        if leaf_info['conditions'] not in rule_set: 
                            # we only count unique conditions; leaves with the same conditions (that define the same region) could be aggregated into one rule
                            rule_lengths.append(leaf_info['depth'])
                            rule_set.add(leaf_info['conditions'])
                
                 # NOTE: we could compensate for the fact that XGBoost builds one tree per class per iteration with result_dict[model_type][constraint]['num_rules'] = len(rule_set) / num_classes, 
                 # but this would give it far more capacity
                result_dict[model_type][constraint]['num_rules'] = len(rule_set)
                result_dict[model_type][constraint]['mean_rule_size'] = np.mean(rule_lengths).item()
                result_dict[model_type][constraint]['std_rule_size'] = np.std(rule_lengths, ddof=1).item()
                
                if evaluate_rule_activation_and_loss:
                    L.seed_everything(RANDOM_SEED)
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
                        features, time_features, targets = convert_list_dataset_to_tree_dataset(dataset, regular, max_relative_distance, padding_id, 
                                                                                                treat_irregular_as_regular, ensure_length=train_len)
                    
                        data = [list(features[i]) + list(time_features[i]) for i in range(len(features))]
                        if is_constraint:
                            if model_type == 'catboost':
                                probs = model.predict_proba(data, ntree_end=max_tree_idx)
                                leaf_ids = model.calc_leaf_indexes(data, ntree_end=max_tree_idx)
                            elif model_type == 'xgboost':
                                probs = model.predict_proba(data, iteration_range = (0, max_tree_idx))
                                leaf_ids = model.apply(data, iteration_range = (0, max_tree_idx)).astype(int)
        
                        else:
                            probs = model.predict_proba(data)
                            if model_type == 'catboost':
                                leaf_ids = model.calc_leaf_indexes(data)
                            elif model_type == 'xgboost':
                                leaf_ids = model.apply(data).astype(int)
                                
                        # extract the rule information; leaf_ids is of shape [num_samples, num_trees (used for prediction)]
                        num_rules_active = []
                        for i in range(leaf_ids.shape[0]):
                            sample_leaf_ids = leaf_ids[i] # [num_trees]
                            sample_active_rules = set()
                            for tree_id, leaf_id in enumerate(sample_leaf_ids):
                                leaf_info = tree_info[tree_id]['leaves'][leaf_id]
                                if leaf_info['depth'] == 0:
                                    continue # no real rule (no condition)
                                if model_type == 'catboost' and leaf_info["leaf_weight"] == 0: 
                                    continue # leaves with 0 entries do not contribute (0 weights for all classes), thus do not count toward rules
                                
                                # we only count unique conditions; leaves with the same conditions (that define the same region) could be aggregated into one rule
                                sample_active_rules.add(leaf_info['conditions'])
                            num_rules_active.append(len(sample_active_rules))
                        
                        # NOTE: Uncomment to use multiprocessing pool; this may not necessarily speed up performance due to overhead
                        # shm_name, size = create_shared_treeinfo(tree_info)
                        # with multiprocessing.Pool(mp_cpu_count, initializer = init_worker_trees, initargs = (shm_name, size, model_type)) as pool:
                        #     num_rules_active = pool.map(count_active_rules_trees, leaf_ids)
                        # shm = shared_memory.SharedMemory(name=shm_name)
                        # shm.close()
                        # shm.unlink()
                            
                        targets = np.array(targets)  
                        result_dict[model_type][constraint][f'{split}_data_loss'] = -np.log(probs[np.arange(len(targets)), targets]).mean().item()
                        result_dict[model_type][constraint][f'{split}_data_acc'] = (probs.argmax(-1) == targets).mean().item()
                        result_dict[model_type][constraint][f'{split}_mean_rules_active'] = np.mean(num_rules_active).item()
                        result_dict[model_type][constraint][f'{split}_std_rules_active'] = np.std(num_rules_active, ddof=1).item()
                    
                
                
        elif model_type == 'rules_processed_rules':
            constraint = 'constraint'
            # model: Rules
            with open(rule_file_rules, 'rb') as f:
                rule_set_rules: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]] = pickle.load(f)
                rule_head_sizes = [len(rule[0]) for rule in rule_set_rules]
            assert len(rule_set_rules) == rules_num_rules == model.num_rules      
            result_dict[model_type][constraint]['num_rules'] = model.num_rules
            result_dict[model_type][constraint]['mean_rule_size'] = np.mean(rule_head_sizes).item()
            result_dict[model_type][constraint]['std_rule_size'] = np.std(rule_head_sizes, ddof=1).item()
            
            if evaluate_rule_activation_and_loss:
                L.seed_everything(RANDOM_SEED)
                for loader, split in zip([train_loader, test_loader], ['train', 'test']):
                    hist_rules_active = activation_histogram_rules(model, loader).numpy()
                    values = np.arange(hist_rules_active.shape[0])
                    mean = ((hist_rules_active * values).sum() / hist_rules_active.sum()).item()
                    var = (hist_rules_active * (values - mean)**2).sum() / (hist_rules_active.sum() - 1) # ddof - 1
                    result_dict[model_type][constraint][f'{split}_mean_rules_active'] = mean
                    result_dict[model_type][constraint][f'{split}_std_rules_active'] = np.sqrt(var).item()
                
                trainer = L.Trainer(logger=False, enable_progress_bar=False)
                L.seed_everything(RANDOM_SEED)
                for loader, split in zip([train_loader, test_loader], ['train', 'test']):
                    res = trainer.test(model, loader)[0]
                    result_dict[model_type][constraint][f'{split}_data_loss'] = res['test_loss_corrected']
                    result_dict[model_type][constraint][f'{split}_data_acc'] = res['test_acc_corrected']
                
        print(f"Evaluating model of type {model_type} took: {datetime.timedelta(seconds=time.time() - start_time)}")
                

    # ====================== Print Results to File ======================
    print(f"Logging results to {log_results}...")

    with open(f"{log_results}/interpretability_results{postfix_treat_irregular_as_regular}.yaml", "w") as f:
        yaml.dump(result_dict, f, default_flow_style=False)
    
    print(f"Done with all results. Took {datetime.timedelta(seconds=time.time() - full_run_time)}")