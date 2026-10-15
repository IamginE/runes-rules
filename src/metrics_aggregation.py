# FUNCTIONS AND CLASSES TO AGGREGATE RESULTS ACROSS 1 SETTING AND ACROSS DIFFERENT SETTINGS
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch
from torch.utils.data import DataLoader
from typing import Literal
import yaml

from data.real_data import (
    MAP_NAME_TO_KIND_DATASET,
    print_rules_to_file,
    print_rules_to_file_with_stats
)
from data.utils import (
    SequenceDataset,
    SequenceDatasetRandomSampling,
    custom_collate_fn,
    remove_unsupported_rules,
    simplify_antecedent_predicates,
    tighten_delay_intervals,
    trim_ruleset
)
from data_generation import (
    CFG_DIR,
    LOGGING_DIR,
    REAL_DATA_DIR
)
from eval_utils import worker_init_fn
from models.rope_transformer import ROPETransformer
from models.rules import Rules
from metrics import (
    activation_histogram_rules,
    calculate_agreement_prediction_models,
    calculate_agreement_rule_sets,
    calculate_support_confidence,
    structural_similarity_rulesets
)

class TrackerSingleDataset:
    
    def __init__(self):
        self.tracking_dict: dict[str, list[tuple[str, str, float, float, int]]] = {
            'rules': [],
            'rules_gt': [],
            'transformer': []
        }
    
    
    def track_model(
        self,
        model_type: Literal['rules', 'rules_gt', 'transformer'], 
        ckpt_path: str,
        log_dir: str,
        metric: float,
        training_time: float,
        run_id: int,
    ):
        """
        Track results for models across the same dataset for different runs.

        Args:
            model_type (Literal['rules', 'rules_gt', 'transformer']): The model type.
            ckpt_path (str): Path to the ckpt.
            log_dir (str): Path to the directory used for logging.
            metric (float): The metric used to find the best models for a model type.
            training_time (float): The training time in secondes.
            run_id (int): The id for the specific run (enumerated random seed).
        """
        if model_type in self.tracking_dict:
            self.tracking_dict[model_type].append((ckpt_path, log_dir, metric, training_time, run_id))
        else:
            raise ValueError(f"Unexpected model type: {model_type}.")
        
        
    def get_best_ckpt(self) -> dict[str, list[tuple[str, str, float, float, int]]]:
        """
        Returns:
            dict[str,list[tuple[str,float,float]]]: The dict with the entries for each model type, sorted by `metric` in ascending order.
        """
        for values in self.tracking_dict.values():
            values.sort(key=lambda x: x[2])
        return self.tracking_dict
        
        
def aggregate_results_single_dataset_synthetic(
    tracker: TrackerSingleDataset,
    log_dir: str,
    path_dataset_list_train: str,
    path_dataset_list_test: str,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    quantile_q: float = 0.95, 
    max_min_threshold: float = 0.9, 
    inclusion_threshold: float = 0.95,
    min_confidence_trim: float = 0.05,
    min_rel_support_trim: float = 0.0005,
    plotting: bool = True
):
    """
    For the three different model types on one synthetic dataset aggregate all the metrics and log them to `log_dir`.
    If `plotting == True`, we also plot training curves to `log_dir`.

    Args:
        tracker (TrackerSingleDataset): The tracker used to track the results for this dataset.
        log_dir (str): Directory to log to.
        path_dataset_list_train (str): Path to the list version of the training dataset.
        path_dataset_list_test (str): Path to the list version of the test dataset.
        train_data_loader (DataLoader): Dataloader for the training dataset.
        test_data_loader (DataLoader): Dataloader for the testing dataset.
        qunatile_q (float, optional): Argument for `Rules.get_rule_set()`.
        max_min_threshold (float, optional): Argument for `Rules.get_rule_set()`.
        inclusion_threshold (float, optional): Argument for `Rules.get_rule_set()`.
        min_confidence_trim (float, optional): Argument for `trim_ruleset()`. Threshold for the confidence when trimming the discovered ruleset.
        min_rel_support_trim (float, optional): Argument for `trim_ruleset()`. Threshold for the relative support (frequency) when trimming the discovered ruleset.
        plotting (bool, optional): Whether to plot learning curves. Defaults to True.
    """
    best_ckpt_dict = tracker.get_best_ckpt()
    best_ckpt_dict_yaml = {k: [list(e) for e in v] for k, v in best_ckpt_dict.items()}
    result_dict: dict[str, dict[str, int | float | list]] = {
        'rules': {}, 
        'rules_gt': {}, 
        'transformer': {},
        'rule_set_gt': {},
        'ckpt_dict': best_ckpt_dict_yaml,
    }
    
    # ====================== Load all models and set them up for evaluation ====================== #
    rules = Rules.load_from_checkpoint(best_ckpt_dict['rules'][0][0])
    rules.symbolic = True # use symbolic rule heads for evaluation
    # sparcify the Rules model and time it
    t = time.time()
    rules.sparcify(train_data_loader)
    result_dict['rules']['time_sparcify'] = time.time() - t
    
    rules_gt = Rules.load_from_checkpoint(best_ckpt_dict['rules_gt'][0][0])
    assert rules_gt.symbolic == True # model should always be symbolic if initialized from rules
    assert rules_gt.gt_rule_heads == True 
    transformer = ROPETransformer.load_from_checkpoint(best_ckpt_dict['transformer'][0][0])
    model_mapping: dict[str, L.LightningModule] = {
        'rules': rules, 
        'rules_gt': rules_gt, 
        'transformer': transformer,
    }
    
    assert rules.regular_sampling == rules_gt.regular_sampling == transformer.regular_sampling \
        == train_data_loader.dataset.metadata['regular_sampling'] == test_data_loader.dataset.metadata['regular_sampling']
    # ====================== Accuracy and loss for the prediction tasks ====================== #
    trainer = L.Trainer(logger=False, enable_progress_bar=False)
    for model_type, model in model_mapping.items():
        res_train = trainer.test(model, train_data_loader)[0]
        result_dict[model_type]['train_data_loss'] = res_train['test_loss_corrected']
        result_dict[model_type]['train_data_acc'] = res_train['test_acc_corrected']
        res_test = trainer.test(model, test_data_loader)[0]
        result_dict[model_type]['test_data_loss'] = res_test['test_loss_corrected']
        result_dict[model_type]['test_data_acc'] = res_test['test_acc_corrected']
        
        result_dict[model_type]['time_training'] = best_ckpt_dict[model_type][0][3]
        
    # trainer moves back the models to cpu after testing lol
    for model in [rules, rules_gt, transformer]:  
        if model.device == torch.device('cpu') and torch.cuda.is_available():
            model.to(torch.device('cuda'))
    
    # ====================== Agreement to Rules with GT rule heads (antecedents) ====================== #
    for model_type in ['rules', 'transformer']:
        model = model_mapping[model_type]
        agreement_gt_pred = calculate_agreement_prediction_models(
            model_gt = rules_gt,
            model = model,
            device_gt = rules_gt.device,
            device_model = model.device,
            data_loader = train_data_loader
        )
        result_dict[model_type]['agreement_gt_pred'] = agreement_gt_pred['agreement_prediction']
        result_dict[model_type]['js_div_gt'] = agreement_gt_pred['avg_js']
        result_dict[model_type]['kl_gt_model'] = agreement_gt_pred['avg_kl_p_q']
        result_dict[model_type]['kl_model_gt'] = agreement_gt_pred['avg_kl_q_p']
    
    
    # ====================== Metrics for discrete rule sets ====================== #
    train_data_list = torch.load(path_dataset_list_train)  
    rule_set_gt = train_data_list['metadata']['rules']
    t = time.time()
    rule_set_model, _ = rules.get_rule_set(quantile_q, max_min_threshold, inclusion_threshold, epsilon_neurons_irregular=0)
    result_dict['rules']['time_discretization'] = time.time() - t
    regular_sampling = train_data_list['metadata']['regular_sampling']
    assert rules.regular_sampling == regular_sampling
    
    rule_set_model_trimmed = None
    for rule_set, model_type in zip([rule_set_model, rule_set_gt], ['rules', 'rule_set_gt']):
        t = time.time()
        confidence_support = calculate_support_confidence(train_data_list['samples'], rule_set, regular_sampling=regular_sampling)
        
        if model_type == 'rules':
            result_dict[model_type]['avg_rel_support_no_trim'] = confidence_support['avg_rel_rule_support']
            result_dict[model_type]['avg_confidence_no_trim']  = confidence_support['avg_rule_confidence']
            result_dict[model_type]['avg_adjusted_confidence_no_trim'] = confidence_support['avg_rule_confidence_adjusted']
            result_dict[model_type]['median_rel_support_no_trim'] = confidence_support['median_rel_rule_support']
            result_dict[model_type]['median_confidence_no_trim']  = confidence_support['median_rule_confidence']
            result_dict[model_type]['median_adjusted_confidence_no_trim'] = confidence_support['median_rule_confidence_adjusted']
            result_dict[model_type]['frequency_rule_consequent_no_trim'] = confidence_support['frequency_rule_consequent']
            result_dict[model_type]['size_rule_set_no_trim'] = len(rule_set)
            
            # remove low-support and low-confidence rules
            t2 = time.time()
            rule_set_model_trimmed, trim_confidences, trim_rel_supports = trim_ruleset(
                rule_set_model,
                confidence_support['rule_confidence_individual'],
                confidence_support['rel_rule_support_individual'],
                min_confidence = min_confidence_trim,
                min_rel_support = min_rel_support_trim
            )
            result_dict[model_type]['time_trim'] = time.time() - t2
            result_dict[model_type]['avg_rel_support_trim'] = np.mean(trim_rel_supports).item() if len(trim_rel_supports) > 0 else 0
            result_dict[model_type]['avg_confidence_trim']  = np.mean(trim_confidences).item() if len(trim_confidences) else 0
            result_dict[model_type]['median_rel_support_trim'] = np.median(trim_rel_supports).item() if len(trim_rel_supports) else 0
            result_dict[model_type]['median_confidence_trim']  = np.median(trim_confidences).item() if len(trim_confidences) else 0
            
            # simplify antecedents and tighten intervals
            t2 = time.time()
            rule_set_model_trimmed, num_removed_predicates = simplify_antecedent_predicates(train_data_list['samples'], rule_set_model_trimmed, regular_sampling=regular_sampling)
            result_dict[model_type]['time_simplify_antecedents'] = time.time() - t2
            result_dict[model_type]['avg_removed_predicates_per_rule'] = np.mean(num_removed_predicates).item() if len(num_removed_predicates) > 0 else 0
            
            t2 = time.time()
            rule_set_model_trimmed, avg_interval_reductions = tighten_delay_intervals(train_data_list['samples'], rule_set_model_trimmed, regular_sampling=regular_sampling)
            result_dict[model_type]['time_tighten_intervals'] = time.time() - t2
            result_dict[model_type]['avg_interval_reductions_across_rules'] = np.mean(avg_interval_reductions).item() if len(avg_interval_reductions) > 0 else 0
            
            confidence_support = calculate_support_confidence(train_data_list['samples'], rule_set_model_trimmed, regular_sampling=regular_sampling)
          
        result_dict[model_type]['time_conf_support'] = time.time() - t
        result_dict[model_type]['avg_rel_support'] = confidence_support['avg_rel_rule_support']
        result_dict[model_type]['avg_confidence']  = confidence_support['avg_rule_confidence']
        result_dict[model_type]['avg_adjusted_confidence']  = confidence_support['avg_rule_confidence_adjusted']
        result_dict[model_type]['median_rel_support'] = confidence_support['median_rel_rule_support']
        result_dict[model_type]['median_confidence']  = confidence_support['median_rule_confidence']
        result_dict[model_type]['median_adjusted_confidence'] = confidence_support['median_rule_confidence_adjusted']
        result_dict[model_type]['num_rule_head_active'] = confidence_support['num_rule_head_active']
        result_dict[model_type]['rel_rule_support_individual'] = confidence_support['rel_rule_support_individual']
        result_dict[model_type]['rule_confidence_individual'] = confidence_support['rule_confidence_individual']
        result_dict[model_type]['adjusted_rule_confidence_individual'] = confidence_support['adjusted_rule_confidence_individual']
        result_dict[model_type]['frequency_rule_consequent'] = confidence_support['frequency_rule_consequent']
        
        r_set_ = rule_set_model_trimmed if model_type == 'rules' else rule_set
        result_dict[model_type]['size_heads_tails'] = [[len(r[0]), len(r[1])] for r in r_set_]
        result_dict[model_type]['avg_size_heads'] = np.mean([len(r[0]) for r in r_set_]).item() if len(r_set_) > 0 else 0
        result_dict[model_type]['avg_size_tails'] = np.mean([len(r[1]) for r in r_set_]).item() if len(r_set_) > 0 else 0
        result_dict[model_type]['median_size_heads'] = np.median([len(r[0]) for r in r_set_]).item() if len(r_set_) > 0 else 0
        result_dict[model_type]['median_size_tails'] = np.median([len(r[1]) for r in r_set_]).item() if len(r_set_) > 0 else 0
        result_dict[model_type]['size_rule_set'] = len(r_set_)
            
    
    agreement_rule_sets = calculate_agreement_rule_sets(
        samples = train_data_list['samples'],
        rule_set_gt = rule_set_gt,
        rule_set_model = rule_set_model_trimmed,
        regular_sampling = regular_sampling,
        rule_matching_max = 1,
    )
    for k, v in agreement_rule_sets.items():
        if k.endswith('by_ant_len'):
            for head_len, metric in v.items():
                result_dict['rules'][f'{k}_{head_len}'] = metric
    agreement_rule_sets = {k: v for k, v in agreement_rule_sets.items() if not k.endswith('by_ant_len')}
    result_dict['rules'].update(agreement_rule_sets)   
    
    sim_gt_model, avg_sim_by_head_len_gt_model = structural_similarity_rulesets(rule_set_gt, rule_set_model_trimmed, regular_sampling=regular_sampling)
    result_dict['rules']['sim_gt_model'] = sim_gt_model
    for l, v in avg_sim_by_head_len_gt_model.items():  
        result_dict['rules'][f'sim_gt_model_ant_len_{l}'] = v  
    
    sim_model_gt, avg_sim_by_head_len_model_gt = structural_similarity_rulesets(rule_set_model_trimmed, rule_set_gt, regular_sampling=regular_sampling)
    result_dict['rules']['sim_model_gt'] = sim_model_gt
    for l, v in avg_sim_by_head_len_model_gt.items():  
        result_dict['rules'][f'sim_model_gt_ant_len_{l}'] = v
    
    # ====================== Suppot and confidence for test data ====================== #
    test_data_list = torch.load(path_dataset_list_test) 
    regular_sampling = test_data_list['metadata']['regular_sampling']
    assert rules.regular_sampling == regular_sampling
    for rule_set, model_type in zip([rule_set_model_trimmed, rule_set_gt], ['rules', 'rule_set_gt']):
        confidence_support_test = calculate_support_confidence(test_data_list['samples'], rule_set, regular_sampling=regular_sampling)
        result_dict[model_type]['avg_rel_support_test'] = confidence_support_test['avg_rel_rule_support']
        result_dict[model_type]['avg_confidence_test']  = confidence_support_test['avg_rule_confidence']
        result_dict[model_type]['avg_adjusted_confidence_test']  = confidence_support_test['avg_rule_confidence_adjusted']
        result_dict[model_type]['median_rel_support_test'] = confidence_support_test['median_rel_rule_support']
        result_dict[model_type]['median_confidence_test']  = confidence_support_test['median_rule_confidence']
        result_dict[model_type]['median_adjusted_confidence_test'] = confidence_support_test['median_rule_confidence_adjusted']
        result_dict[model_type]['rel_rule_support_individual_test'] = confidence_support_test['rel_rule_support_individual']
        result_dict[model_type]['rule_confidence_individual_test'] = confidence_support_test['rule_confidence_individual']
        result_dict[model_type]['adjusted_rule_confidence_individual_test'] = confidence_support_test['adjusted_rule_confidence_individual']
        result_dict[model_type]['num_rule_head_active_test'] = confidence_support_test['num_rule_head_active']
        result_dict[model_type]['frequency_rule_consequent_test'] = confidence_support_test['frequency_rule_consequent']
    
    
    # ====================== Statistics from the Dataset ====================== #
    result_dict['dataset_stats'] = {
        k: v for k, v in train_data_list['metadata'].items() if k in
        {
            'num_samples', 'min_sequence_length', 'max_sequence_length', 'sequence_dim',
            't_all', 'percentage_rules_given', 'frequency_rule_consequent_given', 'rule_confidence_given',
            'targets_all', 'percentage_rules_actual', 'frequency_rule_consequent_actual', 'avg_rule_confidence_actual',
            'median_rule_confidence_actual'
        }
    }
    
    with open(f"{log_dir}/aggregated_results.yaml", "w") as f:
        yaml.dump(result_dict, f, default_flow_style=False)
        
    with open(f"{log_dir}/processed_ruleset_rules.pkl", "wb") as f:
        pickle.dump(rule_set, f)
        
    if plotting:
        metrics = {
            'train_loss_corrected': 'Train Loss',
            'train_acc_corrected': 'Train Accuracy',
            'val_loss_corrected': 'Validation Loss',
            'val_acc_corrected': 'Validation Accuracy'
        }
        linestyles = {
            'rules': '-',
            'rules_gt': '--',
            'transformer': '-.',
        }
        model_names = {
            'rules': 'Rules',
            'rules_gt': 'Rules (GT)',
            'transformer': 'Transformer',
        }
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        axs = axs.flatten()
       
        for model_type, result_list in best_ckpt_dict.items():
            for _, metric_dir, _, _, run_id in result_list:
                metric_file = os.path.join(metric_dir, 'metrics.csv')
                df = pd.read_csv(metric_file)
                df = df.groupby('epoch').last().reset_index()
                
                for idx, (metric_key, _) in enumerate(metrics.items()):
                    ax = axs[idx]
                    if metric_key in df.columns:
                        ax.plot(
                            df['epoch'],
                            df[metric_key],
                            label=f'{model_names[model_type]} - run {run_id}',
                            linestyle=linestyles[model_type],
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
        fig.savefig(f"{log_dir}/training_plots.png")
        plt.close()
  
      
def aggregate_results_setting_synthetic(dir_setting: str, file_results: str = 'aggregated_results.yaml', outfile: str = 'aggregated_results_all.yaml'):
    """
    Further aggregate the results from `aggregate_results_single_dataset_synthetic` for the same setting on different datasets.
    Expects that all results for the different datasets are in the same folder (e.g.: `{log_dir}/synthetic/{regularity_str}_setting_{setting_num}`).

    Args:
        dir_setting (str): The directory where the subfolders should contain the result files for each dataset in this setting.
        file_results (str, optional): The yaml file name with the aggregated results for each dataset in this setting. Defaults to 'aggregated_results.yaml'.
        outfile (str, optional): The name of the file with the aggregated metrics. Defaults to 'aggregated_results_all.yaml'.
    """
    print(f"============= Aggregating results for {dir_setting} =============")
    base_keys = ['rules', 'rules_gt', 'transformer', 'rule_set_gt', 'dataset_stats']
    aggregated_results = {b : {} for b in base_keys}
    gathered_results =  {b : {} for b in base_keys}
    
    subdirs = [f'{dir_setting}/{d}' for d in os.listdir(dir_setting) if os.path.isdir(f'{dir_setting}/{d}')]

    for d in subdirs:
        if not os.path.exists(f'{d}/{file_results}'):
            print(f"Could not find result file {file_results} in {d}. Skipping...")
            continue
        else:
            print(f"Adding {d}/{file_results}...")
        with open(f'{d}/{file_results}', 'r') as f:
            result_dict = yaml.safe_load(f)
        
        for base_key, subdict in result_dict.items():
            if base_key not in base_keys:
                continue
            for k, v in subdict.items():
                if v is None:
                    raise ValueError(f"{d}/{file_results} contains None entry for {k}, which prevents calculating an aggregated metric")
                if not (isinstance(v, float) or isinstance(v, int)):
                    # lists are fine-grained metrics per dataset, use only aggregated metrics
                    continue
                if np.isnan(v):
                    raise ValueError(f"{d}/{file_results} contains NaN entry for {k}, which prevents calculating an aggregated metric")
                if k in gathered_results[base_key]:
                    gathered_results[base_key][k].append(v)
                else:
                    gathered_results[base_key][k] = [v]
            
            if base_key == 'rules':
                assert all([key in subdict for key in ['time_sparcify', 'time_discretization', 'time_simplify_antecedents', 'time_tighten_intervals']])
                time_preprocess_and_discretize = subdict['time_sparcify'] + subdict['time_discretization']
                time_postprocess = subdict['time_simplify_antecedents'] + subdict['time_tighten_intervals']
                if 'time_preprocess_and_discretize' in gathered_results[base_key]:
                    gathered_results[base_key]['time_preprocess_and_discretize'].append(time_preprocess_and_discretize)
                else:
                    gathered_results[base_key]['time_preprocess_and_discretize'] = [time_preprocess_and_discretize]
                    
                if 'time_postprocess' in gathered_results[base_key]:
                    gathered_results[base_key]['time_postprocess'].append(time_postprocess)
                else:
                    gathered_results[base_key]['time_postprocess'] = [time_postprocess]
                
    for base_key, subdict in gathered_results.items():
        for k, v in subdict.items():
            metric_list = gathered_results[base_key][k]
            std = np.std(metric_list, ddof=1).item()
            aggregated_results[base_key][k] = [np.mean(metric_list).item(), np.median(metric_list).item(), 0.0 if np.isnan(std) else std]
            
    with open(f"{dir_setting}/{outfile}", "w") as f:
        yaml.dump(aggregated_results, f, default_flow_style=False)
        
        
def aggregate_results_setting_synthetic_random(dir_setting: str, file_results: str = 'aggregated_results.yaml'):
    """
    Further aggregate the results for the different random datasets.
    Expects that all results for the different datasets are in the same folder (e.g.: `{log_dir}/synthetic/{regularity_str}_random`).

    Args:
        dir_setting (str): The directory where the subfolders should contain the result files for each dataset in this setting.
        file_results (str, optional): The yaml file name with the aggregated results for each dataset in this setting. Defaults to 'aggregated_results.yaml'.
    """
    print(f"============= Aggregating results for {dir_setting} =============")
    aggregated_results = {}
    gathered_results =  {}
    
    subdirs = [f'{dir_setting}/{d}' for d in os.listdir(dir_setting) if os.path.isdir(f'{dir_setting}/{d}')]

    for d in subdirs:
        if not os.path.exists(f'{d}/{file_results}'):
            print(f"Could not find result file {file_results} in {d}. Skipping...")
            continue
        else:
            print(f"Adding {d}/{file_results}...")
        with open(f'{d}/{file_results}', 'r') as f:
            result_dict = yaml.safe_load(f)
        
        for k, v in result_dict.items():
            if v is None:
                raise ValueError(f"{d}/{file_results} contains None entry for {k}, which prevents calculating an aggregated metric")
            if not (isinstance(v, float) or isinstance(v, int)):
                # lists are fine-grained metrics per dataset, use only aggregated metrics
                continue
            if np.isnan(v):
                raise ValueError(f"{d}/{file_results} contains NaN entry for {k}, which prevents calculating an aggregated metric")
            if k in gathered_results:
                gathered_results[k].append(v)
            else:
                gathered_results[k] = [v]
                
    for k, v in gathered_results.items():
        metric_list = gathered_results[k]
        std = np.std(metric_list, ddof=1).item()
        aggregated_results[k] = [np.mean(metric_list).item(), np.median(metric_list).item(), 0.0 if np.isnan(std) else std]
            
    with open(f"{dir_setting}/aggregated_results_all.yaml", "w") as f:
        yaml.dump(aggregated_results, f, default_flow_style=False)
        
        
def aggregate_results_setting_synthetic_ppm_and_trees(dir_setting: str, file_postfix: str = '_aggregated_results.yaml', outfile: str = 'aggregated_results_all_ppm_trees.yaml'):
    """
    Further aggregate the results of PPM and boosting tree methods for the same setting on different datasets.
    Expects that all results for the different datasets are in the same folder (e.g.: `{log_dir}/synthetic/{regularity_str}_setting_{setting_num}`).
    Expects all result files to have the name `'{model_name}{shared_postfix}.yaml'`.

    Args:
        dir_setting (str): The directory where the subfolders should contain the result files for each dataset in this setting.
        file_postfix (str, optional): The postfix of the yaml file with the aggregated results for each dataset in this setting. Defaults to '_aggregated_results.yaml'.
        outfile (str, optional): The name of the file with the aggregated metrics for all baseline models. Defaults to 'aggregated_results_all_ppm_trees.yaml'.
    """
    MODELS = ['ppm_markov_model', 'catboost', 'xgboost']
    print(f"============= Aggregating results for PPM in {dir_setting} =============")
    base_keys = [f'ppm_markov_model_{order}' for order in range(0,6)] + ['catboost', 'xgboost']
    aggregated_results = {b : {} for b in base_keys}
    gathered_results =  {b : {} for b in base_keys}
    
    aggregated_results['ppm_markov_model_best_val'] = {}
    gathered_results['ppm_markov_model_best_val'] = {}
    
    subdirs = [f'{dir_setting}/{d}' for d in os.listdir(dir_setting) if os.path.isdir(f'{dir_setting}/{d}')]

    for d in subdirs:
        for model in MODELS:
            file_path = f'{d}/{model}{file_postfix}'
            if not os.path.exists(file_path):
                print(f"Could not find result file {model}{file_postfix} in {d}. Skipping...")
                continue
            else:
                print(f"Adding {file_path} ...")
            with open(file_path, 'r') as f:
                result_dict = yaml.safe_load(f)
        
            for base_key, subdict in result_dict.items():
                if base_key not in base_keys:
                    continue
                for k, v in subdict.items():
                    if v is None:
                        raise ValueError(f"{file_path} contains None entry for {k}, which prevents calculating an aggregated metric")
                    if not (isinstance(v, float) or isinstance(v, int)):
                        # lists are fine-grained metrics per dataset, use only aggregated metrics
                        continue
                    if np.isnan(v):
                        raise ValueError(f"{file_path} contains NaN entry for {k}, which prevents calculating an aggregated metric")
                    if k in gathered_results[base_key]:
                        gathered_results[base_key][k].append(v)
                    else:
                        gathered_results[base_key][k] = [v]
            
            if model == 'ppm_markov_model':
                best_order = result_dict['model_list'][0][0] # best order according to the validation loss
                for k in ['test_data_acc', 'test_data_loss', 'train_data_acc', 'train_data_loss']:
                    v = result_dict[f'ppm_markov_model_{best_order}'][k]
                    if k in gathered_results['ppm_markov_model_best_val']:
                        gathered_results['ppm_markov_model_best_val'][k].append(v)
                    else:
                        gathered_results['ppm_markov_model_best_val'][k] = [v]
                    
                    
    for base_key, subdict in gathered_results.items():
        if len(subdict) == 0:
            print(f'Found no results for {base_key}, skipping...')
        for k, v in subdict.items():
            metric_list = gathered_results[base_key][k]
            std = np.std(metric_list, ddof=1).item()
            aggregated_results[base_key][k] = [np.mean(metric_list).item(), np.median(metric_list).item(), 0.0 if np.isnan(std) else std]
            
    with open(f"{dir_setting}/{outfile}", "w") as f:
        yaml.dump(aggregated_results, f, default_flow_style=False)
           
        
def evaluation_across_settings_synthetic(
    dir: str, 
    default_setting: dict[str, int | float],
    regular_sampling: bool,
    vary_params: list[tuple[str, ...]],
    names: list[str],
    file_results: str = 'aggregated_results_all.yaml',
):
    """
    AFTER `aggregate_results_single_dataset_synthetic()` was called for each dataset per setting and `aggregate_results_setting_synthetic()` 
    was called for each setting, evaluate the results along different dimensions.
    The `default_setting` specifies the important parameters of the data generation as well as their default value.
    `vary_params` describe which parameters to vary, while keeping the rest fixed to the default.
    Expects that all results for the different settings are in the same folder (e.g.: `{log_dir}/synthetic`) and that no setting occurs more than once.
    The results of this call should be in the same folder as the results of  `aggregate_results_setting_synthetic()`.
    The results are saved in `{regularity_string}_{names[i]}_{metric_name}.png`.
    
    Args:
        dir (str): The directory where the subfolders should contain the result files for each setting.
        default_setting (dict[str, int | float]): A dictionary with keys that specify the important default params and their default values.
            There has to be exactly one setting that matches these params, and there should not be two settings that have the same params if projected onto the given ones.
        regular_sampling (bool): Whether we are considering the regularly setting.
        vary_params (list[tuple[str, ...]]): A list tuples that specify which parameters should be varied together to compare results.
            In other words, the setting will be the same as the specified default except for the parameters that are varied.
        names(list[str]): Names associated with the respective vary_params. These names are also used as prefixes for the resulting files
        file_results (str, optional): The yaml file name with the aggregated results for each setting. Defaults to 'aggregated_results_all.yaml'.
    """ 
    def tuple_x_names(params_to_vary: tuple[str, ...], setting_key: tuple):
        tmp_list = []
        for (param, v1), v2 in zip(default_setting.items(), setting_key):
            if param in params_to_vary:
                tmp_list.append(v2)
        if len(tmp_list) == 1:
            return tmp_list[0]
        return tuple(tmp_list)   
    
    assert len(vary_params) == len(names)
    regularity_string = 'regular' if regular_sampling else 'irregular'            
    print(f"============= Evaluating {dir} =============")
    setting_to_results: dict[tuple[int | float, ...], dict] = {}
    found_default_setting = False
    
    # ==================== Load all available Result Files ====================
    subdirs = [f'{dir}/{d}' for d in os.listdir(dir) if os.path.isdir(f'{dir}/{d}') and d.startswith(regularity_string) and not d.endswith('random')]
    for d in subdirs:
        if not os.path.exists(f'{d}/{file_results}'):
            print(f"Could not find result file {file_results} in {d}. Skipping...")
            continue
        else:
            print(f"Adding {d}/{file_results}...")
        with open(f'{d}/{file_results}', 'r') as f:
            aggregated_results_dict = yaml.safe_load(f)
        
        tmp_flag = True 
        tmp_list = [] 
        for k, v in default_setting.items():
            if k == 'size_rule_set':
                first_key = 'rule_set_gt'
            else:
                first_key = 'dataset_stats'
            value = aggregated_results_dict[first_key][k][0]
            # since these parameters are also aggregated via the mean across result file, we round to avoid floating point errors
            value = round(value, 2)
            aggregated_results_dict[first_key][k][0] = value
            tmp_list.append(value)
            if v != value:
                tmp_flag = False
        if tmp_flag: # found a result file matching the given default setting
            found_default_setting = True
        key_tuple = tuple(tmp_list) # tuple consisting of the parameter values from the default_setting identifies a setting
        if key_tuple in setting_to_results and aggregated_results_dict != setting_to_results[key_tuple]:
            raise KeyError(f"Setting {key_tuple} already exists. Results would be ambiguous.")
        setting_to_results[key_tuple] = aggregated_results_dict
        
        
    if not found_default_setting:
        raise FileNotFoundError("No file that matches the given default parameters was found.")
    
    # ==================== Grouped Bar Plots Function ====================
    def grouped_bar_plots(
        subplot_rows: int, subplot_columns: int, fig_size: tuple[int, int], ordered_keys_settings: list[tuple[int | float, ...]], 
        params_to_vary: tuple[str, ...], grouped_key_list: list[list[tuple[str, str]]], metric_names: list[str], label_names: list[str],
        name: str, post_fix_plot: str, suptitle: str, sharey = False,
    ):
        assert subplot_rows * subplot_columns == len(grouped_key_list) == len(metric_names)
        assert all([len(label_names) == len(l) for l in grouped_key_list])
        fig, axs = plt.subplots(subplot_rows, subplot_columns, figsize=fig_size, sharey=sharey)
        if subplot_rows * subplot_columns > 1:
            axs = axs.flatten()
        for i, (list_groups, metric_name) in enumerate(zip(grouped_key_list, metric_names)):
            if subplot_rows * subplot_columns > 1:
                ax = axs[i]
            else:
                ax = axs
            x = np.arange(len(ordered_keys_settings))  # the label locations
            width = 1.0/len(label_names) * 0.8 # the width of the bars
            multiplier = 0
            x_ticks = []
            for (k1, k2), label in zip(list_groups, label_names):
                flag_x_ticks = len(x_ticks) == 0
                y, errors = [], []
                for setting_key in ordered_keys_settings:
                    if flag_x_ticks: 
                        t = tuple_x_names(params_to_vary, setting_key)
                        x_ticks.append(str(t))
                    metric_dict = setting_to_results[setting_key]
                    if k1 in metric_dict and k2 in metric_dict[k1]:
                        y.append(metric_dict[k1][k2][0]) # first entry is the average
                        errors.append(metric_dict[k1][k2][2]) # third entry is the std
                    else: # not results for this specific setting found
                        y.append(0.0)
                        errors.append(0.0)
                offset = width * multiplier
                ax.bar(x + offset, y, width, label=label, yerr=errors, edgecolor='black', capsize=5)
                multiplier += 1 
            ax.set_title(metric_name)
            ax.set_xlabel(name)
            x_ticks_loc = x + ((len(label_names) - 1) / 2) * width
            ax.set_xticks(x_ticks_loc, x_ticks)
            ax.legend(loc='best', ncols=len(label_names))
        fig.suptitle(suptitle, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{dir}/{regularity_string}_{name.lower().replace(',','').replace('(', '').replace(')', '').replace(' ', '_')}_{post_fix_plot}')
        plt.close()   
    
    # ==================== Define Plotting Groups and Metrics ====================
    rule_set_method_names = ['rules', 'rule_set_gt']
    model_method_names = ['rules', 'rules_gt', 'transformer']
    
    f1_metrics = [
        [(name, metric)  for name in rule_set_method_names if name != 'rule_set_gt'] for metric in 
        ['f1', 'precision', 'recall', 'avg_iou_indiv_gt_model', 'avg_iou_indiv_model_gt', 'avg_iou_indiv_symmetric']
    ]
    
    conf_support_metrics = [
        [(name, metric) for metric in metric_names for name in rule_set_method_names] for metric_names in [
            ['avg_confidence', 'avg_confidence_test'],
            ['median_confidence', 'median_confidence_test'],
            ['avg_rel_support', 'avg_rel_support_test'],
            ['median_rel_support', 'median_rel_support_test'],
            ['frequency_rule_consequent', 'frequency_rule_consequent_test'],
        ]
    ]
    
    training_objective_metrics = [
        [(name, metric) for name in model_method_names] for metric in ['train_data_loss', 'test_data_loss', 'train_data_acc', 'test_data_acc']
    ]
    
    agreement_gt_pred_metrics = [
        [(name, metric) for name in model_method_names if name != 'rules_gt'] for metric in ['js_div_gt', 'agreement_gt_pred']
    ]
    
    num_rule_metrics = [
        [(name, metric) for name in rule_set_method_names for metric in ['size_rule_set', 'size_rule_set_no_trim'] if name != 'rule_set_gt' or metric == 'size_rule_set']
    ]

    runtime_metrics = [
        [('rules', 'time_training'), ('rules', 'time_sparcify'),  ('rules', 'time_discretization'), 
         ('rules', 'time_simplify_antecedents'), ('rules', 'time_tighten_intervals'), ('rules', 'time_conf_support')]
    ]
    
    # ==================== Aggregate settings according to vary_params and plot them ====================
    for params_to_vary, name in zip(vary_params, names):
        settings = {}
        ordered_keys_settings = []
        # Check if two settings are the same except for the parameters that are being varied
        for k in setting_to_results.keys():
            add_k = True
            for (param, v1), v2 in zip(default_setting.items(), k):
                if param not in params_to_vary:
                    if v1 != v2:
                        add_k = False
                        break
            if add_k:
                settings[k] = setting_to_results[k]
                ordered_keys_settings.append(k)
        ordered_keys_settings.sort()
        
        # ==================== Plot Everything ====================
        # ================= Plot F1 Metrics =================
        metric_names = ['F1-Score', 'Precision', 'Recall', 'Avg. IoU GT -> Model', 'Avg. IoU Model -> GT', 'Avg. IoU Symmetric']
        label_names = ['Rules']
        fig_size = (35, 15)
            
        grouped_bar_plots(
            subplot_rows = 2,
            subplot_columns = 3,
            fig_size = fig_size,
            sharey = False,
            ordered_keys_settings = ordered_keys_settings,
            params_to_vary = params_to_vary,
            grouped_key_list = f1_metrics,
            metric_names = metric_names,
            label_names = label_names,
            name = name,
            post_fix_plot = 'f1',
            suptitle = 'Explained Events shared with the Ground Truth Rule Set'
        )
        
        # ================= Plot Confidence and Support Metrics =================
        metric_names = ['Avg. Confidence', 'Median Confidence', 'Avg. Frequency', 'Median Frequency', 'Frequency Rule Set']
        label_names = ['Rules', 'True Rule Set', 'Rules Test', 'True Rule Set Test']
        fig_size = (15, 25)
        
        grouped_bar_plots(
            subplot_rows = 5,
            subplot_columns = 1,
            fig_size = fig_size,
            sharey = False,
            ordered_keys_settings = ordered_keys_settings,
            params_to_vary = params_to_vary,
            grouped_key_list = conf_support_metrics,
            metric_names = metric_names,
            label_names = label_names,
            name = name,
            post_fix_plot = 'support_conf',
            suptitle = 'Confidence and Support'
        )
        
        # ================= Training Objective Metrics =================
        metric_names = ['Train Loss', 'Test Loss', 'Train Accuracy', 'Test Accuracy']
        label_names = ['Rules', 'Rules(GT)', 'Transformer']
        fig_size = (25, 15)
        
        grouped_bar_plots(
            subplot_rows = 2,
            subplot_columns = 2,
            fig_size = fig_size,
            sharey = 'row',
            ordered_keys_settings = ordered_keys_settings,
            params_to_vary = params_to_vary,
            grouped_key_list = training_objective_metrics,
            metric_names = metric_names,
            label_names = label_names,
            name = name,
            post_fix_plot = 'training',
            suptitle = 'Training Metrics'
        )
        
        # ================= Agreement Prediction Metrics =================
        metric_names = ['JS Divergence', 'Agreement Prediction']
        label_names = ['Rules', 'Transformer']
        fig_size = (15, 15)
            
        grouped_bar_plots(
            subplot_rows = 2,
            subplot_columns = 1,
            fig_size = fig_size,
            sharey = False,
            ordered_keys_settings = ordered_keys_settings,
            params_to_vary = params_to_vary,
            grouped_key_list = agreement_gt_pred_metrics,
            metric_names = metric_names,
            label_names = label_names,
            name = name,
            post_fix_plot = 'agreement_pred',
            suptitle = 'Agreement on Softmax Predictions'
        )
        
        # ================= Num Rules =================
        metric_names = ['Rule Set Size']
        label_names = ['Rules w/ trim', 'Rules w/o trim', 'Ground Truth Rule Set']
        fig_size = (12, 12)
        
        grouped_bar_plots(
            subplot_rows = 1,
            subplot_columns = 1,
            fig_size = fig_size,
            sharey = False,
            ordered_keys_settings = ordered_keys_settings,
            params_to_vary = params_to_vary,
            grouped_key_list = num_rule_metrics,
            metric_names = metric_names,
            label_names = label_names,
            name = name,
            post_fix_plot = 'num_rules',
            suptitle = 'Number of Discovered Rules'
        )
        
        # ================= Run Time =================
        metric_names = ['Time']
        label_names = ['Training', 'Sparcify', 'Discretize', 'Simplify', 'Tighten', 'Support and Conf.']
        grouped_bar_plots(
            subplot_rows = 1,
            subplot_columns = 1,
            fig_size = (12, 12),
            sharey = False,
            ordered_keys_settings = ordered_keys_settings,
            params_to_vary = params_to_vary,
            grouped_key_list = runtime_metrics,
            metric_names = metric_names,
            label_names = label_names,
            name = name,
            post_fix_plot = 'time',
            suptitle = 'Running Time'
        )       
        

def aggregate_results_single_dataset_real(
    tracker: TrackerSingleDataset,
    log_dir: str,
    path_dataset_list_train: str,
    path_dataset_list_test: str,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    random_seed: int,
    quantile_q: float = 0.95, 
    max_min_threshold: float = 0.9, 
    inclusion_threshold: float = 0.95,
    treat_irregular_as_regular: bool = False,
    plotting: bool = True
):
    """
    For Rules on one real dataset aggregate all the metrics and log them to `log_dir`.
    If `plotting == True`, we also plot training curves to `log_dir`.

    Args:
        tracker (TrackerSingleDataset): The tracker used to track the results for this dataset.
        log_dir (str): Directory to log to.
        path_dataset_list_train (str): Path to the list version of the training dataset.
        path_dataset_list_test (str): Path to the list version of the test dataset.
        train_data_loader (DataLoader): Dataloader for the training dataset.
        test_data_loader (DataLoader): Dataloader for the test dataset.
        random_seed (int): Random seed to ensure equal comparison when loading data from random starting point datasets.
        qunatile_q (float, optional): Argument for `Rules.get_rule_set()`.
        max_min_threshold (float, optional): Argument for `Rules.get_rule_set()`.
        inclusion_threshold (float, optional): Argument for `Rules.get_rule_set()`.
        treat_irregular_as_regular (bool, optional). Flag for experiments that treat irregular sequences as regularly sampled. Defaults to False.
        plotting (bool, optional): Whether to plot learning curves. Defaults to True.
    """
    best_ckpt_dict = tracker.get_best_ckpt()
    best_ckpt_dict_yaml = {k: [list(e) for e in v] for k, v in best_ckpt_dict.items()}
    result_dict: dict[str, dict[str, int | float | list]] = {
        'rules': {},
        'ckpt_dict': best_ckpt_dict_yaml,
    }
    run_with_transformer = len(best_ckpt_dict['transformer']) > 0
    transformer = None
    if run_with_transformer:
        result_dict['transformer'] = {}
        
    postfix_treat_irregular_as_regular = '_regular' if treat_irregular_as_regular else ''
    
    # ====================== Load all models and set them up for evaluation ====================== #
    rules = Rules.load_from_checkpoint(best_ckpt_dict['rules'][0][0])
    rules.symbolic = True # use symbolic rule heads for evaluation
    # sparcify the Rules model and time it
    t = time.time()
    rules.sparcify(train_data_loader, threshold_min_usage=0.0, threshold_max_usage=1.0)
    result_dict['rules']['time_sparcify'] = time.time() - t
    if treat_irregular_as_regular:
        assert rules.regular_sampling == (not train_data_loader.dataset.metadata['regular_sampling'])
    else:
        assert rules.regular_sampling == train_data_loader.dataset.metadata['regular_sampling']
    model_mapping: dict[str, L.LightningModule] = {
        'rules': rules, 
    }
    
    if run_with_transformer:
        transformer = ROPETransformer.load_from_checkpoint(best_ckpt_dict['transformer'][0][0])
        if treat_irregular_as_regular:
            assert transformer.regular_sampling == (not train_data_loader.dataset.metadata['regular_sampling'])
        else:
            assert transformer.regular_sampling == train_data_loader.dataset.metadata['regular_sampling']
        model_mapping['transformer'] = transformer
    
    # ====================== Accuracy and loss for the prediction tasks ====================== #
    trainer = L.Trainer(logger=False, enable_progress_bar=False)
    for model_type, model in model_mapping.items():
        L.seed_everything(random_seed)
        res_train = trainer.test(model, train_data_loader)[0]
        result_dict[model_type]['train_data_loss'] = res_train['test_loss_corrected']
        result_dict[model_type]['train_data_acc'] = res_train['test_acc_corrected']
        result_dict[model_type]['time_training'] = best_ckpt_dict[model_type][0][3]
        res_test = trainer.test(model, test_data_loader)[0]
        result_dict[model_type]['test_data_loss'] = res_test['test_loss_corrected']
        result_dict[model_type]['test_data_acc'] = res_test['test_acc_corrected']
        
    # trainer moves back the models to cpu after testing
    if rules.device == torch.device('cpu') and torch.cuda.is_available():
        rules.to(torch.device('cuda')) 
    if run_with_transformer: 
        if transformer.device == torch.device('cpu') and torch.cuda.is_available():
            transformer.to(torch.device('cuda')) 
    
    # ====================== Metrics for discrete rule sets ====================== #
    train_data_list = torch.load(path_dataset_list_train)  
    t = time.time()
    rule_set, rule_set_neg = rules.get_rule_set(quantile_q, max_min_threshold, inclusion_threshold, epsilon_neurons_irregular=0)
    result_dict['rules']['time_discretization'] = time.time() - t
    if treat_irregular_as_regular:
        regular_sampling = True
    else:
        regular_sampling = train_data_list['metadata']['regular_sampling']
    assert rules.regular_sampling == regular_sampling
    
    t = time.time()
    confidence_support = calculate_support_confidence(train_data_list['samples'], rule_set, regular_sampling=regular_sampling)       
    result_dict['rules']['time_conf_support_before_ant_processing'] = time.time() - t
    result_dict['rules'].update({
        f'{k}_before_ant_processing': v for k, v in confidence_support.items() if k in 
        {'num_rule_holds_by_tail_elem', 'avg_rel_rule_support', 'median_rel_rule_support', 'num_rule_head_active', 
         'avg_rule_confidence', 'median_rule_confidence', 'avg_rule_confidence_adjusted', 'median_rule_confidence_adjusted'}
    })
    int_to_symbol = train_data_list['metadata']['int_to_symbol']  
    print_rules_to_file_with_stats(
        rule_set,
        confidence_support['targets_all'],
        confidence_support['num_rule_head_active'],
        confidence_support['num_rule_holds_by_tail_elem'],
        confidence_support['adjusted_rule_confidence_individual'],
        int_to_symbol, 
        f'{log_dir}/ruleset_before_ant_processing{postfix_treat_irregular_as_regular}.txt'
    )
    print_rules_to_file(rule_set_neg, int_to_symbol, f'{log_dir}/ruleset_neg_before_ant_processing{postfix_treat_irregular_as_regular}.txt')
    
    # remove rules that are unsupported, also from the negative rule set
    t = time.time()
    rule_set, removed_rules = remove_unsupported_rules(train_data_list['samples'], rule_set, regular_sampling=regular_sampling)
    result_dict['rules']['time_remove_rules'] = time.time() - t
    rule_set_neg, removed_rules_neg = remove_unsupported_rules(train_data_list['samples'], rule_set_neg, regular_sampling=regular_sampling)
    result_dict['rules']['removed_rules_unsupp'] = removed_rules
    result_dict['rules']['removed_rules_unsupp_neg'] = removed_rules_neg
    
    # simplify antecedents and tighten intervals
    t = time.time()
    rule_set, num_removed_predicates = simplify_antecedent_predicates(train_data_list['samples'], rule_set, regular_sampling=regular_sampling)
    result_dict['rules']['time_simplify_antecedents'] = time.time() - t
    result_dict['rules']['removed_predicates_per_rule'] = num_removed_predicates
    result_dict['rules']['avg_removed_predicates_per_rule'] = np.mean(num_removed_predicates).item() if len(num_removed_predicates) > 0 else 0
    rule_set_neg, num_removed_predicates_neg = simplify_antecedent_predicates(train_data_list['samples'], rule_set_neg, regular_sampling=regular_sampling)
    result_dict['rules']['removed_predicates_per_rule_neg'] = num_removed_predicates_neg
    result_dict['rules']['avg_removed_predicates_per_rule_neg'] = np.mean(num_removed_predicates_neg).item() if len(num_removed_predicates_neg) > 0 else 0
    
    t = time.time()
    rule_set, avg_interval_reductions = tighten_delay_intervals(train_data_list['samples'], rule_set, regular_sampling=regular_sampling)
    result_dict['rules']['time_tighten_intervals'] = time.time() - t
    result_dict['rules']['avg_interval_reductions_per_rule'] = avg_interval_reductions
    result_dict['rules']['avg_interval_reductions_across_rules'] = np.mean(avg_interval_reductions).item() if len(avg_interval_reductions) > 0 else 0
    rule_set_neg, avg_interval_reductions_neg = tighten_delay_intervals(train_data_list['samples'], rule_set_neg, regular_sampling=regular_sampling)
    result_dict['rules']['avg_interval_reductions_per_rule_neg'] = avg_interval_reductions_neg
    result_dict['rules']['avg_interval_reductions_across_rules_neg'] = np.mean(avg_interval_reductions_neg).item() if len(avg_interval_reductions_neg) > 0 else 0
    
    t = time.time() 
    confidence_support = calculate_support_confidence(train_data_list['samples'], rule_set, regular_sampling=regular_sampling)
    result_dict['rules']['time_conf_support'] = time.time() - t
    result_dict['rules'].update(confidence_support)
    
    # ====================== Suppot and confidence for test data ====================== #
    test_data_list = torch.load(path_dataset_list_test) 
    if treat_irregular_as_regular:
        regular_sampling = True
    else:
        regular_sampling = test_data_list['metadata']['regular_sampling']
    confidence_support_test = calculate_support_confidence(test_data_list['samples'], rule_set, regular_sampling=regular_sampling)
    for k, v in confidence_support_test.items():
        result_dict['rules'][f'{k}_test'] = v
    
    # ====================== Print results to files and plotting ======================
    with open(f"{log_dir}/aggregated_results{postfix_treat_irregular_as_regular}.yaml", "w") as f:
        yaml.dump(result_dict, f, default_flow_style=False)
     
    print_rules_to_file_with_stats(
        rule_set,
        confidence_support['targets_all'],
        confidence_support['num_rule_head_active'],
        confidence_support['num_rule_holds_by_tail_elem'],
        confidence_support['adjusted_rule_confidence_individual'],
        int_to_symbol, 
        f'{log_dir}/ruleset{postfix_treat_irregular_as_regular}.txt'
    )
    print_rules_to_file(rule_set_neg, int_to_symbol, f'{log_dir}/ruleset_neg{postfix_treat_irregular_as_regular}.txt')
    
    with open(f"{log_dir}/processed_ruleset_rules{postfix_treat_irregular_as_regular}.pkl", "wb") as f:
        pickle.dump(rule_set, f)
        
    if plotting:
        
        metrics = {
            'train_loss_corrected': 'Train Loss',
            'train_acc_corrected': 'Train Accuracy',
        }
        linestyles = {
            'rules': '-',
        }
        model_names = {
            'rules': 'Rules',
        }
        if run_with_transformer:
            metrics.update({
                'val_loss_corrected': 'Validation Loss',
                'val_acc_corrected': 'Validation Accuracy'
            })
            linestyles['transformer'] = '-.'
            model_names['transformer'] = 'Transformer'
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 2, figsize=(16, 12))
        
        for model_type, result_list in best_ckpt_dict.items():
            for _, metric_dir, _, _, run_id in result_list:
                metric_file = os.path.join(metric_dir, 'metrics.csv')
                df = pd.read_csv(metric_file)
                df = df.groupby('epoch').last().reset_index()
                
                for idx, (metric_key, _) in enumerate(metrics.items()):
                    ax = axs[idx]
                    if metric_key in df.columns:
                        ax.plot(
                            df['epoch'],
                            df[metric_key],
                            label=f'{model_names[model_type]} - run {run_id}',
                            linestyle=linestyles[model_type],
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
        fig.savefig(f"{log_dir}/training_plots{postfix_treat_irregular_as_regular}.png")
        plt.close()
        
        
def plot_histograms_rule_usage_real_datasets(dataset: str):

    train_path_torch = f'{REAL_DATA_DIR}/{dataset}/train_torch.pt'
    test_path_torch = f'{REAL_DATA_DIR}/{dataset}/test_torch.pt'
    tracker_path_rules = f'{LOGGING_DIR}/real/{dataset}/tracker.pkl'
    res_path_rules_processed_rules =  f'{LOGGING_DIR}/real/{dataset}/rules_processed_rules_aggregated_results.yaml'
    for p in [train_path_torch, test_path_torch, tracker_path_rules, res_path_rules_processed_rules]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}") 
    
    init_fn = None
    if MAP_NAME_TO_KIND_DATASET[dataset] == 'samples':
        dataset_train_torch = SequenceDataset(torch.load(train_path_torch))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
        dataset_test_torch = SequenceDataset(torch.load(test_path_torch))
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
         
    elif MAP_NAME_TO_KIND_DATASET[dataset] == 'random_start':
        dataset_train_torch = SequenceDatasetRandomSampling(torch.load(train_path_torch))
        init_fn = worker_init_fn
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
        dataset_test_torch = SequenceDatasetRandomSampling(torch.load(test_path_torch))
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
    
    seed_val = 1445
    L.seed_everything(seed_val)
    train_data_loader = DataLoader(
        dataset_train_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = 4,
        persistent_workers = False,
        worker_init_fn = init_fn
    )
    
    test_data_loader = DataLoader(
        dataset_test_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = 4,
        persistent_workers = False,
        worker_init_fn = init_fn
    )

    with open(tracker_path_rules, 'rb') as f:
        tracker: TrackerSingleDataset = pickle.load(f) 
    best_ckpt_dict = tracker.get_best_ckpt()
    rules = Rules.load_from_checkpoint(best_ckpt_dict['rules'][0][0])
    if torch.cuda.is_available():
        rules.to(torch.device('cuda'))
    rules.symbolic = True # use symbolic rule heads for evaluation
    # sparcify the Rules model
    assert rules.regular_sampling == train_data_loader.dataset.metadata['regular_sampling']
    rules.sparcify(train_data_loader, threshold_min_usage=0.0, threshold_max_usage=1.0)
    
    with open(res_path_rules_processed_rules, 'r') as f:
        res_dict_rules_processed_rules = yaml.safe_load(f)
    best_ckpt_list = res_dict_rules_processed_rules['ckpt_list']
    rules_processed_rules = Rules.load_from_checkpoint(best_ckpt_list[0][0])
    rules_processed_rules.symbolic = True
    if torch.cuda.is_available():
        rules_processed_rules.to(torch.device('cuda'))
    assert rules_processed_rules.regular_sampling == train_data_loader.dataset.metadata['regular_sampling']
    
    model_mapping = {
        'rules': rules,
        'rules_processed_rules': rules_processed_rules,
    }
    count_dict = {model_type: {} for model_type in model_mapping.keys()}
    
    for model_type, model in model_mapping.items():
        L.seed_everything(seed_val)
        count_dict[model_type]['train'] = activation_histogram_rules(model, train_data_loader)
        count_dict[model_type]['test'] = activation_histogram_rules(model, test_data_loader)
        
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharey='row', sharex='all')
    plot_idx = 0

    for i, split in enumerate(['train', 'test']):
        for j, model_type in enumerate(model_mapping.keys()):
            if model_type == 'rules':
                model_name = 'Rules w/o Proc.'
            elif model_type == 'rules_processed_rules':
                model_name = 'Rules w/ Proc.'
        
            ax = axs[i,j]
            plot_idx += 1

            # Convert tensor to numpy if needed
            counts = count_dict[model_type][split].numpy()
            max_idx = counts.nonzero()[0].max()
            counts = counts[:max_idx + 1]
            ax.bar(range(len(counts)), counts)
            ax.set_title(f"{model_name} — {split}")
            ax.set_xlabel("Number of Active Rules per Event")
            ax.set_ylabel("Count")
            ax.grid(True, linestyle='--', alpha=0.4)
            
    for ax in axs.flatten():
        ax.tick_params(labelbottom=True, labelleft=True)
    
    fig.tight_layout()
    fig.savefig(f'{LOGGING_DIR}/real/{dataset}/activations_hists.png')
    
    
    
def get_rule_processing_stats_real_datasets(dataset: str):
    """
    Get some statistics on how symbolic rule processing changes the rules.

    Args:
        dataset (str): The name of the dataset.
    """

    train_path_torch = f'{REAL_DATA_DIR}/{dataset}/train_torch.pt'
    test_path_torch = f'{REAL_DATA_DIR}/{dataset}/test_torch.pt'
    cfg_path_rules = f'{CFG_DIR}/real/rules/{dataset}.yaml'
    path_processed_rules = f'{LOGGING_DIR}/real/{dataset}/processed_ruleset_rules.pkl'
    tracker_path_rules = f'{LOGGING_DIR}/real/{dataset}/tracker.pkl'
    res_path_rules_processed_rules =  f'{LOGGING_DIR}/real/{dataset}/rules_processed_rules_aggregated_results.yaml'
    for p in [
        train_path_torch, test_path_torch, path_processed_rules, cfg_path_rules,
        tracker_path_rules, res_path_rules_processed_rules
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}") 
    
    init_fn = None
    if MAP_NAME_TO_KIND_DATASET[dataset] == 'samples':
        dataset_train_torch = SequenceDataset(torch.load(train_path_torch))
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
        dataset_test_torch = SequenceDataset(torch.load(test_path_torch))
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == False, "Batching strategy does not match the strategy in the metadata of the dataset." 
         
    elif MAP_NAME_TO_KIND_DATASET[dataset] == 'random_start':
        dataset_train_torch = SequenceDatasetRandomSampling(torch.load(train_path_torch))
        init_fn = worker_init_fn
        assert dataset_train_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
        dataset_test_torch = SequenceDatasetRandomSampling(torch.load(test_path_torch))
        assert dataset_test_torch.metadata['random_starting_point_sampling'] == True, "Batching strategy does not match the strategy in the metadata of the dataset."
    
    seed_val = 1445
    L.seed_everything(seed_val)
    train_data_loader = DataLoader(
        dataset_train_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = 4,
        persistent_workers = False,
        worker_init_fn = init_fn
    )
    
    test_data_loader = DataLoader(
        dataset_test_torch,
        batch_size = 64, 
        collate_fn = custom_collate_fn,
        shuffle = False, 
        pin_memory = True,
        num_workers = 4,
        persistent_workers = False,
        worker_init_fn = init_fn
    )

    with open(tracker_path_rules, 'rb') as f:
        tracker: TrackerSingleDataset = pickle.load(f) 
    best_ckpt_dict = tracker.get_best_ckpt()
    rules = Rules.load_from_checkpoint(best_ckpt_dict['rules'][0][0])
    if torch.cuda.is_available():
        rules.to(torch.device('cuda'))
    rules.symbolic = True # use symbolic rule heads for evaluation
    # sparcify the Rules model
    assert rules.regular_sampling == train_data_loader.dataset.metadata['regular_sampling']
    rules.sparcify(train_data_loader, threshold_min_usage=0.0, threshold_max_usage=1.0)
    with open(cfg_path_rules, 'r') as f:
        cfg = yaml.safe_load(f)
    quantile_q = cfg['discretization']['quantile_q']
    inclusion_threshold = cfg['discretization']['inclusion_threshold']
    max_min_threshold = 0.9
    original_rule_set, _ = rules.get_rule_set(quantile_q, max_min_threshold, inclusion_threshold, epsilon_neurons_irregular=0)
    with open(path_processed_rules, 'rb') as f:
        processed_rule_set: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]] = pickle.load(f)
    
    
    with open(res_path_rules_processed_rules, 'r') as f:
        res_dict_rules_processed_rules = yaml.safe_load(f)
    best_ckpt_list = res_dict_rules_processed_rules['ckpt_list']
    rules_processed_rules = Rules.load_from_checkpoint(best_ckpt_list[0][0])
    rules_processed_rules.symbolic = True
    if torch.cuda.is_available():
        rules_processed_rules.to(torch.device('cuda'))
    assert rules_processed_rules.regular_sampling == train_data_loader.dataset.metadata['regular_sampling']
    
    stats_dict = {
        'before_processing': {},
        'after_processing': {}
    }
    for key, model in zip(stats_dict.keys(), [rules, rules_processed_rules]):
        L.seed_everything(seed_val)
        stats_dict[key]['train_act_histogram'] = activation_histogram_rules(model, train_data_loader).tolist()
        stats_dict[key]['test_act_histogram'] = activation_histogram_rules(model, test_data_loader).tolist()
        
    for key, rule_set in zip(stats_dict.keys(), [original_rule_set, processed_rule_set]):
        delay_interval_widths = []
        ant_sizes = []
        for rule in rule_set:
            head = rule[0]
            ant_sizes.append(len(head))
            for element, lower_interval, upper_interval in head:
                delay_interval_widths.append(upper_interval - lower_interval)
            
        stats_dict[key]['ant_size_mean'] = np.mean(ant_sizes).item()
        stats_dict[key]['ant_size_std'] = np.std(ant_sizes, ddof=1).item()
        
        stats_dict[key]['interval_width_mean'] = np.mean(delay_interval_widths).item()
        stats_dict[key]['interval_width_std'] = np.std(delay_interval_widths, ddof=1).item()
        
    
    with open(f'{LOGGING_DIR}/real/{dataset}/rule_processing_stats.yaml', 'w') as f:
        yaml.dump(stats_dict, f, default_flow_style=False)
        

if __name__ == "__main__":
    
    
    for i in range(28):
        aggregate_results_setting_synthetic(f'{LOGGING_DIR}/synthetic/regular_setting_{i}')
    aggregate_results_setting_synthetic_random(f'{LOGGING_DIR}/synthetic/regular_random')
    
    for i in range(28):
        aggregate_results_setting_synthetic_ppm_and_trees(f'{LOGGING_DIR}/synthetic/regular_setting_{i}')
    
    # Plotting of Statistics, not used in the paper
    """
    evaluation_across_settings_synthetic(
        'logs/synthetic',
        default_setting = {
            'sequence_dim': 128,
            'size_rule_set': 20,
            'num_samples': 1024,
            'percentage_rules_given': 0.3,
            'frequency_rule_consequent_given': 0.4,
            'rule_confidence_given': 0.85
        },
        regular_sampling = True,
        vary_params = [('sequence_dim',), ('size_rule_set',), ('num_samples',), ('percentage_rules_given',), ('frequency_rule_consequent_given', 'rule_confidence_given')],
        names = ['Number of Events', 'Ground Truth Rule Set Size', 'Number of Sequences', 'Percentage Rules in Sequence', '(Frequency Rule Consequent, Rule Confidence)']
    )
    """
    
    # for dataset in ['holidays_irregular', 'chorales', 'commits', 'rolling_mill', 'amazon_purchases']:
    #     get_rule_processing_stats_real_datasets(dataset)

    