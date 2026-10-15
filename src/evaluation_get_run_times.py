import gc
import lightning as L
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from data_generation import (
    LOGGING_DIR
)
from data.synthetic_data import (
    generate_random_rule_set_uniform_tails,
    generate_synthetic_data_targets_regular
)
from data.utils import (
    convert_dataset_to_torch,
    custom_collate_fn,
    SequenceDataset
)
from models.utils import loss_next_token
from models.rope_transformer import ROPETransformer
from models.rules import Rules


RANDOM_SEED = 1445
BATCH_SIZE = 32
NUM_SAMPLES = 1024 # samples per epoch
NUM_PREDICATE_NEURONS = 300
WARMUP_EPOCHS = 1
BENCHMARK_EPOCHS = 1
WORKERS_DATALOADERS = 4

SEQUENCE_DIMS = [32, 128, 512, 2048]
SEQUENCE_LENGTHS = [32, 64, 128, 256]
LOG_RESULTS = f'{LOGGING_DIR}/runtime'

def summarize(l: list[float | int]) -> list[float]:
    return [np.mean(l).item(), np.std(l, ddof=1).item()] 

def benchmark_model_training(
    model: nn.Module,
    dataloader:  torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, list[float]]:
    model.to(device)
    model.train()

    forward_times = []
    loss_calc_times = []
    backward_times = []
    optimizer_step_times = []
    full_step_times = []

    # --- Warm-up ---
    for _ in range(WARMUP_EPOCHS):
        
        for batch in dataloader:
            x, sampling_times, input_padding_mask, target_mask, _, _ = batch
            x = x.to(device)
            if sampling_times is not None:
                sampling_times = sampling_times.to(device)
            if input_padding_mask is not None:
                input_padding_mask = input_padding_mask.to(device)
            if target_mask is not None:
                target_mask = target_mask.to(device)
                
            return_dict = model(x, sampling_times)
       
            loss, acc, num_elements_loss = loss_next_token(
                x = x, 
                prediction_logits = return_dict['prediction_logits'],
                input_padding_mask = input_padding_mask,
                target_mask = target_mask,
                full_window_mask = return_dict['full_window_mask'],
            )
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(loss.item())
        model.train_epoch += 1

    torch.cuda.synchronize() if device.type == "cuda" else None
    # --- Benchmark ---
    for _ in range(BENCHMARK_EPOCHS):    
        for batch in dataloader:
            x, sampling_times, input_padding_mask, target_mask, _, _ = batch
            x = x.to(device)
            if sampling_times is not None:
                sampling_times = sampling_times.to(device)
            if input_padding_mask is not None:
                input_padding_mask = input_padding_mask.to(device)
            if target_mask is not None:
                target_mask = target_mask.to(device)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            
            return_dict = model(x, sampling_times)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
       
            loss, acc, num_elements_loss = loss_next_token(
                x = x, 
                prediction_logits = return_dict['prediction_logits'],
                input_padding_mask = input_padding_mask,
                target_mask = target_mask,
                full_window_mask = return_dict['full_window_mask'],
            )
            torch.cuda.synchronize() if device.type == "cuda" else None
            t2 = time.perf_counter()
        
            loss.backward()
            torch.cuda.synchronize() if device.type == "cuda" else None
            t3 = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize() if device.type == "cuda" else None
            t4 = time.perf_counter()
            
            forward_times.append(t1 - t0)
            loss_calc_times.append(t2 - t1)
            backward_times.append(t3 - t2)
            optimizer_step_times.append(t4 - t3)
            full_step_times.append(t4 - t0)
            # print(loss.item())
        model.train_epoch += 1

    return {
        "forward": summarize(forward_times),
        "loss": summarize(loss_calc_times),
        "backward": summarize(backward_times),
        "optimizer": summarize(optimizer_step_times),
        "step": summarize(full_step_times),
    }
    
 
@torch.no_grad()    
def benchmark_model_inference(
    model: nn.Module,
    dataloader:  torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, list[float]]:
    model.to(device)
    model.eval()

    forward_times = []

    # --- Warm-up ---
    for _ in range(WARMUP_EPOCHS):
        
        for batch in dataloader:
            x, sampling_times, input_padding_mask, target_mask, _, _ = batch
            x = x.to(device)
            if sampling_times is not None:
                sampling_times = sampling_times.to(device)
            if input_padding_mask is not None:
                input_padding_mask = input_padding_mask.to(device)
            if target_mask is not None:
                target_mask = target_mask.to(device)
                
            return_dict = model(x, sampling_times)

    torch.cuda.synchronize() if device.type == "cuda" else None
    # --- Benchmark ---
    for _ in range(BENCHMARK_EPOCHS):
        
        for batch in dataloader:
            x, sampling_times, input_padding_mask, target_mask, _, _ = batch
            x = x.to(device)
            if sampling_times is not None:
                sampling_times = sampling_times.to(device)
            if input_padding_mask is not None:
                input_padding_mask = input_padding_mask.to(device)
            if target_mask is not None:
                target_mask = target_mask.to(device)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
            return_dict = model(x, sampling_times)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            
            forward_times.append(t1 - t0)


    return {
        "forward": summarize(forward_times),
    }
    
# ====================== Run ====================== #
# Get runtime results for saved models.
if __name__ == '__main__':
    regular_sampling = True
    results = {}
    if not os.path.exists(LOG_RESULTS):
        os.makedirs(LOG_RESULTS)
    
    # ====================== Setup ====================== #
    print(f"Cuda is available {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    for sequence_length in SEQUENCE_LENGTHS:
        results[sequence_length] = {}
        for sequence_dim in SEQUENCE_DIMS:
            results[sequence_length][sequence_dim] = {}

            print(f"\nTesting (sequence length, sequence dimension): ({sequence_length}, {sequence_dim})")
            print(f"Generating rules and the dataset for evaluation.")
            print(f"Seeding random with seed: {RANDOM_SEED}")
            random.seed(RANDOM_SEED)
            # ====================== Generate a synthetic dataset ======================
            rule_set = generate_random_rule_set_uniform_tails(
                num_rules = 20,
                sequence_dim = sequence_dim,
                min_size_head = 1,
                max_size_head = 8,
                min_size_tail = 1,
                max_size_tail = 1,
                min_delay = 1,
                max_delay = 64,
                min_span_int = 1,
                max_span_int = 10,
                regular_sampling = regular_sampling,
                random_seed = RANDOM_SEED,
            )
            
            data = generate_synthetic_data_targets_regular(
                num_samples = NUM_SAMPLES,
                min_sequence_length = sequence_length,
                max_sequence_length = sequence_length,
                sequence_dim = sequence_dim,
                rules = rule_set,
                percentage_rules = 0.3,
                frequency_rule_consequent = 0.4,
                rule_confidence = 0.85,
                uniform_random_placement = False, # for fast generation
                ensure_disjoint = False,
                random_seed = RANDOM_SEED,
                retries_fast_placement = 5,
            )

            data = convert_dataset_to_torch(
                data, 
                has_padding = True
            )
            dataset = SequenceDataset(data)

            # ====================== Benchmark Models ======================
            print(f"Benchmarking Models.")
            for kind_model in ['rules', 'transformer']:
                L.seed_everything(RANDOM_SEED, workers=True)
                dataloader = DataLoader(
                    dataset,
                    batch_size = BATCH_SIZE, 
                    collate_fn = custom_collate_fn,
                    shuffle = False, 
                    pin_memory = True,
                    num_workers = WORKERS_DATALOADERS,
                    persistent_workers = True,
                )
                model = None
                
                if kind_model == 'rules':
                    print("Benchmarking Rules.")
                    model = Rules(
                        sequence_dim = sequence_dim,
                        num_neurons = NUM_PREDICATE_NEURONS,
                        num_rules = 64,
                        max_sequence_length = sequence_length,
                        max_relative_distance = sequence_length,
                        initial_temperature = 1.0,
                        decay_conjunction_temp = 0.9,
                        decay_neuron_temp = 0.9,
                        regular_sampling = regular_sampling,
                        only_full_window = False,
                        optimization_dict = {
                                'lambda_reg_int': 0.0,
                                'lr_decay': 1.0,
                                'lr_patience': 0,
                                'num_lr_decreases': 0,
                                'optimizer':{
                                    'kind': 'adam',
                                    'params': {'lr': 0.03}
                                },
                                'plateau_delta': 0.0,
                                'steps_temp_annealing': 32,
                                'warmup_epochs': 60
                        }
                    )
                    optimizer = model.configure_optimizers()
                    
                elif kind_model == 'transformer':
                    print("Benchmarking Transformer.")
                    model = ROPETransformer(
                        sequence_dim = sequence_dim,
                        activation = 'gelu',
                        att_dim = 128,
                        dropout = 0.1,
                        embedding_dim = 128,
                        final_nn_dim = 128,
                        final_nn_num_layers = 2,
                        max_relative_distance = sequence_length,
                        max_sequence_length = sequence_length,
                        num_heads = 4,
                        only_full_window = False,
                        regular_sampling = regular_sampling,
                        transformer_ff_nn_dim = 256,
                        transformer_num_layers = 4,
                        optimization_dict = {
                            'optimizer':{
                                    'kind': 'adam',
                                    'params': {'lr': 0.0003}
                                },
                        }
                    )
                    optimizer = model.configure_optimizers()
                
                training_results = benchmark_model_training(model, dataloader, optimizer, device)
                inference_results = benchmark_model_inference(model, dataloader, device)
                results[sequence_length][sequence_dim][f'{kind_model}_training'] = training_results
                results[sequence_length][sequence_dim][f'{kind_model}_inference'] = inference_results 
                
                if kind_model == 'rules':
                    print("Benchmarking Rules with symbolic antecedents.")
                    model.symbolic = True
                    results[sequence_length][sequence_dim][f'{kind_model}_inference_symbolic'] = benchmark_model_inference(model, dataloader, device) 
                 
                del model, optimizer
                gc.collect()
                torch.cuda.empty_cache()    
                print("Done, saving results...")
                with open(f'{LOG_RESULTS}/wall_clock_times.yaml', 'w') as f:
                    yaml.dump(results, f, default_flow_style=False)