import csv
import datetime
import holidays
import math
import os
import pandas as pd
from pathlib import Path
import random
import torch
from typing import Any

from .utils import convert_dataset_to_torch

AVAILABLE_DATASETS = {'amazon_purchases', 'chorales', 'commits', 'holidays_irregular', 'rolling_mill'} # datasets used in the paper
KIND_1 = 'samples'
KIND_2 = 'random_start'


MAP_NAME_TO_KIND_DATASET = {
    'amazon_purchases': KIND_2,
    'chorales': KIND_1,
    'commits': KIND_2,
    'holidays_irregular': KIND_2,
    'rolling_mill': KIND_1
}

class SymbolMapping:
    """
    Class to map events (items) to integers and vice versa.
    Can be used on-the-fly to encode sequences by keeping track of seen events and dynamically assigning ids to new events.
    """
    
    def __init__(self):
        self.symbol_mapping: dict[Any, int] = dict()
        self.inverse_mapping: dict[int, Any] = dict()
        self.next_id = 0
        
    def get_num_symbols(self):
        return self.next_id
        
    
    def map_sequence(self, sequence: list[Any]) -> list[int]:
        """
        Map a list of symbols to its representation as integer list.
        
        Args:
            sequence (list[Any]): The sequence to be mapped.

        Returns:
            list[int]: The integer id list.
        """
        return list(map(self.map_symbol_to_int, sequence))
    

    def map_symbol_to_int(self, symbol: Any) -> int:
        """
        Given a symbol, dynamically map it to its id.

        Args:
            symbol (Any): The symbol.

        Returns:
            int: The symbol's integer representation.
        """
        if symbol not in self.symbol_mapping:
            self.symbol_mapping[symbol] = self.next_id
            self.inverse_mapping[self.next_id] = symbol
            self.next_id += 1
        return self.symbol_mapping[symbol]
    
    
    def map_int_to_symbol(self, id: int) -> Any:
        """
        Given an id, map it to its symbol. The entry should have been set by `translate_symbol_to_int` first.

        Args:
            id (int): The symbol id.

        Returns:
            Any: The symbol's representation.
        """
        if id not in self.inverse_mapping:
            raise KeyError("Id has not been associated with a symol yet.")
        return self.inverse_mapping[id]
    
    def __str__(self):
        string_repr = ""
        for symbol, id in sorted(self.symbol_mapping.items(),key=lambda x: x[1]):
            string_repr += f"{id:6d}: {symbol}\n"
        return string_repr
    

def print_rules_to_file_with_stats(
    rule_set: list[tuple[list[tuple[int, int, int]], list[tuple[float, int]]]],
    num_targets: int,
    num_rule_head_active: list[int],
    num_rule_holds_by_tail_elem: list[list[int]],
    adjusted_confidences: list[float],
    int_to_symbol: dict[int, Any],
    file_path: str
):
    """
    Given a rule set in its list representation, print it to the given file with additional statistics.

    Args:
        rule_set (list[tuple[list[tuple[int, int, int]], list[tuple[float, int]]]]): A list of rules, where the length indicates the number of rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`, where the target item is sampled proportionally to its weight.
        num_targets (int): The number of target items in the dataset.
        num_rule_head_active (list[int]): How often each rule head is active in order of the rules in the rule set.
        num_rule_holds_by_tail_elem (list[list[int]]): How often each rule head holds, broken down by tail items, in order of the rules in the rule set.
        adjusted_confidences (list[float]): The confidences adjusted for the rule tail (consequent) size.
        int_to_symbol (dict[int, Any]): The mapping that maps symbol/event ids to it's representation.
        file_path (str): The path to the file.
    """
    assert len(rule_set) == len(num_rule_head_active) == len(num_rule_holds_by_tail_elem) == len(adjusted_confidences)
    rules_string = ""
    for i in range(len(rule_set)):
        rule_head, rule_tail = rule_set[i]
        count_head = num_rule_head_active[i]
        count_rule_by_tail = num_rule_holds_by_tail_elem[i]
        adjusted_confidence = adjusted_confidences[i]
        count_rule = sum(count_rule_by_tail)
        confidence = count_rule / count_head if count_head > 0 else 0
        rel_support = count_rule / num_targets if num_targets > 0 else 0
        rule_str = f"Rule {i}) ant. active {count_head:6d}, rule holds {count_rule:6d}, conf. {confidence:.4f}, adj. conf. {adjusted_confidence:.4f}, rel. supp. {rel_support:.7f}\n"
        rule_str += "AND ("
        for item, int_low, int_high in rule_head:
            item_name = int_to_symbol[item]
            rule_str += f"{item_name} in [{int_low:4.2f}, {int_high:4.2f}], "
        rule_str = rule_str[:-2]
        rule_str += ") --> ("
        
        for j, (weight, item) in enumerate(rule_tail):
            item_name = int_to_symbol[item]
            if weight < 0:
                rule_str += "not "
            rule_str += f'{item_name} [{count_rule_by_tail[j]}], '
        rule_str = rule_str[:-2] + ')\n\n'
        rules_string += rule_str
    with open(file_path, 'w') as f:
        f.write(rules_string)
        
        
def print_rules_to_file(
    rule_set: list[tuple[list[tuple[int, int, int]], list[tuple[float, int]]]],
    int_to_symbol: dict[int, Any],
    file_path: str
):
    """
    Given a rule set in its list representation, print it to the given file.

    Args:
        rule_set (list[tuple[list[tuple[int, int, int]], list[tuple[float, int]]]]): A list of rules, where the length indicates the number of rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`, where the target item is sampled proportionally to its weight.
        int_to_symbol (dict[int, Any]): The mapping that maps symbol/event ids to it's representation.
        file_path (str): The path to the file.
    """
    rules_string = ""
    for i in range(len(rule_set)):
        rule_head, rule_tail = rule_set[i]
        rule_str = "AND ("
        for item, int_low, int_high in rule_head:
            item_name = int_to_symbol[item]
            rule_str += f"{item_name} in [{int_low:4.2f}, {int_high:4.2f}], "
        rule_str = rule_str[:-2]
        rule_str += ") --> ("
        
        for weight, item in rule_tail:
            item_name = int_to_symbol[item]
            if weight < 0:
                rule_str += "not "
            rule_str += f'{item_name}, '
        rule_str = rule_str[:-2] + ')\n'
        rules_string += rule_str
    with open(file_path, 'w') as f:
        f.write(rules_string)
        

    
def materialize_holidays_dataset(
    regular_sampling: bool = False, 
    save_dir: str = 'data/real',
    min_year: int = 2000,
    max_year: int = 2100,
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
):
    """
    Create a dataset with holidays represented as one long sequence and save it in `{save_dir}/holidays_[regular | irregular]/{split}.pt`.
    It includes all holidays from 01-01-{min_year} to 31-12-{max_year - 1} across train, val, and test sets.
    We save two versions, one in list form and one in tensor form with suffix `_torch.pt`.
    Stats
        - Split along time axis.
        - dim: 14 (irreg.), 15 (reg.)
        - t_all: 1200 (irreg.)

    Args:
        regular_sampling (bool, optional): Whether to use the the regular representations (include all days)
            or the irregular representation (only holidays with their time stamp). Defaults to False.
        save_dir (str, optional): Saving directory. Defaults to 'data/real'.
        min_year (int, optional): The starting year. Defaults to 2000.
        max_year (int, optional): The final year to consider. Defaults to 2100.
        split_ratios (tuple[float, float, float], optional): The (train, val, split) ratio.
            Defaults to (0.6, 0.2, 0.2).
    """
    assert len(split_ratios) == 3 and abs(sum(split_ratios) - 1.0) < 1e-6
    
    DATASET_NAME = 'holidays'
    holidays_data = holidays.DE(years=range(min_year, max_year), subdiv='SL', language='en_US')
    val_year = math.ceil(split_ratios[0] * (max_year - min_year)) + min_year
    test_year = math.ceil(split_ratios[1] * (max_year - min_year)) + val_year
    
    samples_per_epoch = 1024
    window_size = 365 if regular_sampling else 12 # One year unless regular sampling and leap year; or two holidays on the same day and irregular sampling; or an additional holiday as in 2017
    min_num_timesteps = 365 if regular_sampling else 12
    
    symbol_translation = SymbolMapping()
    seq_kinds = ['seq', 'target_mask', 'sampling_times']
    splits = ['train', 'val', 'test']
    seq_dict: dict[str,dict[str, list]] = {split: {seq_kind: None for seq_kind in seq_kinds} for split in splits}
    num_targets = {split: 0 for split in splits}
    for split in splits:
        seq_dict[split]['seq'] = []
        if regular_sampling:
            seq_dict[split]['target_mask'] = []
        else:
            seq_dict[split]['sampling_times'] = []

    if regular_sampling:
        current_date = datetime.date(min_year, 1, 1)
        while current_date.year < max_year:
            if current_date.year < val_year:
                split = 'train'
            elif current_date.year < test_year:
                split = 'val'
            else:
                split = 'test'
            
            if current_date in holidays_data:
                seq_dict[split]['seq'].append(symbol_translation.map_symbol_to_int(holidays_data[current_date]))
                seq_dict[split]['target_mask'].append(True)
                num_targets[split] += 1
            else:
                seq_dict[split]['seq'].append(symbol_translation.map_symbol_to_int("normal"))
                seq_dict[split]['target_mask'].append(False)
            current_date += datetime.timedelta(days=1)
    else:
        base_date = datetime.date(min_year, 1, 1)
        current_date = datetime.date(min_year, 1, 1)
        # we do it like this rather than iterating over the dictionary, to include two holidays on one day as one (unique) event, rather than two with the same time stamp 
        # as our models assume one event per time step
        # the holidays class will give a concatenated string '[holiday 1]; [holiday 2]'
        while current_date.year < max_year:
            if current_date.year < val_year:
                split = 'train'
            elif current_date.year < test_year:
                split = 'val'
            else:
                split = 'test'
                
            if current_date in holidays_data:
                seq_dict[split]['seq'].append(symbol_translation.map_symbol_to_int(holidays_data[current_date]))
                sampling_time = (current_date - base_date).days
                seq_dict[split]['sampling_times'].append(sampling_time)
                num_targets[split] += 1
            current_date += datetime.timedelta(days=1)

    for split in splits + ['train_val']:
        if split == 'train_val':
            sequence = seq_dict['train']['seq'] + seq_dict['val']['seq']
            targets_all = num_targets['train'] + num_targets['test']
            if regular_sampling:
                target_mask = seq_dict['train']['target_mask'] + seq_dict['val']['target_mask']
                assert len(target_mask) == len(sequence)
                assert seq_dict['train']['sampling_times'] is None and seq_dict['val']['sampling_times'] is None
                sampling_times = None
            else:
                sampling_times = seq_dict['train']['sampling_times'] + seq_dict['val']['sampling_times']
                assert len(sampling_times) == len(sequence)
                assert seq_dict['train']['target_mask'] is None and seq_dict['val']['target_mask'] is None
                target_mask = None
        
        else:
            sequence = seq_dict[split]['seq']
            targets_all = num_targets[split]
            if regular_sampling:
                target_mask = seq_dict[split]['target_mask']
                assert len(target_mask) == len(sequence)
                assert seq_dict[split]['sampling_times'] is None
                sampling_times = None
            else:
                sampling_times = seq_dict[split]['sampling_times']
                assert len(sampling_times) == len(sequence)
                assert seq_dict[split]['target_mask'] is None
                target_mask = None
        
        if len(sequence) == 0:
            continue
            
        metadata = {
            'synthetic': False,
            'regular_sampling': regular_sampling,
            'random_starting_point_sampling': True,
            'num_samples': 1,
            't_all': len(sequence),
            'targets_all': targets_all,
            'samples_per_epoch': samples_per_epoch,
            'window_size': window_size,
            'min_num_timesteps': min_num_timesteps,
            'sequence_dim': symbol_translation.get_num_symbols(),
            'symbol_to_int': symbol_translation.symbol_mapping,
            'int_to_symbol': symbol_translation.inverse_mapping,
        }
        
        data = {
            'metadata': metadata,
            'samples': [
                {
                    'x': sequence,
                    'sampling_times': sampling_times,
                    'target_mask': target_mask,
                    'T': len(sequence)
                }
            ],
        }
        regularity_string = 'regular' if regular_sampling else 'irregular'
        if not os.path.exists(f"{save_dir}/{DATASET_NAME}_{regularity_string}"):
            os.makedirs(f"{save_dir}/{DATASET_NAME}_{regularity_string}")
        save_loc = f"{save_dir}/{DATASET_NAME}_{regularity_string}/{split}.pt"
        torch.save(data, save_loc)
        print(f"Successfully saved list dataset at {save_loc}")

        data_torch = {
            'metadata': metadata,
            'samples': [
                {
                    'x': torch.tensor(sequence),
                    'sampling_times': torch.tensor(sampling_times) if sampling_times is not None else None,
                    'target_mask':  torch.tensor(target_mask) if target_mask is not None else None,
                }
            ],
        }
        save_loc = f"{save_dir}/{DATASET_NAME}_{regularity_string}/{split}_torch.pt"
        torch.save(data_torch, save_loc)
        print(f"Successfully saved tensor dataset at {save_loc}")
    
    
def midi_to_note_name(midi_note: int) -> str:
    """
    Convert a MIDI note to its musical name.
    """
    if not (0 <= midi_note <= 127):
        raise ValueError("MIDI note must be in the range 0-127")

    # Sharp and flat equivalents
    sharp_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    flat_names  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    index = midi_note % 12
    octave = (midi_note // 12) - 1

    # Only show dual name if it's a sharp/flat note (not natural)
    if sharp_names[index] != flat_names[index]:
        note = f"{sharp_names[index]}/{flat_names[index]}"
    else:
        note = sharp_names[index]

    return f"{note}{octave}"


def materialize_chorales_dataset(
    random_seed: int,
    save_dir: str = 'data/real',
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
):
    """
    Get the chorales dataset (https://archive.ics.uci.edu/dataset/25/bach+chorales) and save the data splits in `{save_dir}/chorales/`.
    We save two versions of each split: one in list form and one in tensor form with the suffix `_torch`.
    One sample is one chorale.
     Stats
        - Split by samples.
        - dim: (19, 143)
        - samples: 100
        - min len, max len: (26, 86)
        - t_all: 4693

    Args:
        random_seed (int): The random seed. Can be None if it should not be set.
        save_dir (str, optional): Saving directory. Defaults to 'data/real'.
        split_ratios (tuple[float, float, float], optional): The (train, val, split) ratio.
            Defaults to (0.6, 0.2, 0.2).
    """
    assert len(split_ratios) == 3 and abs(sum(split_ratios) - 1.0) < 1e-6
    if random_seed is not None:
        random.seed(random_seed)
    
    def lex_num(s: str) -> tuple[int, int]:
        i = 0
        white_spaces = 0
        # ignore leading white spaces
        for char in s:
            if char.isspace():
                white_spaces += 1
            else:
                break
        i = white_spaces
        if s[white_spaces:].startswith('-'):
            i += 1
        while i < len(s) and s[i].isdigit():
            i += 1
        return int(s[white_spaces:i]), i

    DATASET_NAME = 'chorales'
    file_location = Path(__file__).resolve().parent / "dataset_files" / "chorales.lisp"
    symbol_translation = SymbolMapping()
    splits = ['train', 'val', 'test', 'train_val']
    samples = []
    min_len = float('inf')
    max_len = 0
    
    with open(file_location, mode='r') as f:
        # each line is a chorale or empty
        for line in f:
            line = line.strip()
            if line == '':
                continue
            count_paren = 0
            idx = 0
            seq = []
            sampling_times = []
            # depth 1 marks a chorale and its id, after that the notes start
            while count_paren < 1:
                char = line[idx]
                if char == '(':
                    count_paren += 1
                    break
                elif char == ')':
                    raise ValueError("Invalid input: Unmatched closed parenthesis.")
                idx += 1
                if idx > len(line):
                    raise ValueError("Invalid input: Unmatched open parenthesis.")
                 
            note_dict = None  
            while count_paren >= 1:
                if idx > len(line):
                    raise ValueError("Invalid input: Unmatched open parenthesis.")
                idx += 1
                count_paren_old = count_paren
                char = line[idx]
                if char == '(':
                    count_paren += 1
                elif char == ')':
                    count_paren -= 1
                
                # depth 2 are notes
                if count_paren == 2 and count_paren_old == 1:
                    note_dict = {
                        'st': None,         # start-time, measured in 16th notes from chorale beginning (time 0)
                        'pitch': None,      # pitch, MIDI number (60 = C4, 61 = C#4, 72 = C5, etc.)
                        'dur': None,        # duration, measured in 16th notes
                        'keysig': None,     # key signature, number of sharps or flats, positive if key signature has sharps, negative if key signature has flats
                        'timesig': None,    # time signature, in 16th notes per bar
                        'fermata': None,    # fermata, true or false depending on whether event is under a fermata
                    }
                
                # depth 3 are the 6 features of a note
                elif count_paren == 3 and char != '(':
                    if note_dict is None:
                        raise ValueError("note_dict is None, should never happen here.")
                    for key in note_dict.keys():
                        if line[idx:].startswith(key):
                            if note_dict[key] is not None:
                                raise ValueError(f"Feature '{key}' obtained two entries for a note.")
                            feature, len_num = lex_num(line[idx + len(key):])
                            note_dict[key] = feature
                            idx = idx + len(key) + len_num - 1
                            break
                
                # should have parsed one full note
                elif count_paren == 1 and count_paren_old == 2:
                    if note_dict is None:
                        raise ValueError("note_dict is None, should never happen here.")
                    for key, value in note_dict.items():
                        if value is None:
                            raise ValueError(f"Feature '{key}' is missing for this note.")
                    pitch = midi_to_note_name(note_dict['pitch'])
                    # seq.append(symbol_translation.map_symbol_to_int(f'{pitch}_dur_{note_dict['dur']}_fermata_{note_dict['fermata']}'))
                    seq.append(symbol_translation.map_symbol_to_int(f'{pitch}'))
                    sampling_times.append(note_dict['st'])
                    note_dict = None
                
                # there should not be a 4th level    
                elif count_paren > 3:
                    raise ValueError("Invalid input: Unexpected parsing depth exceeding 3.")
            

            if len(seq) > 0:
                min_len = min(min_len, len(seq))
                max_len = max(max_len, len(seq))
                assert len(seq) == len(sampling_times)
                samples.append({
                    'x': seq,
                    'sampling_times': sampling_times,
                    'target_mask': None,
                    'T': len(seq),
                    'used_rules': None,
                })
    
    random.shuffle(samples)          
    n1 = math.ceil(split_ratios[0] * len(samples))
    n2 = math.ceil(split_ratios[1] * len(samples))
    train_samples = samples[:n1]
    val_samples = samples[n1:n1+n2]
    test_samples = samples[n1+n2:]
    train_val_samples = train_samples + val_samples
    
    for samples, split in zip([train_samples, val_samples, test_samples, train_val_samples], splits):
        if len(samples) == 0:
            continue
        t_all = sum([sample['T'] for sample in samples])
        data = {
            'metadata': {
                'synthetic': False,
                'regular_sampling': False,
                'random_starting_point_sampling': False,
                'num_samples': len(samples),
                'min_sequence_length': min_len,
                'max_sequence_length': max_len,
                'sequence_dim': symbol_translation.get_num_symbols(),
                't_all': t_all,
                'targets_all': t_all,
                'symbol_to_int': symbol_translation.symbol_mapping,
                'int_to_symbol': symbol_translation.inverse_mapping,
            },
            'samples': samples
        }
        if not os.path.exists(f"{save_dir}/{DATASET_NAME}"):
            os.makedirs(f"{save_dir}/{DATASET_NAME}")
            
        save_loc = f"{save_dir}/{DATASET_NAME}/{split}.pt"
        torch.save(data, save_loc)
        print(f"Successfully saved list dataset at {save_loc}")
        save_loc = f"{save_dir}/{DATASET_NAME}/{split}_torch.pt"
        convert_dataset_to_torch(data, save_path=save_loc, has_padding=True)
        print(f"Successfully saved tensor dataset at {save_loc}")
        
        
def materialize_commits_dataset(
    save_dir: str = 'data/real',
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
):
    """
    Get the git commits dataset (https://doi.org/10.1007/978-3-030-10928-8_32, https://doi.org/10.1609/aaai.v38i8.28676) and and save the data splits in `{save_dir}/commits/`.
    We save two versions of each split: one in list form and one in tensor form with the suffix `_torch`.
    It is a long sequence of commits to a github repo.
    Stats:
        - Split along time axis.
        - dim: 119
        - t_all: 28879

    Args:
        save_dir (str): Saving directory. Defaults to 'data/real'.
        split_ratios (tuple[float, float, float], optional): The (train, val, split) ratio.
            Defaults to (0.6, 0.2, 0.2).
    """
    assert len(split_ratios) == 3 and abs(sum(split_ratios) - 1.0) < 1e-6

    DATASET_NAME = 'commits'
    file_location = Path(__file__).resolve().parent / "dataset_files" / "samba-commits.txt"
    symbol_translation = SymbolMapping()
    splits = ['train', 'val', 'test', 'train_val']
    samples_per_epoch = 1024
    window_size = 64
    min_num_timesteps = window_size
    seq = []
    
    with open(file_location, mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        # rows are ordered
        for row in reader:
            event = row[1] # event is the commit e-mail
            seq.append(symbol_translation.map_symbol_to_int(event))
            
    
    n1 = math.ceil(split_ratios[0] * len(seq))
    n2 = math.ceil(split_ratios[1] * len(seq))
    train_seq = seq[:n1]
    val_seq = seq[n1:n1+n2]
    test_seq = seq[n1+n2:]
    train_val_seq = train_seq + val_seq
    
    for seq, split in zip([train_seq, val_seq, test_seq, train_val_seq], splits):
        if len(seq) == 0:
            continue
        
        metadata = {
            'synthetic': False,
            'regular_sampling': True,
            'random_starting_point_sampling': True,
            'num_samples': 1,
            't_all': len(seq),
            'targets_all': len(seq),
            'samples_per_epoch': samples_per_epoch,
            'window_size': window_size,
            'min_num_timesteps': min_num_timesteps,
            'sequence_dim': symbol_translation.get_num_symbols(),
            'symbol_to_int': symbol_translation.symbol_mapping,
            'int_to_symbol': symbol_translation.inverse_mapping,
        }
        
        data = {
            'metadata': metadata,
            'samples': [
                {
                    'x': seq,
                    'sampling_times': None,
                    'target_mask': None,
                    'T': len(seq)
                }
            ],
        }

        if not os.path.exists(f"{save_dir}/{DATASET_NAME}"):
            os.makedirs(f"{save_dir}/{DATASET_NAME}")
        save_loc = f"{save_dir}/{DATASET_NAME}/{split}.pt"
        torch.save(data, save_loc)
        print(f"Successfully saved list dataset at {save_loc}")
        
        data_torch = {
            'metadata': metadata,
            'samples': [
                {
                    'x': torch.tensor(seq),
                    'sampling_times': None,
                    'target_mask': None,
                }
            ],
        }
        save_loc = f"{save_dir}/{DATASET_NAME}/{split}_torch.pt"
        torch.save(data_torch, save_loc)
        print(f"Successfully saved tensor dataset at {save_loc}")
    
    
def materialize_rolling_mill_dataset(
    random_seed: int,
    save_dir: str = 'data/real',
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
):
    """
    Get the Rolling Mill dataset (https://doi.org/10.1137/1.9781611976700.28, https://doi.org/10.1609/aaai.v38i8.28676) and save the data splits in `{save_dir}/rolling_mill/`.
    We save two versions of each split: one in list form and one in tensor form with the suffix `_torch`.
    Stats:
        - Split by samples.
        - dim: 556
        - min len, max len: (24, 142)
        - samples: 1000
        - t_all: 53788

    Args:
        random_seed (int): The random seed. Can be None if it should not be set.
        save_dir (str): Saving directory. Defaults to 'data/real'.
        split_ratios (tuple[float, float, float], optional): The (train, val, split) ratio.
            Defaults to (0.6, 0.2, 0.2).
    """
    assert len(split_ratios) == 3 and abs(sum(split_ratios) - 1.0) < 1e-6
    if random_seed is not None:
        random.seed(random_seed)

    DATASET_NAME = 'rolling_mill'
    file_location = Path(__file__).resolve().parent / "dataset_files" / "rolling-mill.dat"
    splits = ['train', 'val', 'test', 'train_val']
    samples = []
    min_len = float('inf')
    max_len = 0
    symbs = set()

    with open(file_location, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('-1')
    
    for l in lines:
        l = list(filter(lambda x : x!= '', l.strip().split(' ')))
        if len(l) == 0: 
            continue
        seq = list(map(int,l))
        symbs = symbs | set(seq)
        min_len = min(min_len, len(seq))
        max_len = max(max_len, len(seq))
        samples.append({
            'x': seq,
            'sampling_times': None,
            'target_mask': None,
            'T': len(seq),
            'used_rules': None,
        })
        
    sequence_dim = len(symbs)
    
    random.shuffle(samples)
    n1 = math.ceil(split_ratios[0] * len(samples))
    n2 = math.ceil(split_ratios[1] * len(samples))
    train_samples = samples[:n1]
    val_samples = samples[n1:n1+n2]
    test_samples = samples[n1+n2:]
    train_val_samples = train_samples + val_samples
    
    for samples, split in zip([train_samples, val_samples, test_samples, train_val_samples], splits):
        if len(samples) == 0:
            continue
        t_all = sum([sample['T'] for sample in samples])
        data = {
            'metadata': {
                'synthetic': False,
                'regular_sampling': True,
                'random_starting_point_sampling': False,
                'num_samples': len(samples),
                'min_sequence_length': min_len,
                'max_sequence_length': max_len,
                'sequence_dim': sequence_dim,
                't_all': t_all,
                'targets_all': t_all,
                'symbol_to_int': {d: d for d in range(sequence_dim)},  # we don't have explicit symbols for this dataset
                'int_to_symbol': {d: d for d in  range(sequence_dim)}, # we don't have explicit symbols for this dataset
            },
            'samples': samples
        }
        if not os.path.exists(f"{save_dir}/{DATASET_NAME}"):
            os.makedirs(f"{save_dir}/{DATASET_NAME}")
            
        save_loc = f"{save_dir}/{DATASET_NAME}/{split}.pt"
        torch.save(data, save_loc)
        print(f"Successfully saved list dataset at {save_loc}")
        save_loc = f"{save_dir}/{DATASET_NAME}/{split}_torch.pt"
        convert_dataset_to_torch(data, save_path=save_loc, has_padding=True)
        print(f"Successfully saved tensor dataset at {save_loc}")
        
       
def materialize_amazon_dataset(
    random_seed: int,
    save_dir: str = 'data/real',
    split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
):
    """
    Get the Amazon purchases dataset (https://www.nature.com/articles/s41597-024-03329-6) and save it in `{save_dir}/amazon_purchases.pt`.
    We save two versions, one in list form and one in tensor form with suffix `_torch.pt`.
    A sequence is the purchase history of a customer. 
    Stats:
        - Split by samples.
        - dim: 993
        - min len, max len: (10, 5057)
        - samples: 4909
        - t_all: 1,760,607

    Args:
        random_seed (int): The random seed. Can be None if it should not be set.
        save_dir (str): Saving directory. Defaults to 'data/real'.
        split_ratios (tuple[float, float, float], optional): The (train, val, split) ratio. 
            Defaults to (0.6, 0.2, 0.2).
    """
    assert len(split_ratios) == 3 and abs(sum(split_ratios) - 1.0) < 1e-6
    if random_seed is not None:
        random.seed(random_seed)
        
    DATASET_NAME = 'amazon_purchases'
    file_location = Path(__file__).resolve().parent / "dataset_files" / "amazon-purchases.csv"
    symbol_translation = SymbolMapping()
    samples = []
    splits = ['train', 'val', 'test', 'train_val']
    min_len = float('inf')
    max_len = 0
    
    MIN_CATEGORY_COUNT = 200
    MIN_SEQ_LEN = 10
    NORMALIZATION_TIME_STEPS = 5
    SAMPLES_PER_EPOCH = 2048
    WINDOW_SIZE = 64
    
    dataset_df = pd.read_csv(file_location)
    # Convert Order Date to datetime for proper sorting
    dataset_df['Order Date'] = pd.to_datetime(dataset_df['Order Date'])
    dataset_df = dataset_df.sort_values(by=['Survey ResponseID', 'Order Date'])
    
    # Drop purchases with unknown category as the category are our events
    dataset_df = dataset_df.dropna(subset=['Category'])
    
    # Group by Survey ResponseID (customers) and create lists of rows (as dicts)
    grouped_sequences = {
        response_id: group.to_dict(orient='records')
        for response_id, group in dataset_df.groupby('Survey ResponseID')
    }
    
    # Clean the dataset
    category_count = {}
    for purchase_seq in grouped_sequences.values():
        for entry in purchase_seq:
            category = entry['Category']
            if category not in category_count:
                category_count[category] = 1
            else:
                category_count[category] += 1
                
    samples = []
    max_shared_events = 0
    min_diff_time = float('inf')
    for purchase_seq in grouped_sequences.values():
        if len(purchase_seq) < MIN_SEQ_LEN:
            continue
        event_time = []
        shared_events = {}
        start = purchase_seq[0]['Order Date']
        for i in range(len(purchase_seq)):
            category = purchase_seq[i]['Category']
            if category_count[category] < MIN_CATEGORY_COUNT:
                category = 'OTHER'
            time = (purchase_seq[i]['Order Date'] - start).days
            event_time.append((time, category))
            
            if time not in shared_events:
                shared_events[time] = 0
            shared_events[time] += 1
            
            if i > 0:
                diff_time = (purchase_seq[i]['Order Date'] - purchase_seq[i-1]['Order Date']).days
                if diff_time > 0:
                    min_diff_time = min(min_diff_time, diff_time)
        max_shared_events = max(max_shared_events, max(shared_events.values()))
        event_time.sort() # sort primarily by time and then by category
        samples.append({
            'x': [x for _, x in event_time], 
            'sampling_times': [t for t, _ in event_time],
            'target_mask': None,
            'T': len(event_time)
        })
        max_len = max(max_len, len(event_time))
        min_len = min(min_len, len(event_time))
     
    # adjust the sampling times
    epsilon = epsilon = (min_diff_time / max_shared_events) * 1/2 # small epsilon to order events with the same timestamp  
    for i in range(len(samples)):
        sample_dict = samples[i]
        sampling_times = sample_dict['sampling_times']
        
        last_sampling_time = None
        counter = 0
        for j in range(len(sampling_times)):
            if last_sampling_time == sampling_times[j]:
                counter += 1
                sampling_times[j] = sampling_times[j] + counter * epsilon 
            else:
                last_sampling_time = sampling_times[j]
                counter = 0
                
        # normalize sampling times
        for j in range(len(sampling_times)):
            sampling_times[j] /= NORMALIZATION_TIME_STEPS
        
        sample_dict['sampling_times'] = sampling_times
        sample_dict['x'] = symbol_translation.map_sequence(sample_dict['x'])
        samples[i] = sample_dict

    random.shuffle(samples)
    n1 = math.ceil(split_ratios[0] * len(samples))
    n2 = math.ceil(split_ratios[1] * len(samples))
    train_samples = samples[:n1]
    val_samples = samples[n1:n1+n2]
    test_samples = samples[n1+n2:]
    train_val_samples = train_samples + val_samples
    
    for samples, split in zip([train_samples, val_samples, test_samples, train_val_samples], splits):
        if len(samples) == 0:
            continue
        t_all = sum([sample['T'] for sample in samples])
        metadata = {
            'synthetic': False,
            'regular_sampling': False,
            'random_starting_point_sampling': True,
            'num_samples': len(samples),
            'min_sequence_length': min_len,
            'max_sequence_length': max_len,
            't_all': t_all,
            'targets_all': t_all,
            'samples_per_epoch': SAMPLES_PER_EPOCH,
            'window_size': WINDOW_SIZE,
            'min_num_timesteps': MIN_SEQ_LEN,
            'sequence_dim': symbol_translation.get_num_symbols(),
            'symbol_to_int': symbol_translation.symbol_mapping,
            'int_to_symbol': symbol_translation.inverse_mapping,
        }
        data = {
            'metadata': metadata,
            'samples': samples
        }
        if not os.path.exists(f"{save_dir}/{DATASET_NAME}"):
            os.makedirs(f"{save_dir}/{DATASET_NAME}")
            
        save_loc = f"{save_dir}/{DATASET_NAME}/{split}.pt"
        torch.save(data, save_loc)
        print(f"Successfully saved list dataset at {save_loc}")
        
        data_torch = {
            'metadata': metadata,
            'samples': [
                {
                    'x': torch.tensor(sample['x']),
                    'sampling_times': torch.tensor(sample['sampling_times']),
                    'target_mask': sample['target_mask'],
                } for sample in samples
            ],
        }
        save_loc = f"{save_dir}/{DATASET_NAME}/{split}_torch.pt"
        torch.save(data_torch, save_loc)
        print(f"Successfully saved tensor dataset at {save_loc}")