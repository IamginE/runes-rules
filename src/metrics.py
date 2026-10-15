# METRICS USED IN THE EVALUATION
from itertools import combinations
import lightning as L
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm

from data.utils import compute_rule_active
from models.rules import Rules

def optimal_iou_match(
    list_1: list[set], 
    list_2: list[set], 
    maximum_subset_size: int = 1, 
) -> tuple[float, float, list[float]]:
    """
    Calculate a matching IoU between two sets of sets: sim(list_1, list_2) = 1/len(list_1) * (sum_{A in list_1} max_{l subset list_2} IoU(A, Union(l))).
    And sim'(list_1, list_2) = median_{A in list_1} max_{l subset list_2} IoU(A, Union(l)).

    Args:
        list_1 (list[set]): The first list of sets.
        list_2 (list[set]): The second list of sets.
        maximum_subset_size (int, optional): The maximum subset size of list 2 to be considered. Defaults to 1.
       
    Returns:
        tuple[float,float,list[float]]: The average similarity, the median similarity, and the individual similarities.
    """
    # Create all possible subsets of set 2 to be considered
    def union_settuple(l: tuple[set]) -> set:
        s = set()
        for next_set in l:
            s = s.union(next_set)
        return s
    
    if len(list_1) == 0: # edge case
        return 0.0, 0.0, []
    subsets = []
    for size in range(0, maximum_subset_size + 1):
        subsets.extend([union_settuple(c) for c in combinations(list_2, size)])
    
    # Similarity
    ious: list[float] = []
    for set_a in list_1:
        max_match = 0
        if len(set_a) > 0:
            for set_b in subsets:
                iou_a_b = len((set_a & set_b)) / len((set_a | set_b))
                max_match = max(max_match, iou_a_b)
        ious.append(max_match)
    return sum(ious) / len(list_1), (np.median(ious)).item(), ious


def weights_adjusted_confidence(num_rule_holds_by_tail_elem: list[list[int]]) -> list[float]:
    """
    Given how often rules hold by tail elements as in `calculate_support_confidence()`, calculate weights to adjust the confidence.
    The weights are in [1/n, 1] and reflect the uncertainty when multiple tail elements are in a rule.
    Namely, `w = 1/n + (1 - 1/n) * (1 - H(P) / log(n))` where, `n` is the number of tail elements and `H(P)` is the entropy by turning the frequencies into a probability distribution.

    Args:
        num_rule_holds_by_tail_elem (list[list[int]]): For a list of rules, how often each rule holds regarding each tail element.

    Returns:
        list[float]: The weights in the same order as the rules
    """
    
    weights: list[float] = []
    
    for freqs in num_rule_holds_by_tail_elem:
        total = sum(freqs)
        n = len(freqs)
        if total == 0 or n == 0:
            weights.append(0)
            continue

        # Normalize frequencies to probabilities
        probs = [f / total for f in freqs]

        # Compute entropy (natural log), then normalize by log(n) (Entropy of a uniform distribution)
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(n)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        w = (1 / n) + (1 - 1 / n) * (1 - normalized_entropy)
        weights.append(w)
    
    assert len(num_rule_holds_by_tail_elem) == len(weights)
    return weights


def calculate_support_confidence(
    samples: list[dict[str, int | list[int] | list[bool]]],
    rule_set: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    regular_sampling: bool,
) -> dict:
    """
    Given a list of samples and a rule set, calculate the support, confidence, and other associated metrics for the individual rules and the rule set.
    Using targets as anchor, we can easily calculate if a rule is active for that target, ignoring if there are multiple activations of the same rule.
    Note that the model only distinguishes between any activation and no activation of a rule.
    Assumes that there are no negations in the rule tail!
    
    Args:
        samples (list[dict[str, int | list[int] | list[bool]]]): List of samples represented as dictionaries. Each sample consists of at least
            - `'x'`: (list[int]) The generated sequence.
            - `'sampling_times'`: (list[float]) The sampling time associated with the sequence x. Can be None. If None, regular sampling has to be used.
            - `'T'`: (int) Length of the sequence.
            - `'target_mask'`: (list[bool]) Binary indicator list if an element is included as target. Can be None. If None, all elements are treated as targets.
        rule_set (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): A list of rules modeling a rule set, where the length indicates the number of rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`. The weight is irrelevant here.
        regular_sampling (bool): Whether sequences are regularly sampled. Rules should also reflect the regular or irregular sampling in their interval bounds.
            
    Returns:
        dict: Dictionary with
            - `'t_all'`: (int) The number of total time steps in the data.
            - `'num_rule_holds'`: (list[int]) How often a rule holds (tail placed after head). (support)
            - `'num_rule_holds_by_tail_elem'`: (list[list[int]]) How often a rule holds broken down by tail item.
            - `'rel_rule_support_individual'`: (list[float]) Relative rule support for the individual rules. 
            - `'avg_rel_rule_support'`: (float) The average relative support for all rules.
            - `'median_rel_rule_support'`: (float) The median relative support for all rules.
            - `'num_rule_head_active'`: (list[int]) How often a rule head is active when an item is predicted. (active)
            - `'rule_confidence_individual'`: (list[float]) Confidence of the individual rules, i.e., how often does a rule hold (is supported) for a predicted item vs. how often the rule head is active.
            - `'avg_rule_confidence'`: (float) Average confidence across rules.
            - `'median_rule_confidence'`: (float) Median confidence across rules.
            - `'adjusted_rule_confidence_individual'`: (list[float]) Confidence of the individual rules, re-weighted to adjust for the tail size.
            - `'avg_rule_confidence_adjusted'`: (float) Avgerage adjusted confidence across rules.
            - `'median_rule_confidence_adjusted'`: (float) Median adjusted confidence across rules.
            - `'targets_predictable_by_rules'`: (int) Targets that can be predicted by any rule. (rule set support)
            - `'targets_all'`: (int) Overall number of targets.
            - `'frequency_rule_consequent'`: Fraction of target items that are tails of any rule. (relative rule set supoort)
    """
    t_all = 0
    targets_all = 0
    targets_predictable_by_rules = 0
    num_rule_head_active = [0 for _ in rule_set]
    num_rule_holds = [0 for _ in rule_set]
    num_rule_holds_by_tail_elem = [[0 for _ in r[1]] for r in rule_set]
    
    for sample in samples:
        t_max: int = sample['T']
        x: list[int] = sample['x']
        sampling_times: list[float] = sample['sampling_times']
        target_mask: list[bool] = sample['target_mask']
        assert t_max == len(x) 
        if not regular_sampling:
            assert sampling_times is not None
            assert t_max == len(sampling_times)
        if target_mask is None:
            target_mask = [True for _ in range(t_max)]
        else:
            assert t_max == len(target_mask)
        t_all += t_max
        targets_all += sum(target_mask)
        
        for pos in range(len(x)):
            if not target_mask[pos]:
                continue
            rules_active = []
            for i in range(len(rule_set)):
                rule = rule_set[i]
                head_active, rule_active = compute_rule_active(x, pos, rule, regular_sampling=regular_sampling, sampling_times=sampling_times) 
                if head_active:
                    num_rule_head_active[i] += 1
                if rule_active:
                    num_rule_holds[i] += 1 
                    for j, (_, tail_element) in enumerate(rule[1]):
                        if tail_element == x[pos]:
                            num_rule_holds_by_tail_elem[i][j] += 1
                rules_active.append(rule_active)
            if any(rules_active):
                targets_predictable_by_rules += 1
    
    confidences = [num_rule_holds[i] / num_rule_head_active[i] if num_rule_head_active[i] > 0 else 0 for i in range(len(rule_set))]
    rel_supports = [support / targets_all for support in num_rule_holds]
    average_confidence = sum(confidences) / len(confidences) if len(confidences) > 0 else 0  
    median_confidence = (np.median(confidences)).item() if len(confidences) > 0 else 0
    avg_rel_support = sum(rel_supports) / len(rel_supports) if len(rel_supports) > 0 else 0  
    median_rel_support = (np.median(rel_supports)).item() if len(rel_supports) > 0 else 0  
    
    weights_conf = weights_adjusted_confidence(num_rule_holds_by_tail_elem)
    adjusted_confidences = [confidences[i] * weights_conf[i] for i in range(len(rule_set))]
    average_adjusted_confidence = sum(adjusted_confidences) / len(adjusted_confidences) if len(adjusted_confidences) > 0 else 0  
    median_adjusted_confidence = (np.median(adjusted_confidences)).item() if len(adjusted_confidences) > 0 else 0
    
    return {
        't_all': t_all,
        'num_rule_holds': num_rule_holds,
        'num_rule_holds_by_tail_elem': num_rule_holds_by_tail_elem,
        'rel_rule_support_individual': rel_supports,
        'avg_rel_rule_support': avg_rel_support,
        'median_rel_rule_support': median_rel_support,
        'num_rule_head_active': num_rule_head_active,
        'rule_confidence_individual': confidences,
        'avg_rule_confidence': average_confidence,
        'median_rule_confidence': median_confidence,
        'adjusted_rule_confidence_individual': adjusted_confidences,
        'avg_rule_confidence_adjusted': average_adjusted_confidence,
        'median_rule_confidence_adjusted': median_adjusted_confidence,
        'targets_predictable_by_rules': targets_predictable_by_rules,
        'targets_all': targets_all,
        'frequency_rule_consequent': targets_predictable_by_rules / targets_all
    }
    
    
def calculate_agreement_rule_sets(
    samples: list[dict[str, int | list[int] | list[bool]]],
    rule_set_gt: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    rule_set_model: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    regular_sampling: bool,
    rule_matching_max: int = 1,
) -> dict[str, int | float | dict[int, float]]:
    """
    Given a list of samples and two rule sets, calculate their confusin matrix (set level) and related metrics by counting target items that they explain.
    A target event/item can be 'explained' by a rule set if at least one rule holds for that item. 
    Additionally, also record the events a rule explains for further metrics on the individual rule-level.
    
    Args:
        samples (list[dict[str, int | list[int] | list[bool]]]): List of samples represented as dictionaries. Each sample consists of at least
            - `'x'`: (list[int]) The generated sequence.
            - `'sampling_times'`: (list[float]) The sampling time associated with the sequence x. Can be None. If None, regular sampling has to be used.
            - `'T'`: (int) Length of the sequence.
            - `'target_mask'`: (list[bool]) Binary indicator list if an element is included as target. Can be None. If None, all elements are treated as targets.
        rule_set_gt (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): A list of rules modeling a ground truh rule set. These rules are treated as the correct rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`. The weight is irrelevant here.
        rule_set_model (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): A second rule set to be combared against the `rule_set_gt`.
        regular_sampling (bool): Whether sequences are regularly sampled. Rules should also reflect the regular or irregular sampling in their interval bounds.
        rule_matching_max (int, optional): The maximum number of rules to consider from set 2 when trying to match events predicted by a single rule in set 1. 
            If `>1`, the union of events from multiple rules is also considered, allowing a single rule in rule set 1 to be matched by multiple rules in rule set 2.
        
    Returns:
        dict[str,int|float,dict[int,float]]: Dictionary with entries `'TP'`, `'FP'`, `'TN'`, and `'FN'`, as well as the derived metrics and the IoU-scores for individual rule comparison
            - `'precision'`: positive predictive value
            - `'NPV'`: negative predictive value
            - `'recall'`: true positive rate
            - `'TNR'`': true negative rate
            - `'F1'`: f1-score
            - `'avg_iou_indiv_gt_model'`: Avg. IoU metric for matched events from the gt rule set to the model rule set
            - `'avg_iou_indiv_model_gt'`: Avg. IoU metric for matched events from the model rule set to the gt rule set.
            - `'avg_iou_indiv_symmetric'`: Average of the two asymmetric avg. IoU metrics.
            - `'avg_iou_indiv_gt_model_by_ant_len'`: Avg. IoU by antecedent (head) length of the gt rule set as dict, where the head length is the dictionary key.
            - `'avg_iou_indiv_model_gt_by_ant_len'`: Avg. IoU by antecedent (head) length of the model rule set as dict, where the head length is the dictionary key.
            - `'median_iou_indiv_gt_model'`: Media IoU metric for matched events from the gt rule set to the model rule set
            - `'median_iou_indiv_model_gt'`: Media IoU metric for matched events from the model rule set to the gt rule set.
            - `'median_iou_indiv_symmetric'`: Average of the two asymmetric median IoU metrics.
            - `'median_iou_indiv_gt_model_by_ant_len'`: Median IoU by antecedent (head) length of the gt rule set as dict, where the head length is the dictionary key.
            - `'median_iou_indiv_model_gt_by_ant_len'`: Median IoU by antecedent (head) length of the model rule set as dict, where the head length is the dictionary key.
    """
    tp = tn = fp = fn = 0
    explained_events_gt: list[set[tuple[int,int,int]]] = [set() for _ in rule_set_gt]
    explained_events_model: list[set[tuple[int,int,int]]] = [set() for _ in rule_set_model]
    
    for n, sample in enumerate(samples):
        t_max: int = sample['T']
        x: list[int] = sample['x']
        sampling_times: list[float] = sample['sampling_times']
        target_mask: list[bool] = sample['target_mask']
        assert t_max == len(x) 
        if not regular_sampling:
            assert sampling_times is not None
            assert t_max == len(sampling_times)
        if target_mask is None:
            target_mask = [True for _ in range(t_max)]
        else:
            assert t_max == len(target_mask)
        
        for pos in range(len(x)):
            if not target_mask[pos]:
                continue
            gt_rule_active = False
            model_rule_active = False
            
            for i in range(len(rule_set_gt)):
                rule = rule_set_gt[i]
                _, rule_active = compute_rule_active(x, pos, rule, regular_sampling=regular_sampling, sampling_times=sampling_times)
                if rule_active:
                    gt_rule_active = True
                    explained_events_gt[i].add((n, pos, x[pos]))
            
            for i in range(len(rule_set_model)):
                rule = rule_set_model[i]
                _, rule_active = compute_rule_active(x, pos, rule, regular_sampling=regular_sampling, sampling_times=sampling_times)
                if rule_active:
                    model_rule_active = True
                    explained_events_model[i].add((n, pos, x[pos]))
                    
                
            if gt_rule_active and model_rule_active:
                tp += 1
            elif not gt_rule_active and model_rule_active:
                fp += 1
            elif gt_rule_active and not model_rule_active: 
                fn += 1
            else:
                tn += 1
                
    avg_iou_indiv_gt_model, median_iou_indiv_gt_model, individual_iou_gt = optimal_iou_match(explained_events_gt, explained_events_model, maximum_subset_size=rule_matching_max)
    avg_iou_indiv_model_gt, median_iou_indiv_model_gt, individual_iou_model = optimal_iou_match(explained_events_model, explained_events_gt, maximum_subset_size=rule_matching_max)
    avg_iou_indiv_symmetric = 0.5 * (avg_iou_indiv_gt_model + avg_iou_indiv_model_gt)
    median_iou_indiv_symmetric = 0.5 * (median_iou_indiv_gt_model + median_iou_indiv_model_gt)
    
    assert len(individual_iou_gt) == len(rule_set_gt)
    assert len(individual_iou_model) == len(rule_set_model)
    
    iou_by_head_len_gt: dict[int, list[float]] = {}
    for rule, iou in zip(rule_set_gt, individual_iou_gt):
        head_len = len(rule[0])
        if head_len not in iou_by_head_len_gt:
            iou_by_head_len_gt[head_len] = [iou]
        else:
            iou_by_head_len_gt[head_len].append(iou)
    
    iou_by_head_len_model: dict[int, list[float]] = {}
    for rule, iou in zip(rule_set_model, individual_iou_model):
        head_len = len(rule[0])
        if head_len not in iou_by_head_len_model:
            iou_by_head_len_model[head_len] = [iou]
        else:
            iou_by_head_len_model[head_len].append(iou)
    
    precision = None
    if tp + fp > 0:
        precision = tp / (tp + fp)
    elif tp + fn == 0:
        precision = 1
    else:
        precision = 0
    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'precision': precision , # positive predictive value (PPV)
        'NPV': tn / (tn + fn) if tn + fn > 0 else None, # negative predictive value (NPV)
        'recall': tp / (tp + fn) if tp + fn > 0 else None, # TPR
        'TNR': tn / (tn + fp) if tn + fp > 0 else None,
        'f1': 2 * tp / (2*tp + fp + fn) if 2*tp + fp + fn > 0 else None,
        'avg_iou_indiv_gt_model': avg_iou_indiv_gt_model,
        'avg_iou_indiv_model_gt': avg_iou_indiv_model_gt,
        'avg_iou_indiv_symmetric': avg_iou_indiv_symmetric,
        'avg_iou_indiv_gt_model_by_ant_len': {k: sum(v)/len(v) for k, v in iou_by_head_len_gt.items()},
        'avg_iou_indiv_model_gt_by_ant_len': {k: sum(v)/len(v) for k, v in iou_by_head_len_model.items()},
        'median_iou_indiv_gt_model': median_iou_indiv_gt_model,
        'median_iou_indiv_model_gt': median_iou_indiv_model_gt,
        'median_iou_indiv_symmetric': median_iou_indiv_symmetric,
        'median_iou_indiv_gt_model_by_ant_len': {k: np.median(v).item() for k, v in iou_by_head_len_gt.items()},
        'median_iou_indiv_model_gt_by_ant_len': {k: np.median(v).item() for k, v in iou_by_head_len_model.items()},
    }
    
    
def structural_similarity_rules(
    rule_1: tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]], 
    rule_2: tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]],
    regular_sampling: bool
) -> float:
    """
    Similarity between two rules A => B, A' => B': IoU(B, B') * sim(A, A')
    sim(A, A') = 1/|A| * sum_{x[s1, s2] in A} max_{y[t1, t2] in A'} [x == y] * IoU([s1, s2], [t1, t2])

    Each rule is is a tuple of the form `(antecedent items, consequent items)`.
        - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
        - consequent (tail) items: A list of tuples `(weight, target item)`. The weight is irrelevant here.
            
    Args:
        rule_1 (tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]): Rule 1.
        rule_2 (tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]): Rule 2.
        regular_sampling (bool): Whether the interval bounds describe regular sampling or not.

    Returns:
        float: The asymmetric similarity.
    """
    cons_1 = set([x for _, x in rule_1[1]])
    cons_2 = set([x for _, x in rule_2[1]])
    iou_consequent = len(cons_1 & cons_2) / len(cons_1 | cons_2)
    
    sim_antecedent = 0
    for item_1, int_low_1, int_high_1 in rule_1[0]:
        max_match = 0
        for item_2, int_low_2, int_high_2 in rule_2[0]:
            if item_1 != item_2:
                continue
            if regular_sampling: # interval bounds are integers
                iou_intervals = len(set(range(int_low_1, int_high_1 + 1)) & set(range(int_low_2, int_high_2 + 1))) / len(set(range(int_low_1, int_high_1 + 1)) | set(range(int_low_2, int_high_2 + 1)))
            # NOTE: could consider a case that is irregularly sampled, but we know that distances are integers 
            else:
                # edge case if all interval bounds coincide:
                if int_high_1 == int_high_2 == int_low_1 == int_low_2:
                    iou_intervals = 1
                else:          
                    iou_intervals = (min(int_high_1, int_high_2) - max(int_low_1, int_low_2)) / (max(int_high_1, int_high_2) - min(int_low_1, int_low_2))
                    iou_intervals = max(iou_intervals, 0) # in case the intervals have no overlap, the above calculation is negative
            max_match = max(max_match, iou_intervals)
        sim_antecedent += max_match
    sim = (1/len(rule_1[0]) * sim_antecedent) * iou_consequent
    return sim
    
    
def structural_similarity_rulesets(
    rule_set_1: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    rule_set_2: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    regular_sampling: bool
) -> tuple[float, dict[int, float]]:
    """
    Similarity between two rule sets.
    sim(R, R') = 1/|R| * sum_{A => B in R} max_{A' => B' in R'} sim(A => B, A' => B')
    A rule set is given as a list of rules.  Each rule is is a tuple of the form `(antecedent items, consequent items)`.
        - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
        - consequent (tail) items: A list of tuples `(weight, target item)`. The weight is irrelevant here.

    Args:
        rule_set_1 (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): Rule set 1.
        rule_set_2 (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): Rule set 2.
        regular_Sampling (bool): Whether the interval bounds describe regular sampling or not.

    Returns:
        tuple[float,dict[int,float]]: 
        - float: The asymmetric similarity.
        - dict[int, float]: The similarity by antecedent (head) length.
    """
    if len(rule_set_1) == 0:
        return 0, {}
    
    sim = 0
    sim_by_head_len: dict[int, list[float]] = {}
    for rule_1 in rule_set_1:
        head_len = len(rule_1[0])
        max_match = 0
        for rule_2 in rule_set_2:
            sim_rules = structural_similarity_rules(rule_1, rule_2, regular_sampling)
            max_match = max(max_match, sim_rules)
        sim += max_match
        if head_len not in sim_by_head_len:
            sim_by_head_len[head_len] = [max_match]
        else:
            sim_by_head_len[head_len].append(max_match)
            
    return sim / len(rule_set_1), {k: sum(v)/len(v) for k, v in sim_by_head_len.items()}     
    
    
def kl_divergence_logits(
    log_p: torch.Tensor, 
    log_q: torch.Tensor, 
    given_as_logits: bool, 
    mask: torch.Tensor, 
    dim: int = -1
) -> torch.Tensor:
    """
    Compute the (unreduced) KL divergence D_KL(p||q) between two categorical distributions p and q, 
    given via softmax logits if `given_as_logits = True` or as log probabilities if `given_as_logits = False`.

    Args:
        log_p (torch.Tensor): [..., num_classes, ...] The softmax logits or log probabilities of distribution p.
        log_q (torch.Tensor): [..., num_classes, ...] The softmax logits or log probabilities of distribution q.
        given_as_logits (bool): Whether the distributions are given as softmax logits or log probabilities.
        mask (torch.Tensor): [...] A mask indicating which elements to mask in the loss. Can be None. If None, no elements are excluded from the loss.
        dim (int, optional): The dimension along which to compute the KL-divergence. Defaults to -1.

    Returns:
        torch.Tensor: [...] The KL-divergence that is only summed across the given class dimension.
    """
    # KL(p || q) = sum_i p_i * (log(p_i / q_i))
    # if p_i = 0, then p_i * log(p_i / q_i) = 0 by convention, even if q_i = 0 with the rational that lim_{x -> 0} x log(x) = 0
    # if q_i = 0 and p_i > 0, then p_i * log(p_i / q_i) = p_i * log(inf) = inf
    if given_as_logits:
        log_p = F.log_softmax(log_p, dim=dim) # [..., num_classes, ...]
        log_q = F.log_softmax(log_q, dim=dim) # [..., num_classes, ...]
    
    kl = torch.exp(log_p) * (log_p - log_q) # [..., num_classes, ...]
    # handle the case where p_i = 0 
    kl = kl.masked_fill(log_p == float("-inf"), 0)
    kl = kl.sum(dim=dim)
    return kl.masked_fill(mask, 0) if mask is not None else kl
    
    
def js_divergence_logits(
    log_p: torch.Tensor, 
    log_q: torch.Tensor, 
    given_as_logits: bool, 
    mask: torch.Tensor, 
    dim: int = -1
) -> torch.Tensor:
    """
    Compute the (unreduced) JS divergence D_JS(p||q) between two categorical distributions p and q, 
    given via softmax logits if `given_as_logits = True` or as log probabilities if `given_as_logits = False`.

    Args:
        log_p (torch.Tensor): [..., num_classes, ...] The softmax logits or log probabilities of distribution p.
        log_q (torch.Tensor): [..., num_classes, ...] The softmax logits or log probabilities of distribution q.
        given_as_logits (bool): Whether the distributions are given as softmax logits or log probabilities.
        mask (torch.Tensor): [...] A mask indicating which elements to mask in the loss. Can be None. If None, no elements are excluded from the loss.
        dim (int, optional): The dimension along which to compute the KL-divergence. Defaults to -1.

    Returns:
        torch.Tensor: [...] The JS-divergence that is only summed across the given class dimension.
    """
    if given_as_logits:
        log_p = F.log_softmax(log_p, dim=dim) # [..., num_classes, ...]
        log_q = F.log_softmax(log_q, dim=dim) # [..., num_classes, ...]

    # Mixture distribution
    m = 0.5 * (log_p.exp() + log_q.exp()) # [..., num_classes, ...]
    log_m = m.log()

    # KL divergences (avoids log(0) with masking)
    kl_pm = kl_divergence_logits(log_p, log_m, given_as_logits=False, mask=mask, dim=dim) # KL(p || m) [...]
    kl_qm = kl_divergence_logits(log_q, log_m, given_as_logits=False, mask=mask, dim=dim) # KL(q || m) [...]
    # masked elements are already 0
    js = 0.5 * (kl_pm + kl_qm) # [batch_size, max_sequence_length]
    return js


@torch.no_grad()
def calculate_agreement_prediction_models(
    model_gt: nn.Module,
    model: nn.Module,
    device_gt: torch.device,
    device_model: torch.device,
    data_loader: torch.utils.data.DataLoader,
) -> dict[str, float]:
    """
    Given two models and a next-event prediction dataset compute agreement metrics. 
    For the confusion matrix, the predictions of `model_gt` are treated as the true predictions.

    Args:
        model_gt (nn.Module): The first model (p).
        model (nn.Module): The second model (q).
        device_gt (torch.device): The device where the first model is located.
        device_model (torch.device): The device where the second model is located.
        data_loader (torch.utils.data.DataLoader): Data loader for the dataset.

    Returns:
        dict[str,float]: Dictionary with the metrics
            - `'agreement_prediction'`: (float) Agreement in terms of predictions that match exactly.
            - `'avg_kl_p_q'`: (float) The average KL divergence in the prediction distributions between the first and the second model.
            - `'avg_kl_q_p'`: (float) The average KL divergence in the prediction distributions between the second and the first model.
            - `'avg_js'`: (float) The average JS divergence in the prediction distributions between the first and the second model.
    """
    model_gt.eval()
    model.eval()
    
    total_count = 0
    predictions_gt = []
    predictions_model = []
    js_total = kl_p_q_total = kl_q_p_total = 0
    
    for batch in tqdm.tqdm(data_loader, desc="Calculating Agreement"):
        x, sampling_times, input_padding_mask, target_mask, _, _ = batch
        sampling_times_gt = sampling_times.to(device_gt) if sampling_times is not None else None
        sampling_times_model = sampling_times.to(device_model) if sampling_times is not None else None
        return_dict_gt = model_gt(x.to(device_gt), sampling_times_gt)
        return_dict_model = model(x.to(device_model), sampling_times_model)
        
        batch_size, sequence_length = x.size()
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.to(device_gt)
        if target_mask is not None:
            target_mask = target_mask.to(device_gt)
        
        # full window mask should be the same for both models, otherwise we are comparing models trained with a different setup
        full_window_mask = return_dict_gt["full_window_mask"] #  [batch_size, max_sequence_length]
        full_window_mask_model = return_dict_model["full_window_mask"] #  [batch_size, max_sequence_length]
        if full_window_mask is None:
            assert full_window_mask_model is None
        else:
            assert torch.all(full_window_mask == full_window_mask_model.to(device_gt))
        
        mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device_gt)
        if input_padding_mask is not None:
            mask = mask & (~input_padding_mask) # [batch_size, max_sequence_length]
        if full_window_mask is not None:
            mask = mask & full_window_mask # [batch_size, max_sequence_length]    
        if target_mask is not None:
            mask = mask & target_mask # [batch_size, max_sequence_length]   
        
        # TODO: keep track of the choice of skipping the first time step
        # we don't predict for the fist timestep
        mask = mask[:,1:] # [batch_size, max_sequence_length - 1]
        prediction_logits_gt: torch.Tensor = return_dict_gt['prediction_logits'] # [batch_size, max_sequence_length, sequence_dim]
        prediction_logits_model: torch.Tensor = return_dict_model['prediction_logits'].to(device_gt) # [batch_size, max_sequence_length, sequence_dim]
        prediction_logits_gt = prediction_logits_gt[:,1:] # [batch_size, max_sequence_length - 1, sequence_dim]
        prediction_logits_model = prediction_logits_model[:,1:] # [batch_size, max_sequence_length - 1, sequence_dim]
        
        predictions_gt += prediction_logits_gt.max(dim=-1).indices[mask].tolist()
        predictions_model += prediction_logits_model.max(dim=-1).indices[mask].tolist() 
        
        
        # note that the mask so far includes all elements that should be in the loss, but the functions need the negation
        kl_p_q = kl_divergence_logits(prediction_logits_gt, prediction_logits_model, given_as_logits=True, mask=~mask) # [batch_size, max_sequence_length - 1]
        kl_q_p = kl_divergence_logits(prediction_logits_model, prediction_logits_gt, given_as_logits=True, mask=~mask) # [batch_size, max_sequence_length - 1]
        js = js_divergence_logits(prediction_logits_gt, prediction_logits_model, given_as_logits=True, mask=~mask)     # [batch_size, max_sequence_length - 1]
        kl_p_q_total += kl_p_q.sum().item()
        kl_q_p_total += kl_q_p.sum().item()
        js_total += js.sum().item()
        total_count += mask.sum().item()

    return {
        'agreement_prediction': (torch.tensor(predictions_gt) == torch.tensor(predictions_model)).sum().item() / total_count,
        'avg_kl_p_q': kl_p_q_total / total_count,
        'avg_kl_q_p': kl_q_p_total / total_count,
        'avg_js': js_total / total_count
    }

 
@torch.no_grad()
def activation_histogram_rules(
    rules: Rules,
    data_loader: DataLoader
) -> torch.Tensor:
    """
    For a Rules model, return the histogram counting for how many elements a certain number of rules is active.

    Args:
        rules (Rules): The model.
        data_loader (DataLoader): The data loader of the dataset to evaluate the model on.

    Returns:
        torch.Tensor: The histogram counts, where the indices correspond to number of active rules.
    """
    assert rules.symbolic == True
    device = rules.device
    rules.eval() 
    count_rules = torch.zeros(rules.num_rules, device=device)
    
    for batch in data_loader:
        x, sampling_times, input_padding_mask, target_mask, _, _ = batch
        x = x.to(device)
        batch_size, sequence_length = x.size()
        if sampling_times is not None:
            sampling_times = sampling_times.to(device)
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.to(device)
        if target_mask is not None:
            target_mask = target_mask.to(device)
            
        return_dict = rules(x, sampling_times)
        rule_head_activations = return_dict["rule_head_activations"]  #  [batch_size, max_sequence_length, num_rules]
        full_window_mask = return_dict["full_window_mask"]
        
        mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
        if input_padding_mask is not None:
            mask = mask & (~input_padding_mask) # [batch_size, max_sequence_length]
        if full_window_mask is not None:
            mask = mask & full_window_mask # [batch_size, max_sequence_length]    
        if target_mask is not None:
            mask = mask & target_mask # [batch_size, max_sequence_length]   
        
        # TODO: keep track of this choice
        # we don't predict for the fist timestep
        mask = mask[:,1:]
        rule_head_activations = rule_head_activations[:,1:][mask] > 0.5 # [batch_prediction_timesteps, num_rules]
        rule_head_activations = rule_head_activations.sum(dim=-1) # [batch_prediction_timesteps]
        
        count_rules += torch.bincount(rule_head_activations, minlength=rules.num_rules)
    return count_rules.detach().cpu()
        
        
@torch.no_grad() 
def plot_top1_calibration(
    models: dict[str, L.LightningModule],
    n_bins: int,
    save_file: str,
    train_loader: DataLoader = None,
    test_loader: DataLoader = None,
    random_seed: int = None,
):
    """
    Plot the calibration curves for the given models.

    Args:
        models (dict[str, L.LightningModule]): A dictionary of `(model_name, model)` pairs for which to plot the calibration curves
        n_bins (int): The number of bins for calculating the calibration curve.
        save_file (str): The file to save the plot.
        train_loader (DataLoader, optional): The data loader for the training data. Defaults to None. If None, the calibration curve is not calculated for the training data.
        test_loader (DataLoader, optional): The data loader for the test data. Defaults to None. If None, the calibration curve is not calculated for the tests data.
        random_seed (int, optional): Random seed. Defaults to None. If None, no random seed is set.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    for model_name, model in models.items():
        model.eval()
        for data_split, data_loader in zip(['Train', 'Test'], [train_loader, test_loader]):
            subplot_num = 0 if data_split == 'Train' else 1
            ax = axs[subplot_num]
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)

            ax.set_title(f'{data_split} Calibration Curve', fontsize=14)
            ax.set_xlabel('Predicted confidence (Top-1)')
            ax.set_ylabel('Empirical accuracy')
            ax.grid(True)
            
            if data_loader is None:
                continue
            
            if random_seed is not None:
                L.seed_everything(random_seed)
            device = model.device
            true_labels = []
            top1_preds = []
            top1_probs = []
            
        
            for batch in data_loader:
                x, sampling_times, input_padding_mask, target_mask, _, _ = batch
                x = x.to(device)
                batch_size, sequence_length = x.size()
                if sampling_times is not None:
                    sampling_times = sampling_times.to(device)
                if input_padding_mask is not None:
                    input_padding_mask = input_padding_mask.to(device)
                if target_mask is not None:
                    target_mask = target_mask.to(device)
                    
                return_dict = model(x, sampling_times)
                prediction_logits = return_dict["prediction_logits"] # [batch_size, max_sequence_length, sequence_dim]
                full_window_mask = return_dict["full_window_mask"]
                
                mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=device)
                if input_padding_mask is not None:
                    mask = mask & (~input_padding_mask) # [batch_size, max_sequence_length]
                if full_window_mask is not None:
                    mask = mask & full_window_mask # [batch_size, max_sequence_length]    
                if target_mask is not None:
                    mask = mask & target_mask # [batch_size, max_sequence_length]   
                
                # TODO: keep track of this choice
                # we don't predict for the fist timestep
                mask = mask[:,1:]
                probs = F.softmax(prediction_logits, dim=-1)
                probs = probs[:,1:][mask] # [batch_prediction_timesteps, sequence_dim]
                top1_probs_, top1_pred_ = probs.max(dim=-1) # both [batch_prediction_timesteps]
                true_labels.append(x[:,1:][mask])
                top1_probs.append(top1_probs_)
                top1_preds.append(top1_pred_)
        
            # ---- Compute calibration curve ----
            true_labels = torch.cat(true_labels).detach().cpu().numpy()
            top1_probs = torch.cat(top1_probs).detach().cpu().numpy()
            top1_preds = torch.cat(top1_preds).detach().cpu().numpy()
            correct = (top1_preds == true_labels).astype(int)

            prob_true, prob_pred = calibration_curve(correct, top1_probs, n_bins=n_bins, strategy='uniform')

            # ---- Plot ----
            ax.plot(prob_pred, prob_true, marker='o', label=model_name)
            
    axs[0].legend(title='Models')
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close(fig)
