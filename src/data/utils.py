# UTILS NEEDED FOR DATASETS AND DATA-BASED REFINEMENT OF RULE SETS
from itertools import combinations
import numpy as np
import random
import torch
from torch.utils.data._utils.collate import default_collate
import tqdm

def compute_rule_active(
    x: list[int],
    pos: int,
    rule: tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]],
    regular_sampling: bool,
    sampling_times: list[float] = None
) -> tuple[bool,bool]:
    """
    Given a target position `pos` in sequence `x`, calculate if 1) the rule head is active and 2) if the target item can be predicted from a rule (is in its tail items).
    Assumes that there are no negations in the rule tail!

    Args:
        x (list[int]): The sequence.
        pos (int): The position of the target element.
        rule (tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]): The rule.
        regular_sampling (bool): Whether the sequence is regularly sampled. The rule should also reflect the regular or irregular sampling in its interval bounds.
        sampling_tims (list[float], optional): Sampling times. Defaults to None. Has to be given if `regular_sampling = False`, ignored otherwise.
        
    Returns:
        tuple[bool,bool]: Binary indicator if 1) a rule head is active and 2) if a rule head is active AND the target item is from the tail of the rule.
    """
    if not regular_sampling:
        assert sampling_times is not None
        assert len(x) == len(sampling_times)
    
    
    head, tail = rule
    tail_items = [item for _, item in tail]
    head_holds = True
    
    if regular_sampling: # delay interval bounds are integers
        for head_item, int_low, int_high in head:
            matched = False
            for offset in range(int_low, int_high + 1):
                if pos - offset < 0:
                    break # out of the possible sequence indeces
                if head_item == x[pos - offset]:
                    matched = True
                    break
            if not matched:
                head_holds = False
                break
    else:
        curr_time = sampling_times[pos]
        for head_item, int_low, int_high in head:
            matched = False
            # find the first item that is located in the delay interval
            pos_offset = 1
            time_delay = None
            while pos - pos_offset >= 0:
                delay = curr_time - sampling_times[pos - pos_offset]
                if delay >= int_low:
                    time_delay = delay
                    break
                pos_offset += 1
            
            if time_delay is not None:
                # try to find matches
                while int_low <= time_delay <= int_high:
                    if pos - pos_offset < 0:
                        break # out of the possible sequence indeces
                    if head_item == x[pos - pos_offset]:
                        matched = True
                        break
                    pos_offset += 1
                    time_delay = curr_time - sampling_times[pos - pos_offset]
            if not matched:
                head_holds = False
                break
                 
    return head_holds, head_holds and (x[pos] in tail_items)


def trim_ruleset(
    rule_set: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    confidences: list[float],
    rel_supports: list[float],
    min_confidence: float,
    min_rel_support: float
) -> tuple[
    list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    list[float],
    list[float]
]:
    """
    Given a rule set with corresponding confidences and relative_supports (IN THE SAME ORDER!) aswell as minimum confidence and minimum relative support thresholds,
    trim all rules that do not satisfy these 

    Args:
        rule_set (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float,int]]]]): A list of rules modeling a rule set.
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`. The weight is irrelevant here.
        confidences (list[float]): The confidences for the rules in the same order as in the rule set.
        rel_supports (list[float]): The relative supports for the rules in the same order as in the rule set.
        min_confidence (float): Minimum confidence.
        min_rel_support (float): Minimum relative support.

    Returns:
        tuple[list[tuple[list[tuple[int,int|float,int|float]],list[tuple[float,int]]]],list[float],list[float]]: Tuple consisting of
            - The updated rule set.
            - The updated confidences.
            - The updated relative supports.
    """
    assert len(rule_set) == len(confidences) == len(rel_supports)
    assert 0 <= min_confidence <= 1
    assert 0 <= min_rel_support <= 1
    
    trimmed_rule_set = []
    trimmed_confidences = []
    trimmed_rel_supports = []

    for rule, conf, supp in zip(rule_set, confidences, rel_supports):
        if conf >= min_confidence and supp >= min_rel_support:
            trimmed_rule_set.append(rule)
            trimmed_confidences.append(conf)
            trimmed_rel_supports.append(supp)

    print(f"Trimmed {len(rule_set) - len(trimmed_rule_set)} rules.")
    return trimmed_rule_set, trimmed_confidences, trimmed_rel_supports


def calculate_statistics_after_generation(
    samples: list[dict[str, int | list[int] | list[bool]]],
    rules: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    regular_sampling: bool,
) -> dict:
    """
    Given a list of generated samples and rules, calculate all important statistics to verify against the given generation parameters,
    `occupied_by_rule` only counts rules that we planted, not coincidental ones. Otherwise we would have to do exhaustive matching.
    However, using targets as anchor, we can easily calculate if a rule is active for that target, ignoring if there are multiple activations of the same rule.
    Note that the network only distinguishes between any activation and no activation of a rule.
    
    Args:
        samples (list[dict[str, int | list[int] | list[bool]]]): List of samples represented as dictionaries. Each sample consists of
            - `'x'`: (list[int]) The generated sequence.
            - `'sampling_times'`: (list[float]) The sampling time associated with the sequence x. Can be None. If None, regular sampling has to be used.
            - `'used_rules'`: (list[int]) A list that indicates if a target was generated from a rule.
            - `'T'`: (int) Length of the sequence.
            - `'target_mask'`: (list[bool]) Binary indicator list if an element is included as target. Can be None. If None, all elements are treated as targets.
            - `'occupied_by_rule'`: (int): Number of elements that were placed to have rule heads and tails.    
        rules (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): A list of rules, where the length indicates the number of rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`. The weight is irrelevant here.
        regular_sampling (bool): Whether sequences are regularly sampled. Rules should also reflect the regular or irregular sampling in their interval bounds.
            
    Returns:
        dict: Dictionary with
            - `'t_all'`: (int) The number of total time steps in the data.
            - `'percentage_rules'`: (float) How much of the data is made up by planted rule heads and tails.
            - `'num_rule_holds'`: (list[int]) How often a rule holds (tail placed after head).
            - `'num_rule_head_active'`: (list[int]) How often a rule head is active when an item is predicted.
            - `'rel_rule_support_individual'`: (list[float]) Relative rule support for the individual rules. 
            - `'rule_confidence_individual'`: (list[float]) Confidence of the individual rules, i.e., how often is does a rule hold (is supported) for a predicted item vs. how often the rule head is active.
            - `'avg_rule_confidence'`: (float) Average confidence across rules.
            - `'median_rule_confidence'`: (float) Average confidence across rules.
            - `'targets_predictable_by_rules'`: (int) Targets that can be predicted by any rule.
            - `'targets_all'`: (int) Overall number of targets.
            - `'frequency_rule_consequent'`: Fraction of target items that are tails of any rule.
    """
    t_all = 0
    occupied_by_rule_all = 0
    targets_all = 0
    targets_predictable_by_rules = 0
    num_rule_head_active = [0 for _ in rules]
    num_rule_holds = [0 for _ in rules]
    
    for sample in samples:
        occupied_by_rule: int = sample['occupied_by_rule']
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
        occupied_by_rule_all += occupied_by_rule
        
        for pos in range(len(x)):
            if not target_mask[pos]:
                continue
            rules_active = []
            for i in range(len(rules)):
                rule = rules[i]
                head_active, rule_active = compute_rule_active(x, pos, rule, regular_sampling=regular_sampling, sampling_times=sampling_times)
                if head_active:
                    num_rule_head_active[i] += 1
                if rule_active:
                    num_rule_holds[i] += 1 
                rules_active.append(rule_active)
            if any(rules_active):
                targets_predictable_by_rules += 1
    
    confidences = [num_rule_holds[i] / num_rule_head_active[i] if num_rule_head_active[i] > 0 else 0 for i in range(len(rules))]
    average_confidence = sum(confidences) / len(rules) if len(confidences) > 0 else 0
    median_confidence = (np.median(confidences)).item() if len(confidences) > 0 else 0
    rel_supports = [support / targets_all for support in num_rule_holds]
    return {
        't_all': t_all,
        'percentage_rules': occupied_by_rule_all / t_all,
        'num_rule_holds': num_rule_holds,
        'num_rule_head_active': num_rule_head_active,
        'rel_rule_support_individual': rel_supports,
        'rule_confidence_individual': confidences,
        'avg_rule_confidence': average_confidence,
        'median_rule_confidence': median_confidence,
        'targets_predictable_by_rules': targets_predictable_by_rules,
        'targets_all': targets_all,
        'frequency_rule_consequent': targets_predictable_by_rules / targets_all
    }


def get_activity_antecedent_items(
    x: list[int],
    pos: int,
    rule: tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]],
    regular_sampling: bool,
    sampling_times: list[float] = None
) -> tuple[list[bool], list[list[int | float]]]:
    """
    Given a target position `pos` in sequence `x`, return a boolean list in order of the antecedent (head) predicates indicating if the respective predicate (element) is active
    and another list indicating the relative delays of elemnts that satisfy the antecedent predicates.

    Args:
        x (list[int]): The sequence.
        pos (int): The position of the target element.
        rule (tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]): The rule.
        regular_sampling (bool): Whether the sequence is regularly sampled. The rule should also reflect the regular or irregular sampling in its interval bounds.
        sampling_tims (list[float], optional): Sampling times. Defaults to None. Has to be given if `regular_sampling = False`, ignored otherwise.
        
    Returns:
        tuple[list[bool],list[list[int|float]]]:
            - list[bool]: Binary indicator list if the predicates are active.
            - list[list[int | float]]: Relative time delay of elements that satisfy this predicate.
    """
    if not regular_sampling:
        assert sampling_times is not None
        assert len(x) == len(sampling_times)
    
    head, _ = rule
    indicators_head: list[bool] = []
    antecedent_satisfied_rel_time: list[list[int]] = [[] for _ in head]
    
    if regular_sampling: # delay interval bounds are integers
        for i, (head_item, int_low, int_high) in enumerate(head):
            matched = False
            for offset in range(int_low, int_high + 1):
                if pos - offset < 0:
                    break # out of the possible sequence indeces
                if head_item == x[pos - offset]:
                    matched = True
                    antecedent_satisfied_rel_time[i].append(offset)
            indicators_head.append(matched)
            
    else:
        curr_time = sampling_times[pos]
        for i, (head_item, int_low, int_high) in enumerate(head):
            matched = False
            # find the first item that is located in the delay interval
            pos_offset = 1
            time_delay = None
            while pos - pos_offset >= 0:
                delay = curr_time - sampling_times[pos - pos_offset]
                if delay >= int_low:
                    time_delay = delay
                    break
                pos_offset += 1
            
            if time_delay is not None:
                # try to find matches
                while int_low <= time_delay <= int_high:
                    if pos - pos_offset < 0:
                        break # out of the possible sequence indeces
                    if head_item == x[pos - pos_offset]:
                        matched = True
                        antecedent_satisfied_rel_time[i].append(time_delay)
                    pos_offset += 1
                    time_delay = curr_time - sampling_times[pos - pos_offset]
            indicators_head.append(matched)
                 
    assert len(indicators_head) == len(head) == len(antecedent_satisfied_rel_time)
    return indicators_head, antecedent_satisfied_rel_time
    

def check_interval_dict(
    x: list[int],
    pos: int,
    interval_dict: dict[int, dict[tuple[int, ...], tuple[int | float, int | float]]],
    regular_sampling: bool,
    sampling_times: list[float] = None
):
    """
    Given a target position `pos` in sequence `x` and an dictionary `interval_dict` as used in `simplify_antecedent_predicates()`,
    edit the dictionary in place to remove all interval bounds that do not work.

    Args:
        x (list[int]): The sequence.
        pos (int): The position of the target element.
        interval_dict: dict[int, dict[tuple[int, ...], tuple[int | float, int | float]]]: Interval dict as used in `simplify_antecedent_predicates()`. 
            The first key is the item. The nested dict contains a key that does not matter here and interval bounds.
        regular_sampling (bool): Whether the sequence is regularly sampled. The rule should also reflect the regular or irregular sampling in its interval bounds.
        sampling_tims (list[float], optional): Sampling times. Defaults to None. Has to be given if `regular_sampling = False`, ignored otherwise.
    """
    if not regular_sampling:
        assert sampling_times is not None
        assert len(x) == len(sampling_times)
        
        
    if regular_sampling: # delay interval bounds are integers
        for item in interval_dict.keys():
            del_keys = []
            item_dict = interval_dict[item]
            for key, (int_low, int_high) in item_dict.items():
                matched = False
                for offset in range(int_low, int_high + 1):
                    if pos - offset < 0:
                        break # out of the possible sequence indeces
                    if item == x[pos - offset]:
                        matched = True
                        break
                if not matched:
                    del_keys.append(key)
            for del_key in del_keys:
                del item_dict[del_key]
                    
    else:
        curr_time = sampling_times[pos]
        for item in interval_dict.keys():
            del_keys = []
            item_dict = interval_dict[item]
            for key, (int_low, int_high) in item_dict.items():
                matched = False
                # find the first item that is located in the delay interval
                pos_offset = 1
                time_delay = None
                while pos - pos_offset >= 0:
                    delay = curr_time - sampling_times[pos - pos_offset]
                    if delay >= int_low:
                        time_delay = delay
                        break
                    pos_offset += 1
                
                if time_delay is not None:
                    # try to find matches
                    while int_low <= time_delay <= int_high:
                        if pos - pos_offset < 0:
                            break # out of the possible sequence indeces
                        if item == x[pos - pos_offset]:
                            matched = True
                            break
                        pos_offset += 1
                        time_delay = curr_time - sampling_times[pos - pos_offset]
                if not matched:
                    del_keys.append(key)
            for del_key in del_keys:
                del item_dict[del_key]
    
    del_items = [k for k, v in interval_dict.items() if len(v) == 0]
    for del_item in del_items:
        del interval_dict[del_item]

       
def simplify_antecedent_predicates(
    samples: list[dict[str, int | list[int] | list[bool]]],
    rule_set: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    regular_sampling: bool,
    max_intersection_int: int = 8,
) -> tuple[
    list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    list[int]
]:
    """
    Given a list of samples and a rule set, check for predicates that imply other predicates. (If a -> b for all target items, b can be pruned).
    Furthermore, check for predicates with overlapping delays for the same item and unify them if possible.
    Since the antecedents are conjunctions, if two delay intervals for the same item overlap, do not subsume one another, but there is always an item that is in both intervals at the same time
    when a rule holds, then we can collapse two predicates a[s_1, s_2], a[t_1, t_2] into a([s_1, s_2] intersect [t_1, t_2]) without hurting performance (confidence or support).
    For multiple possible intersections (more than two intervals might be able to get merged), we heuristically merge predicates, starting with the highest number of intersected predicates.
    
    Args:
        samples (list[dict[str, int | list[int] | list[bool]]]): List of samples represented as dictionaries. Each sample consists of at least
            - `'x'`: (list[int]) The generated sequence.
            - `'sampling_times'`: (list[float]) The sampling time associated with the sequence x. Can be None. If None, regular sampling has to be used.
            - `'T'`: (int) Length of the sequence.
            - `'target_mask'`: (list[bool]) Binary indicator list if an element is included as target. Can be None. If None, all elements are treated as targets.
        rule_set (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): A list of rules modeling a rule set, where the length indicates the number of rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`.
        regular_sampling (bool): Whether sequences are regularly sampled. Rules should also reflect the regular or irregular sampling in their interval bounds.
        max_intersection_int (int, optional): The maximum number of predicates to consider for intersection (to avoid an exponential blowup in the worst case). Defaults to 8. 
            
    Returns:
        tuple[list[tuple[list[tuple[int,int|float,int|float]],list[tuple[float,int]]]],list[int]]: Tuple consisting of
            - The updated rule set.
            - The number of removed predicates per rule.
    """
    if len(rule_set) == 0: # rule set is empty
        return [], []
    # matrix read as row -> column
    predicate_implications = [torch.ones(len(r[0]), len(r[0]), dtype=torch.bool) for r in rule_set]
    overlapping_intervals_same_item: list[dict[int, list[tuple[int | float, int | float, int]]]] = [dict() for _ in range(len(rule_set))] 
    for i, rule in enumerate(rule_set):
        head, _ = rule
        for pos, (item, int_low, int_high) in enumerate(head):
            if item in overlapping_intervals_same_item[i]:
                overlapping_intervals_same_item[i][item].append((int_low, int_high, pos))
            else:
                overlapping_intervals_same_item[i][item] = [(int_low, int_high, pos)]
     
    # track the possible intersections for each kind of item       
    possible_intersections: list[dict[int, dict[tuple[int, ...], tuple[int | float, int | float]]]] = []
    for i, item_to_list_interval_dict in enumerate(overlapping_intervals_same_item):
        item_to_intersection_dict: dict[int, dict[tuple[int, ...], tuple[int | float, int | float]]] = {}
        for item, interval_list in item_to_list_interval_dict.items():
            if len(interval_list) <= 1: # no intersection possible
                continue
            
            for num_ints in range(2, min(len(interval_list), max_intersection_int) + 1): # combinations of 2 or more intervals
                for combo in combinations(range(len(interval_list)), num_ints):
                    # Start with first interval in the combination
                    low, high, pos = interval_list[combo[0]]
                    positions = [pos]

                    # Intersect with the rest
                    for idx in combo[1:]:
                        ilow, ihigh, ipos = interval_list[idx]
                        positions.append(ipos)
                        low = max(low, ilow)
                        high = min(high, ihigh)
                        if low > high:  # Empty intersection
                            break
                    else:
                        if item not in item_to_intersection_dict:
                            item_to_intersection_dict[item] = {}
                        item_to_intersection_dict[item][tuple(positions)] = (low, high)
        possible_intersections.append(item_to_intersection_dict)
    
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
        
        for pos in range(len(x)):
            if not target_mask[pos]:
                continue
            for i in range(len(rule_set)):
                rule = rule_set[i]
                head, tail = rule
                possible_intersections_rule = possible_intersections[i]
                pos_tail = [item for w, item in tail if w >= 0]
                neg_tail = [item for w, item in tail if w < 0]
                
                # check implications
                head_predicate_satisfied, _ = get_activity_antecedent_items(x, pos, rule, regular_sampling, sampling_times)
                for a in range(len(head)):
                    for b in range(len(head)):
                        if head_predicate_satisfied[a] and not head_predicate_satisfied[b]: # a -> b does not hold
                            predicate_implications[i][a,b] = False
                
                # check the intervals, we only need to check this if a) there is an intersection left and b) if the rule holds
                if len(possible_intersections_rule) == 0:
                    continue
                head_active = all(head_predicate_satisfied)
                rule_active = head_active and x[pos] in pos_tail
                if not rule_active and len(neg_tail) > 0 :
                    rule_active = head_active and x[pos] not in neg_tail
                if not rule_active:
                    continue
                check_interval_dict(x, pos, possible_intersections_rule, regular_sampling, sampling_times)
                
    # simplify the rules if possible
    new_rule_set = []
    num_removed_predicates = []
    for i, rule in enumerate(rule_set):
        implications = predicate_implications[i]
        intersection_dict = possible_intersections[i]
        delete_ids: set[int] = set()
        merges: list[tuple[int, ...]] = []
        new_head = []
        head, tail = rule
        head_items = [item for item, _, _ in head]
        
        # record all implications that hold everywhere
        for a in range(len(head)):
            for b in range(len(head)):
                if a == b:
                    continue
                elif implications[a,b] and implications[b,a]: # equivalence, don't want to delete both
                    delete_ids.add(b)
                    implications[b,a] = False # prevent putting both items in an equivalence in the delete ids
                elif implications[a,b]: # notably, implications are transitive
                    delete_ids.add(b)
                    
        # record merges
        for item_merge_dict in intersection_dict.values():
            for merge_pos, (int_low, int_high) in sorted(item_merge_dict.items(),  key=lambda x: len(x[0]), reverse=True): # sort by number of intersecting intervals
                if any([pos in delete_ids for pos in merge_pos]):
                    continue
                merges.append(merge_pos)
                delete_ids = delete_ids | set(merge_pos)

        # exectue merges
        for merge_pos in merges:
            item = head_items[merge_pos[0]]
            for pos in merge_pos:
                assert item == head_items[pos]
            int_low, int_high = intersection_dict[item][merge_pos]
            new_head.append((item, int_low, int_high))
        # add all remaining items, merged items are already in the delet ids
        new_head += [pred for id, pred in enumerate(head) if id not in delete_ids]
        new_rule_set.append((new_head, tail))
        num_removed_predicates.append(len(head) - len(new_head))
    
    assert len(new_rule_set) == len(num_removed_predicates) == len(rule_set)
    return new_rule_set, num_removed_predicates


def tighten_delay_intervals(
    samples: list[dict[str, int | list[int] | list[bool]]],
    rule_set: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    regular_sampling: bool,
) -> tuple[
    list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    list[float]
]:
    """
    Given a list of samples and a rule set, check if the delay intervals for predicates can be tightened.
    We only have to consider instances were the rule holds in order to keep the support and confidence because tighter intervals imply wider ones.
    
    Args:
        samples (list[dict[str, int | list[int] | list[bool]]]): List of samples represented as dictionaries. Each sample consists of at least
            - `'x'`: (list[int]) The generated sequence.
            - `'sampling_times'`: (list[float]) The sampling time associated with the sequence x. Can be None. If None, regular sampling has to be used.
            - `'T'`: (int) Length of the sequence.
            - `'target_mask'`: (list[bool]) Binary indicator list if an element is included as target. Can be None. If None, all elements are treated as targets.
        rule_set (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): A list of rules modeling a rule set, where the length indicates the number of rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`.
        regular_sampling (bool): Whether sequences are regularly sampled. Rules should also reflect the regular or irregular sampling in their interval bounds.
            
    Returns:
        tuple[list[tuple[list[tuple[int,int|float,int|float]],list[tuple[float,int]]]],list[int]]: Tuple consisting of
            - The updated rule set.
            - The average achieved interval reduction for each rule antecedent.
    """
    def update_intervals(possible_intervals: list[tuple[int | float, int | float]], delays: list[int | float]) -> list[tuple[int | float, int | float]]:
        """
        Given a list of sorted possible delay intervals (ascending order), in format `(int_low, int_high)`, and a list of sorted delays (ascending order), 
        where each interval must at least encompass one of the delays of the list.
        Update the interval list `possible_intervals` by doing minimal interval extensions if needed and avoiding intervals that would be subsumed.
        
        **Assumes that no subsumed intervals are in the sorted list of intervals, otherwise everything breaks!**
        
        Returns:
            list[tuple[int|float,int|float]]: The updated list.
        """
        new_possible_intervals: list[tuple[int | float, int | float]] = []
        if len(possible_intervals) == 0: # first time we insert an interval
            for delay in delays:
                new_possible_intervals.append((delay, delay))
        else:
            interval_id = 0
            delay_id = 0
            while interval_id < len(possible_intervals) and delay_id < len(delays):
                int_low, int_high = possible_intervals[interval_id]
                next_int_low, next_int_high = possible_intervals[interval_id + 1] if interval_id + 1 < len(possible_intervals) else None, None
                delay = delays[delay_id]
                next_delay = delays[delay_id + 1] if delay_id + 1 < len(delays) else None
                
                if delay < int_low:
                    if next_delay is None or next_delay > int_high: # possible minimal extension is to extend the lower bound
                        new_possible_intervals.append((delay, int_high))
                        delay_id += 1 # no later interval can be extended to this point, otherwise it would be subsumed by the current interval
                    else:
                        delay_id += 1 # there exists a tighter extension with the next delay
                elif int_low <= delay <= int_high: # interval does not have to be extended
                    new_possible_intervals.append((int_low, int_high))
                    interval_id += 1
                elif delay > int_high:
                    if next_int_high is None or delay <= next_int_high: # possible minimal extension is to extend the upper bound
                        new_possible_intervals.append((int_low, delay))
                        interval_id += 1 # an interval is at most extended in two ways: 1) lower bound (earlier), 2) upper bound (now, done)
                    else:
                        interval_id += 1 # an interval extension would be subsumed by the following interval extension of the next interval
        return new_possible_intervals
    

    if len(rule_set) == 0: # rule set is empty
        return [], []
    interval_tracker: list[list[list[tuple[int | float, int | float]]]] = [[[] for _ in head] for head, tail in rule_set]
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
        
        for pos in range(len(x)):
            if not target_mask[pos]:
                continue
            for i in range(len(rule_set)):
                rule = rule_set[i]
                head, tail = rule
                pos_tail = [item for w, item in tail if w >= 0]
                neg_tail = [item for w, item in tail if w < 0]
                
                head_predicate_satisfied, elements_in_interval = get_activity_antecedent_items(x, pos, rule, regular_sampling, sampling_times)
                # only consider target where the rule is active
                head_active = all(head_predicate_satisfied)
                rule_active = head_active and x[pos] in pos_tail
                if not rule_active and len(neg_tail) > 0 :
                    rule_active = head_active and x[pos] not in neg_tail
                if not rule_active:
                    continue
                
                for j in range(len(head)):
                    interval_tracker[i][j] = update_intervals(interval_tracker[i][j], delays=elements_in_interval[j])
    
    # tighten intervals if possible
    new_rule_set = []
    avg_reductions = [] 
    for i, rule in enumerate(rule_set):
        head, tail = rule
        new_head = []
        reductions = []
        interval_list_per_item = interval_tracker[i]
        for j, (item, int_low, int_high) in enumerate(head):
            interval_list = interval_list_per_item[j]
            new_int_low, new_int_high = min(interval_list, key=lambda x: x[1] - x[0]) # use the smallest avilable interval
            assert int_low <= new_int_low <= new_int_high <= int_high 
            new_head.append((item, new_int_low, new_int_high))
            reductions.append((int_high - int_low) - (new_int_high - new_int_low))
        assert len(new_head) == len(head)
        new_rule_set.append((new_head, tail))
        avg_reductions.append(sum(reductions) / len(reductions))
    assert len(new_rule_set) == len(rule_set) == len(avg_reductions)
    return new_rule_set, avg_reductions


def remove_unsupported_rules(
    samples: list[dict[str, int | list[int] | list[bool]]],
    rule_set: list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    regular_sampling: bool,
) -> tuple[
    list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]],
    int
]:
    """
    Given a list of samples and a rule set, check if rules are supported at all and remove them if not. Also works with negative tail elements.
    
    Args:
        samples (list[dict[str, int | list[int] | list[bool]]]): List of samples represented as dictionaries. Each sample consists of at least
            - `'x'`: (list[int]) The generated sequence.
            - `'sampling_times'`: (list[float]) The sampling time associated with the sequence x. Can be None. If None, regular sampling has to be used.
            - `'T'`: (int) Length of the sequence.
            - `'target_mask'`: (list[bool]) Binary indicator list if an element is included as target. Can be None. If None, all elements are treated as targets.
        rule_set (list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]): A list of rules modeling a rule set, where the length indicates the number of rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`.
        regular_sampling (bool): Whether sequences are regularly sampled. Rules should also reflect the regular or irregular sampling in their interval bounds.
            
    Returns:
        tuple[list[tuple[list[tuple[int,int|float,int|float]],list[tuple[float,int]]]],list[int]]: Tuple consisting of
            - The updated rule set.
            - The number of removed rules.
    """
    rule_active_list = [False for _ in rule_set]
    
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
        
        for pos in range(len(x)):
            if not target_mask[pos]:
                continue
            for i in range(len(rule_set)):
                rule = rule_set[i]
                head, tail = rule
                pos_tail = [item for w, item in tail if w >= 0]
                neg_tail = [item for w, item in tail if w < 0]
                head_active, _ = compute_rule_active(x, pos, rule, regular_sampling=regular_sampling, sampling_times=sampling_times)
                rule_active = head_active and x[pos] in pos_tail
                if not rule_active and len(neg_tail) > 0:
                    rule_active = head_active and x[pos] not in neg_tail
                if rule_active:
                    rule_active_list[i] = True
    new_rule_set = [rule for i, rule in enumerate(rule_set) if rule_active_list[i]]
    return new_rule_set, len(rule_set) - len(new_rule_set)


def convert_dataset_to_torch(
    data: dict[str, dict | list[dict]],
    save_path: str = None,
    has_padding: bool = True
) -> dict[str, dict | list[dict[str, torch.Tensor]]]:
    """
    Convert a dataset given as dictionary samples to PyTorch tensors.

    Args:
        data (dict[str, dict | list[dict]]): The dictionary generated by the data generation functions.
        save_path (str, optional): Path to save the converted dataset. Defaults to None. If None, the dataset is not saved. Should end with '.pt'.
        has_padding (bool, optional): Whether the dataset has any sequences that need to be padded. Defaults to True.
        
    Returns:
        dict[str,dict|list[dict[str,torch.Tensor]]]: A dictionary with metadata and the samples. 
        The list of the samples consists of dictionaries with the following keys
            - `'x'`: Observed categorical variables X_1, ..., X_T.
            - `'sampling_times'`: Real-valued sampling times. Can be None. If None, the sequence is treated as regularly sampled.
            - `'padding_mask'`: Mask for the padding of the sequences.
            - `'target_mask'`: Mask for the targeted timesteps that are used in the loss. Can be None. If None, all items are treated as targets.
            - `'T'`: The sequence length.
            - `'used_rules'`: Indicator for the used rules. Can be None, e.g., if the dataset is not synthetic.
    """
    if save_path is not None:
        assert save_path.endswith('.pt'), "The save path must end with '.pt'"
    
    metadata = data['metadata']
    samples = data['samples']
    
    # convert the samples to PyTorch tensors
    samples_as_tensors = []
    for sample in tqdm.tqdm(samples, desc="Converting samples to PyTorch tensors"):
        x = sample['x']
        sampling_times = sample['sampling_times']
        used_rules = sample['used_rules']
        target_mask = sample['target_mask']
        t_max = sample['T']
        
        assert len(x) == t_max
        x_torch = torch.tensor(x) # [t_max]
        if used_rules is not None:
            assert len(x) == len(used_rules)
            used_rules = torch.tensor(used_rules) # [t_max]
        if target_mask is not None:
            assert len(target_mask) == len(x)
            target_mask = torch.tensor(target_mask)
        if sampling_times is not None:
            assert len(sampling_times) == len(x)
            sampling_times = torch.tensor(sampling_times)
        
        padding_mask = None
        # pad x and used_rules to the maximum sequence length
        if has_padding:
            max_sequence_length: int = metadata['max_sequence_length']
            padding_mask = torch.zeros(max_sequence_length, dtype=torch.bool)
            padding_mask[t_max:] = True # [max_sequence_length]
            # although 0 is technically a class, we will mask it in the loss so there is no conflict
            x_torch_unpadded = x_torch
            x_torch = torch.zeros(max_sequence_length, dtype=x_torch.dtype)
            x_torch[:t_max] = x_torch_unpadded # [max_sequence_length] 
            # -2 since -1 indicates no rule rather than padding
            if used_rules is not None:
                used_rules_unpadded = used_rules
                used_rules = torch.zeros(max_sequence_length, dtype=used_rules.dtype) - 2
                used_rules[:t_max] = used_rules_unpadded # [max_sequence_length]
            if target_mask is not None:
                target_mask_unpadded = target_mask
                target_mask = torch.zeros(max_sequence_length, dtype=torch.bool) 
                target_mask[:t_max] = target_mask_unpadded # [max_sequence_length]
            if sampling_times is not None:
                sampling_times_unpadded = sampling_times
                sampling_times = torch.ones(max_sequence_length, dtype=sampling_times.dtype) + sampling_times.max() # ensure proper (causal) masking in our models
                sampling_times[:t_max] = sampling_times_unpadded # [max_sequence_length]

        
        samples_as_tensors.append({
            'x': x_torch,
            'sampling_times': sampling_times,
            'padding_mask': padding_mask,
            'target_mask': target_mask,
            'T': torch.tensor(t_max),
            'used_rules': used_rules,
        })
        
    return_dict = {
        'metadata': metadata,
        'samples': samples_as_tensors
    }
    
    if save_path is not None:
        torch.save(return_dict, save_path)
    
    return return_dict


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict[str, dict | list[dict[str, torch.Tensor]]]):
        """
        Args:
            data (dict[str, dict | list[dict[str, torch.Tensor]]]): Dictionary with the samples and some metadata.
        """
        self.metadata = data['metadata']
        self.samples = data['samples']
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample['x'], # sequence
            sample['sampling_times'], # only needed if sequence is irregularly sampled, otherwise typically None
            sample['padding_mask'], # only if there is padding needed in the dataset, otherwise None and no elements are treated as padding
            sample['target_mask'], # only if we predict only a subset of events, otherwise None and every event is used in the loss
            sample['T'], # sequence length (number of positions, not time window; for regular sampling #positions == time window)
            sample['used_rules'], # indicator that is only there for synthetic data, otherwise None; not used for model training
        )
        

class SequenceDatasetRandomSampling(torch.utils.data.Dataset):
    """
    Dataset for data that should be randomly sub-sampled during training.
    E.g., for one very long sequence, we can randomly sample a starting point, and return a sequence slice of a given length.
    """
    def __init__(
        self, 
        data: dict[str, dict | list[dict[str, torch.Tensor]]],
    ):
        """
        Args:
            data (dict[str, dict | list[dict[str, torch.Tensor]]]): Dictionary with the samples and some metadata.
        """
        self.metadata = data['metadata']
        self.samples = data['samples']
        self.samples_per_epoch = data['metadata']['samples_per_epoch'] # The number of samples (not batches!) that should be used for one training epoch. Since we sample random starting points, this is used as length.
        self.window_size = data['metadata']['window_size'] # window_size (int): The window length when sampling.
        self.min_num_timesteps = data['metadata']['min_num_timesteps'] #  min_num_timesteps (int): The minimum number of timesteps that should be in a sample. It should hold that `min_num_timesteps <= window_size`.
        assert self.window_size >= self.min_num_timesteps
        
        # trim out samples that do not have min_num_timesteps
        delete_ids = []
        for i, sample in enumerate(self.samples):
            if len(sample['x']) < self.min_num_timesteps:
                delete_ids.append(i)
        if len(delete_ids) > 0:
            print(f"WARNING: {len(delete_ids)} samples do not have the minimum amount of required timesteps. Deleting these entries.")
            self.samples = [sample for i, sample in enumerate(self.samples) if i not in delete_ids]
        assert len(self.samples) > 0, "There are no samples to sample from."
        
        # calculate the sampling weight for each sequence
        weights = []
        for sample in self.samples:
            weights.append(len(sample['x']) - self.min_num_timesteps + 1)
        weights = np.array(weights)
        self.weights = weights / weights.sum()
        
    def __len__(self):
        return self.samples_per_epoch # controls the number of samples loaded with a data loader
       
    
    def __getitem__(self, idx):
        # We ignore the idx given by the data loader (since we do random slicing of samples)
        # Sample a sequence proportional to to the number of it's starting points to get a uniform distribution over all possible starting points
        random_idx = np.random.choice(len(self.samples), p=self.weights)
        sample = self.samples[random_idx]
        sequence = sample['x']
        needs_padding = self.min_num_timesteps < self.window_size # can we have sequences that do not fill the full window?
        
        starting_point = random.randint(0, len(sequence)-self.min_num_timesteps)
        end_point = min(starting_point + self.window_size, len(sequence))
        t_max = end_point - starting_point
        x = sequence[starting_point:end_point]
        sampling_times = None
        target_mask = None
        padding_mask = None
        if sample['sampling_times'] is not None:
            sampling_times = sample['sampling_times'][starting_point:end_point]
        if sample['target_mask'] is not None:
            target_mask = sample['target_mask'][starting_point:end_point]
        
        padding_mask = None
        if needs_padding:
            padding_mask = torch.zeros(self.window_size, dtype=torch.bool)
            padding_mask[t_max:] = True # [window_size]
            # although 0 is technically a class, we will mask it in the loss so there is no conflict
            x_unpadded = x
            x = torch.zeros(self.window_size, dtype=x.dtype)
            x[:t_max] = x_unpadded # [window_size] 
            if target_mask is not None:
                target_mask_unpadded = target_mask
                target_mask = torch.zeros(self.window_size, dtype=torch.bool) 
                target_mask[:t_max] = target_mask_unpadded # [window_size]
            if sampling_times is not None:
                sampling_times_unpadded = sampling_times
                sampling_times = torch.ones(self.window_size, dtype=sampling_times.dtype) + sampling_times.max() # ensure proper (causal) masking in our models
                sampling_times[:t_max] = sampling_times_unpadded # [window_size]  
            
        return (
            x, # sequence
            sampling_times, # only needed if sequence is irregularly sampled, otherwise typically None
            padding_mask, # only if there is padding needed in the dataset, otherwise None and no elements are treated as padding
            target_mask, # only if we predict only a subset of events, otherwise None and every event is used in the loss
            t_max, # sequence length (number of positions, not time window; for regular sampling #positions == time window)
            None, # indicator that is only there for synthetic data, otherwise None; not used for model training
        )
        

def custom_collate_fn(batch):
    """
    Collate function to handle possible None values. If a field is None for any sample, it is set to None for the whole batch.
    """
    transposed = list(zip(*batch))  # transpose list of tuples
    collated = []
    for field_values in transposed:
        if any(v is None for v in field_values):
            collated.append(None)
        else:
            collated.append(default_collate(field_values))
    return tuple(collated)
