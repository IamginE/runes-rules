# SYNTHETIC DATA GENERATION
import random
import tqdm

from .utils import calculate_statistics_after_generation


def rec_rule_placements_reg(
    x: list[int],
    elements_to_place: list[tuple[int, int, int]],
    tail_position: int
) -> list[list[tuple[int,int]]]:
    """
    Recursive function for `calculate_rule_placements`.
    
    Args:
        x (list[int]): Current state of a sequence during the generation process.
        elements_to_place (list[tuple[int, int, int]]): The remaining elements to place.
        tail_position (int): Where the target/tail item is placed.
        
    Returns:
        list[list[tuple[int,int]]]: A list with all possible placements of the head items. Each entry is a list (placement) with elements `(item, relative position)`.
    """
    # NOTE: keep track: We could also just place the minimal amounts of elements instead of one element per predicate by moving the x[abs_pos] == item check before the loop and always use this position for an item if it exists.
    if elements_to_place == []:
        return [[]]
    results = []
    item, int_low, int_high = elements_to_place.pop()
    for rel_pos in range(int_low, int_high + 1):
        abs_pos = tail_position - rel_pos
        if abs_pos < 0: # we cannot place things outside of the sequence
            break 
        if x[abs_pos] == item: # by magic, somebody else already placed the element correctly
            rec_possibilities = rec_rule_placements_reg(x, elements_to_place.copy(), tail_position)
            results += [[(item, rel_pos)] + p for p in rec_possibilities]
        elif x[abs_pos] == -1: # free placement
            x[abs_pos] = item # this is inplace, so we have to correct it later
            rec_possibilities = rec_rule_placements_reg(x, elements_to_place.copy(), tail_position)
            results += [[(item, rel_pos)] + p for p in rec_possibilities]
            x[abs_pos] = -1 # correction
    return results


def calculate_rule_placements_reg(
    x: list[int],
    rule: tuple[list[tuple[int, int, int]], list[tuple[float, int]]],
    tail_position: int
) -> list[list[tuple[int,int]]]:
    """
    Given the current state of a sample (with -1 indicating unused positions), a rule, and where to place the head item, calculate all possible rule placements.
    We use a naive implementation of checking all possibilities.
    This scales poorly with large rules (#elements and long time intervals) in large sequences, but is fine otherwise.
    We will leave the optimal solution for fast sampling in this problem to algorithm and data structure experts.

    Args:
        x (list[int]): Current state of a sequence during the generation process.
        rule (tuple[list[tuple[int, int, int]], list[tuple[float, int]]]): The rule that is used.
        tail_position (int): Where the target/tail item is placed.
        
    Returns:
        list[list[tuple[int,int]]]: A list with all possible placements of the head items. Each entry is a list (placement) with elements `(item, position)`.
    """
    # if any element cannot be placed at all w.r.t. the chosen position, the rule cannot be placed
    for element in rule[0]:
        int_low = element[1]
        if tail_position - int_low < 0:
            return []
    # Sort the list in descending order of interval lengths. Idea: place smaller-interval elements first in the recursive call (pop). No effect on worst case runtime, but potentially decreases amount of recursive calls.
    elements_to_place = sorted(rule[0], key=lambda x: x[2] - x[1], reverse=True)
    
    return rec_rule_placements_reg(x, elements_to_place, tail_position)


def calculate_rule_placement_fast_reg(
    x: list[int],
    rule: tuple[list[tuple[int, int, int]], list[tuple[float, int]]],
    tail_position: int,
    retries: int
) -> list[tuple[int,int]]:
    """
    Given the current state of a sample (with -1 indicating unused positions), a rule, and where to place the head item, calculate one random rule placement.
    This does not give a uniform distribution over all possibilites, but rather places each element uniformly at random.
    To see this, consider a[1,2] and b[1,3] and we uniformly at random select whether we first put a or b. Then p([(a,1),(b,2)]) = 14/48, but p([(a,1),(b,3)]) = 10/48.
    An empty list indicates no placement.

    Args:
        x (list[int]): Current state of a sequence during the generation process.
        rule (tuple[list[tuple[int, int, int]], list[tuple[float, int]]]): The rule that is used.
        tail_position (int): Where the target/tail item is placed.
        retries (int): The amount of retries. Since we don't exhaustively enumerat all possibilities, we don't know if the rule could be placed.
        
    Returns:
        list[tuple[int,int]]: A list with the placement of the head iems conisting of pairs `(item, position)`.
    """
    # if any element cannot be placed at all w.r.t. the chosen position, the rule cannot be placed
    for element in rule[0]:
        int_low = element[1]
        if tail_position - int_low < 0:
            return []
   
    elements_to_place = rule[0].copy()
    r = 0
    while r <= retries:
        placement = []
        random.shuffle(elements_to_place)
        used_rel_pos = set()
        
        for item, int_low, int_high in elements_to_place:
            possible_rel_pos = []
            for rel_pos in range(int_low, int_high + 1):
                abs_pos = tail_position - rel_pos
                if abs_pos < 0: # we cannot place things outside of the sequence
                    break 
                elif x[abs_pos] == item:
                    possible_rel_pos.append(rel_pos)
                elif x[abs_pos] == -1:
                    if rel_pos in used_rel_pos and (item, rel_pos) in placement: # we can only share positions if the item is the same for both placed predicates
                        possible_rel_pos.append(rel_pos)
                    if rel_pos not in used_rel_pos:
                        possible_rel_pos.append(rel_pos)
                
            if len(possible_rel_pos) == 0: # no possible placements in this order
                r += 1
                break
            else:
                rel_pos_choice = random.choice(possible_rel_pos)
                used_rel_pos.add(rel_pos_choice)
                placement.append((item, rel_pos_choice))
        
        if len(placement) == len(elements_to_place):
            return placement
    return [] 

              
# ============================================================================================
def generate_synthetic_data_targets_regular(
    num_samples: int,
    min_sequence_length: int,
    max_sequence_length: int,
    sequence_dim: int,
    rules: list[tuple[list[tuple[int, int, int]], list[tuple[float, int]]]],
    percentage_rules: float,
    frequency_rule_consequent: float,
    rule_confidence: float,
    uniform_random_placement: bool,
    ensure_disjoint: bool,
    random_seed: int = None,
    rule_weights: list[float] = None,
    retries_fast_placement: int = None,
) -> dict[str, dict | list[dict]]:
    """
    Generate a synthetic dataset according to the given rules. Here, we assume regular sampling, i.e., that two consecutive observations have a fixed temporal distance.
    Rules are of the form `item_1 [int_low_1, int_high_1] & item_2 [int_low_2, int_high_2] & ... --> item_target_1 | item_target_2, ....`. 
    We create a target mask such that only certain items need to be predicted (tail items + items that regulate frequency_rule_consequent and rule_confidence). This ensures that we know the ground truth by not introducing spurious rules via rule heads.
    
    A sequence is generated in 3 steps: 
    1) Place rules until the upper bound `percentage_rules` is reached or no rule can be placed anymore. Here we already incorporate the rule confidence.
        If we sample that a rule should hold, we place one of its tail items as target, otherwise we place a non-tail item as target.
    2) Place items included for prediction randomly in the remaining gaps until the `frequency_rule_consequent` constraint is satisfied, i.e., only a certain fraction of the prediction items is generated from rules.
    3) Fill the rest of the gaps with items that are not included in the target mask.

    To place a rule:
    1) We select a tail item's position from free positions.
    2) We randomly draw if the rule holds via the `rule_confidence`. If the rules holds, we pick a tail item according to the tail item's weight (probability).
        Otherwise, we pick a target item that is not from the rule tail uniformly at random.
    3) If `uniform_random_placement == True`, we generate all positions to place item_1, ...., item_n and pick one uniformly at random (Can be expensive especially with a lot of generation).
        Otherwise, just place each element uniformly at random, which overall does not give a uniform distribution.
    
    This procedure does technically not ensure that any of the constraints fully hold, because:
    - We might not be able to place enough rules to reach `percentage_rules`.
    - The `rule_confidence` can be changed by randomly placed targets that act as a tail, or randomly occuring heads with no corresponding tail.
    - Similarly, the `frequency_rule_consequent` may be influenced by random occurences of a rule head.
    
    Hence, we recalculate the statistics in the end.

    Args:
        num_samples (int): Number of samples to generate for the dataset.
        min_sequence_length (int): Minimum length of a sequence in the dataset.
        max_sequence_length (int): Maximum length of a sequence in the dataset. If max_sequence_length == min_sequence_length, all sequences will have the same length.
        sequence_dim (int): Dimensionality of the sequences/number of different items. Has to be at least the number of different tail and head items.
        rules (list[tuple[list[tuple[int, int, int]], list[tuple[float, int]]]]): A list of rules for regularly sampled sequences, where the length indicates the number of rules. 
            Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`, where the target item is sampled proportionally to its weight.
        percentage_rules (float): Relative upper bound on the positions that are ocuppied by placing rules (head or tails).
        frequency_rule_consequent (float): Approximate percentage of target items that are generated from rules.
        rule_confidence (float): Approximate probability that the tail of a rule appears after its head appears.
        uniform_random_placement (bool): Whether to place full rule heads uniformly at random over all their possible placements. If False, individual head items are instead placed at random, 
            which is much faster but does not guarantee a uniform placement over all possible head positions.
        ensure_disjoint (bool): Whether to ensure that the head and tail items across all rules are disjoint. This prevents possible rule chains that give rise to alternate rules, e.g., 0[2,3] -> 1 and 1[1,2] -> 2 introduces the rule 0[3,5] -> 2.
        random_seed (int, optional): Random seed to generate the dataset. Defaults to None. If None, no seed is set
        rule_weights (list[float], optional): The weights how likely each rule is used, which is converted into a probability distribution. If None, rules are sampled uniformly at random.
            Note that the rule distribution might not be fully correct, as it will not be possible to always generate sequences with the given rule distribution.
        retries_fast_placement (int): Retries for the fast placement method if `uniform_random_placement == False`. Has to be given if `uniform_random_placement == False`, otherwise ignored.

    Returns:
        dict[str,dict|list[dict]]: A dictionary with metadata and the samples. 
        The list of the samples consists of dictionaries with the following keys
            - `'x'`: (list[int]) Observed categorical variables x_1, ..., x_T, i.e., the sequence.
            - '`sampling_times`' (list[int]) The observation time points. Here None to indicate regular sampling.
            - `'used_rules'`: (list[int]) Indicators of which rule was used to generate a target item. Note that this is not an exhaustive list of rules that hold at this position.
            - `'T'`: Length of the sequence.
            - `'target_mask'`: (list[bool]) Target mask indicating which items should be included in the loss / considered for prediction.
            - `'occupied_by_rule'`: (int) The number of time steps in the sequence that were used to place rule heads and rule tails.
    """
    if not uniform_random_placement:
        assert retries_fast_placement is not None, "retries_fast_placement has to be given if uniform_random_placement == False."
    # fix the random seed
    if random_seed is not None:
        random.seed(random_seed)
    
    # Asserts to ensure we do not do anything unexpected
    assert min_sequence_length <= max_sequence_length, "min_sequence_length must be less than or equal to max_sequence_length."
    all_head_items, all_tail_items = [], []
    for head, tail in rules:
        for element in head:
            all_head_items.append(element[0])
            assert 0 < element[1]
            assert element[1] <= element[2]
        for target in tail:
            assert target[0] > 0, f"Only positive weights are allowed in the tail. Got {target[0]} instead."
            all_tail_items.append(target[1])
    all_head_items, all_tail_items = set(all_head_items), set(all_tail_items)
    if ensure_disjoint:
        assert len(all_head_items & all_tail_items) == 0, "With ensure_disjoint this methods expects that head/antecedent items and tail/consequent items are completely disjoint across all rules."
    for e in all_head_items & all_tail_items:
        assert 0 <= e < sequence_dim, "Expected items to be in range [0, sequence_dim - 1]"
    if rule_weights is not None:
        assert len(rules) == len(rule_weights), "The number of rules must match the number of rule weights."
    # we smoothly removed all duplicates via the two conversions   
    all_head_items: list[int] = list(all_head_items)
    all_tail_items: list[int] = list(all_tail_items)
    all_items: list[int] = list(range(sequence_dim))
    
    # convert the rule weights to a proper probability distribution
    if rule_weights is not None:
        rule_weights_cum = sum(rule_weights)
        rule_probs = [w/rule_weights_cum for w in rule_weights]
        
    else:
        rule_probs = [1/len(rules) for _ in rules]
    
    # ===== GENERATE THE SAMPLES =====
    samples = []
    for _ in tqdm.tqdm(range(num_samples), desc="Generating samples"):
        t_max = random.randint(min_sequence_length, max_sequence_length)
        free_pos = {i for i in range(t_max)}
        num_targets_from_rules = 0
        num_targets = 0
        occupied_by_rule = 0
        # we save this w.r.t. the tail item's position
        possible_pos_rules = [
            {j for j in range(t_max)} for _ in range(len(rules))
        ]
        # From which rule a target item was generated -1 if no rule was used to generate the item
        targets_from_rules = [-1 for _ in range(t_max)]
        x = [-1 for _ in range(t_max)]
        target_mask = [False for _ in range(t_max)]
        
        # STEP 1: PLACE THE RULES
        can_place_rule = True
        while occupied_by_rule / t_max < percentage_rules and can_place_rule:
            possible_rule_ids = [id for id in range(len(rules))]
            retry = True
            
            while(retry):
                # choose a rule
                if rule_weights is None:
                    rule_id = random.choice(possible_rule_ids)
                else:
                    rule_id = random.choices(possible_rule_ids, weights=[rule_weights[i] for i in possible_rule_ids])[0]
                rule = rules[rule_id]
                tail_items = [item for _, item in rule[1]]
                # Should the rule hold?
                rule_is_confident = random.random() <= rule_confidence
                
                # try to place the rule in the sequence starting with the tail item
                # lazy updates
                placement = None
                possible_pos_rules[rule_id] = free_pos & possible_pos_rules[rule_id]
                free_pos_rule = list(possible_pos_rules[rule_id])
                random.shuffle(free_pos_rule)
                
                for pos_tail in free_pos_rule:
                    if uniform_random_placement:
                        possible_placements = calculate_rule_placements_reg(x, rule, pos_tail) 
                        if len(possible_placements) > 0:
                            # uniformly at random choose a placement of the head items
                            placement = random.choice(possible_placements)
                            break
                        else: # update possible placements
                            possible_pos_rules[rule_id].remove(pos_tail)      
                    else:
                        placement = calculate_rule_placement_fast_reg(x, rule, pos_tail, retries_fast_placement) 
                        if len(placement) > 0:
                            break
                        else:
                            placement = None # no valid placement found
                            # update possible placements
                            possible_pos_rules[rule_id].remove(pos_tail)
                        
                # success
                if placement is not None:
                    # we can place the rule
                    retry = False
                    if rule_is_confident:
                        weights = [w for w, item in rule[1]]
                        item_to_place = random.choices(tail_items, weights=weights)[0]
                        targets_from_rules[pos_tail] = rule_id
                        num_targets_from_rules += 1
                        occupied_by_rule += 1
                    else:
                        # to ensure that we predict an item that is not from the rule while the rule head is active
                        # this does not count toward the targets generated by rules and the spaces that are occupied by a rule
                        items = [item for item in all_items if item not in tail_items]
                        item_to_place = random.choice(items)
                    x[pos_tail] = item_to_place
                    target_mask[pos_tail] = True
                    num_targets += 1
                    free_pos.remove(pos_tail)
                    
                    # asserts can be removed if we are really sure the code is right
                    head = rule[0]
                    for head_item, int_low, int_high in head:
                        matched = False
                        for it, rel_p in placement:
                            if head_item == it and int_low <= rel_p <= int_high:
                                matched = True
                                break
                        assert matched, f"Head item {head_item} to be placed in [{int_low},{int_high}] did not find a match in the generated placement."
                    
                    for it, rel_p in placement:
                        abs_pos = pos_tail - rel_p
                        if x[abs_pos] == -1:
                            x[abs_pos] = it
                            free_pos.remove(abs_pos)
                            occupied_by_rule += 1
                        else:
                            assert x[abs_pos] == it, f"The placement tried to overried the already occupied position {abs_pos} that has item {x[abs_pos]} with item {it}."
                
                # the rule does not fit anywhere    
                else:
                    possible_rule_ids.remove(rule_id)
                    if len(possible_rule_ids) == 0:
                        # no rules can be placed anymore
                        retry = False
                        can_place_rule = False
                
        # STEP 2: PLACE ADDITIONAL TARGET ELEMENTS THAT ARE NOT FROM RULES
        free_pos = list(free_pos)
        # TODO: Keep track of this choice: We ignore the first time step in the loss, so placing a target at the first time step does not influence the model
        flag_0 = 0 in free_pos
        if flag_0:
            free_pos.remove(0)
        while num_targets_from_rules / num_targets > frequency_rule_consequent and len(free_pos) > 0:
            pos = random.choice(free_pos)
            x[pos] = random.choice(all_items)
            target_mask[pos] = True
            num_targets += 1
            free_pos.remove(pos)
              
        # STEP 3: FILL ALL GAPS WITH NON-TARGET ITEMS
        if flag_0:
            free_pos.append(0)
        for p in free_pos:
            x[p] = random.choice(all_items)
        
        samples.append({
            'x': x, 
            'sampling_times': None, # regular sampling
            'used_rules': targets_from_rules, 
            'T': t_max, 
            'target_mask': target_mask,
            'occupied_by_rule': occupied_by_rule 
        })
    
    # STEP 4: RECALCULATE THE STATISTICS
    statistics = calculate_statistics_after_generation(samples, rules, regular_sampling=True)
        
    return {
        'metadata': {
            'synthetic': True,
            'regular_sampling': True,
            'random_starting_point_sampling': False,
            'num_samples': num_samples,
            'min_sequence_length': min_sequence_length,
            'max_sequence_length': max_sequence_length,
            'sequence_dim': sequence_dim,
            'rules': rules,
            'rule_probs': rule_probs,
            'rule_holds': statistics['num_rule_holds'],
            'rule_head_active': statistics['num_rule_head_active'],
            't_all': statistics['t_all'],
            'percentage_rules_given': percentage_rules,
            'frequency_rule_consequent_given': frequency_rule_consequent,
            'rule_confidence_given': rule_confidence,
            'targets_predictable_by_rules': statistics['targets_predictable_by_rules'],
            'targets_all': statistics['targets_all'],
            'percentage_rules_actual': statistics['percentage_rules'],
            'rel_support_individual_rules_actual': statistics['rel_rule_support_individual'],
            'confidence_individual_rules_actual': statistics['rule_confidence_individual'],
            'frequency_rule_consequent_actual': statistics['frequency_rule_consequent'],
            'avg_rule_confidence_actual': statistics['avg_rule_confidence'],
            'median_rule_confidence_actual': statistics['median_rule_confidence'],
            'ensure_disjoint': ensure_disjoint,
            'random_seed': random_seed,
        },
        'samples': samples
    }
# ============================================================================================
def head_match_rule(
    rule_1: tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]], 
    rule_2: tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]],
    regular_sampling: bool,
    epsilon_neurons_irregular: float = 0.0
) -> bool:
        """
        Check if two rules have the same head modulo a small variation in the interval bounds in the irregularly sampled case.
        
        Args:
            rule_1 (tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]): The first rule.
            rule_2 (tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]): The second rule.
            regular_sampling (bool): Whether the rules are for regularly sampled sequences. If True, the delay interval bounds are expected to be integers.
            epsilon_neurons_irregular (float, optional). Allowed deviation in interval bounds for equivalence checking of predicates in the irregularly sampled case. Defaults to 0.0. Only relevant if `regular_sampling = False`.
        """
        # NOTE: Due to the epsilon comparison in the irregular case, we cannot just check if both lists have equal lengths, but we should rather do it like this.
        for item, int_low, int_high in rule_1[0]:
            matched = False
            for item_2, int_low_2, int_high_2 in rule_2[0]:
                if regular_sampling: # bounds are integers
                    matched = (item == item_2) and (int_low == int_low_2) and (int_high == int_high_2)
                else: 
                    matched = (item == item_2) and abs(int_low - int_low_2) <= epsilon_neurons_irregular and abs(int_high - int_high_2) <= epsilon_neurons_irregular
                if matched:
                    break
            if not matched:
                return False
            
        for item, int_low, int_high in rule_2[0]:
            matched = False
            for item_2, int_low_2, int_high_2 in rule_1[0]:
                if regular_sampling: # bounds are integers
                    matched = (item == item_2) and (int_low == int_low_2) and (int_high == int_high_2)
                else: 
                    matched = (item == item_2) and abs(int_low - int_low_2) <= epsilon_neurons_irregular and abs(int_high - int_high_2) <= epsilon_neurons_irregular
                if matched:
                    break
            if not matched:
                return False
        return True
    

def predicate_subsume_check(
    predicate_1: tuple[int, int | float, int | float], 
    predicate_2: tuple[int, int | float, int | float],
    regular_sampling: bool,
    epsilon_irregular: float = 0.0
) -> bool:
    """Checks if predicate_1 subsumes predicate two, i.e., if `item_1 == item_2 and int_low_1 >= int_low_2 and int_high_1 <= int_high_2` in the regular case and
    if ` item_1 == item_2 and int_low_1 + epsilon_irregular >= int_low_2 and int_high_1 <= int_high_2 + epsilon_irregular` in the irregular case.

    Args:
        predicate_1 (tuple[int, int | float, int | float]): Predicate 1 (antecedent item with delay interval).
        predicate_2 (tuple[int, int | float, int | float]): Predicate 2 (antecedent item with delay interval).
        regular_sampling (bool): Whether sequences are regularly sampled.
        epsilon_irregular (float, optional). Allowed deviation in interval bounds for equivalence checking of predicates in the irregularly sampled case. Defaults to 0.0. Only relevant if `regular_sampling = False`.

        
    Returns:
        bool: The truth value.
    """
    item_1, int_low_1, int_high_1 = predicate_1
    item_2, int_low_2, int_high_2 = predicate_2
    if regular_sampling:
        return item_1 == item_2 and int_low_1 >= int_low_2 and int_high_1 <= int_high_2
    else:
        return item_1 == item_2 and int_low_1 + epsilon_irregular >= int_low_2 and int_high_1 <= int_high_2 + epsilon_irregular

    

def generate_random_rule_set_uniform_tails(
    num_rules: int,
    sequence_dim: int,
    min_size_head: int,
    max_size_head: int,
    min_size_tail: int,
    max_size_tail: int,
    min_delay: int | float,
    max_delay: int | float,
    min_span_int: int | float,
    max_span_int: int | float,
    regular_sampling: bool,
    random_seed: int = None,
    epsilon_irregular: float = 0.0
) -> list[tuple[list[tuple[int, int | float, int | float]], list[tuple[float, int]]]]:
    """
    Generate a random set of rules. Elements from the tail get unfirom weights.

    Args:
        num_rules (int): The number of rules to be included in the set.
        sequence_dim (int): The sequence dimensionlity, i.e, the number of items/events.
        min_size_head (int): Minimum number of elements in the head (antecedent).
        max_size_head (int): Maximum number of elements in the head (antecedent).
        min_size_tail (int): Minimum number of elements in the tail (consequent).
        max_size_tail (int): Maximum number of elements in the tail (consequent).
        min_delay (int | float): Minimum delay for head items with delay interval [a_i, b_i], i.e., `a_i >= min_delay`.
        max_delay (int | float): Maximum delay for head items with delay interval [a_i, b_i], i.e., `b_i <= max_delay`.
        min_span_int (int | float): The minimum span of a delay interval, i.e., `b_i - a_i >= min_span_int`.
        max_span_int (int | float): The maximum span of a delay interval, i.e., `b_i - a_i <= max_span_int`.
        regular_sampling (bool): Whether the rules are for regularly sampled sequences. If True, the delay interval bounds are integers.
        random_seed (int, optional): The random seed. If None, the random generator is not seeded
        epsilon_irregular (float, optional). Allowed deviation in interval bounds for equivalence checking of predicates in the irregularly sampled case. Defaults to 0.0. Only relevant if `regular_sampling = False`.

    Returns:
        list[tuple[list[tuple[int,float|int,float|int]],list[tuple[float,int]]]]: A list of rules, where the length indicates the number of rules. 
        Each rule is is a tuple of the form `(antecedent items, consequent items)`.
            - antecedent (head) items: A list of tuples `(item, int_low, int_high)` treated as conjunction.
            - consequent (tail) items: A list of tuples `(weight, target item)`, where the target item is sampled proportionally to its weight.
    """
    assert max_size_tail <= sequence_dim
    assert min_size_head <= max_size_head
    assert min_size_tail <= max_size_tail
    assert min_delay <= max_delay
    assert min_span_int <= max_span_int
    assert min_delay <= max_delay - min_span_int
    
    if random_seed is not None:
        random.seed(random_seed)
    rule_set = []
    while len(rule_set) < num_rules:
        num_head_elements = random.randint(min_size_head, max_size_head)
        num_tail_elements = random.randint(min_size_tail, max_size_tail)
        tail_elements_ = sorted(random.sample(range(sequence_dim), k=num_tail_elements))
        tail_elements = [(1.0/ len(tail_elements_), x) for x in tail_elements_]
        
        while True:
            head_elements = []
            while len(head_elements) < num_head_elements:
                item = random.randint(0, sequence_dim - 1)
                if regular_sampling:
                    int_low = random.randint(min_delay, max_delay - min_span_int)
                    int_high = random.randint(int_low + min_span_int, min(int_low + max_span_int, max_delay))
                    p = (item, int_low, int_high)
                else:
                    int_low = random.uniform(min_delay, max_delay - min_span_int)
                    int_high = random.uniform(int_low + min_span_int, min(int_low + max_span_int, max_delay))
                    p = (item, int_low, int_high)
                
                if any([(predicate_subsume_check(p, p2, regular_sampling, epsilon_irregular) or predicate_subsume_check(p2, p, regular_sampling, epsilon_irregular)) 
                        for p2 in head_elements]):
                    continue # avoid subsumption in the rule head
                head_elements.append(p)
            rule = (head_elements, tail_elements)
            if not any([head_match_rule(rule, r, regular_sampling, epsilon_irregular) for r in rule_set]):
                break
        rule_set.append(rule)
    return rule_set


def generate_fully_random_dataset_regular(
    num_samples: int,
    min_sequence_length: int,
    max_sequence_length: int,
    sequence_dim: int,
    random_seed: int = None,
) -> dict[str, dict | list[dict]]:
    """ 
    Args:
        num_samples (int): Number of samples to generate for the dataset.
        min_sequence_length (int): Minimum length of a sequence in the dataset.
        max_sequence_length (int): Maximum length of a sequence in the dataset. If max_sequence_length == min_sequence_length, all sequences will have the same length.
        sequence_dim (int): Dimensionality of the sequences/number of different items. Has to be at least the number of different tail and head items.
        random_seed (int, optional): Random seed to generate the dataset. Defaults to None. If None, no seed is set
        
    Returns:
        dict[str,dict|list[dict]]: A dictionary with metadata and the samples. 
        The list of the samples consists of dictionaries with the following keys
            - `'x'`: (list[int]) Observed categorical variables x_1, ..., x_T, i.e., the sequence.
            - '`sampling_times`' (list[int]) The observation time points. Here None to indicate regular sampling.
            - `'used_rules'`: (list[int]) Indicators of which rule was used to generate a target item, here None.
            - `'T'`: Length of the sequence.
            - `'target_mask'`: (list[bool]) Target mask indicating which items should be included in the loss / considered for prediction, here None as all elements are targets.
    """
    assert min_sequence_length <= max_sequence_length, "min_sequence_length must be less than or equal to max_sequence_length."
    # fix the random seed
    if random_seed is not None:
        random.seed(random_seed)
        
    # ===== GENERATE THE SAMPLES =====
    samples = []
    t_all = 0
    for _ in tqdm.tqdm(range(num_samples), desc="Generating samples"):
        x = []
        t_max = random.randint(min_sequence_length, max_sequence_length)
        t_all += t_max
        for t in range(t_max):
            x.append(random.randint(0, sequence_dim - 1))
        samples.append({
            'x': x, 
            'sampling_times': None, # regular sampling
            'used_rules': None, 
            'T': t_max, 
            'target_mask': None,
        })
    
    return {
        'metadata': {
            'synthetic': True,
            'regular_sampling': True,
            'random_starting_point_sampling': False,
            'num_samples': num_samples,
            'min_sequence_length': min_sequence_length,
            'max_sequence_length': max_sequence_length,
            'sequence_dim': sequence_dim,
            't_all': t_all,
            'targets_all': t_all,
            'random_seed': random_seed,
        },
        'samples': samples
    }