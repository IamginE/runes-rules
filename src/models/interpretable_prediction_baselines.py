import catboost
from itertools import product
import json
import numpy as np
import pandas as pd
import random
import xgboost

class PPM_MarkovModel:
    
    def __init__(self, k: int, sequence_dim: int):
        """
        Args:
            k (int): The maximum order of the PPM-C (Prediction by Partial Matching, method C) Markov Model (ALISTAIR MOFFAT, Implementing the PPM Data Compression Scheme).
            sequence_dim (int): The number of different elements (maximum id) in the processed sequences.
        """
        assert k >= 0
        self.k = k
        self.sequence_dim = sequence_dim
        self.counts: dict[tuple[int, ...], dict[int, int]] = {}
        
    
    def train_on_sequence(self, x: list[int], target_mask: list[bool] = None):
        """
        Adds the counts in the given sequence to the model. Can be used iteratively to fit the model on multiple sequences.

        Args:
            x (list[int]): The list of categorical events.
            target_mask (list[bool], optional): Binary indicator list if an element is included as target. Can be None. If None, all elements are treated as targets.
        """
        if target_mask is not None:
            assert len(x) == len(target_mask)
         
        # TODO: keep track of the choice of skipping the first time step
        for i in range(1, len(x)):
            if x[i] >= self.sequence_dim:
                raise ValueError(f"Encountered element during training that has id {x[i]}, which is larger than the given number of elements: {self.sequence_dim}.")
            
            if target_mask is not None and not target_mask[i]:
                continue
            
            for order in range(self.k+1):
                if i - order < 0:
                    break
                context = tuple(x[i - order:i])
                if not context in self.counts:
                    self.counts[context] = {}
                if not x[i] in self.counts[context]:
                    self.counts[context][x[i]] = 1
                else:
                    self.counts[context][x[i]] += 1
                    
                         
    def predict_probabilities(self, context: list[int] | tuple[int]) -> np.ndarray:
        already_matched = np.zeros(self.sequence_dim, dtype=bool)
        probs = np.ones(self.sequence_dim, dtype=float)
        
        for order in range(self.k, -2, -1): 
            # if 0-order context has never been observed, the model is empty for some symbols
            if order == -1:
                probs[~already_matched] *= 1 / (~already_matched).sum()
                break 
            
            if len(context) < order:
                continue
            
            order_context = tuple(context[-order:]) if order > 0 else ()
            newly_matched = np.zeros(self.sequence_dim, dtype=bool)
            order_counts = np.zeros(self.sequence_dim, dtype=float)
            if order_context in self.counts:
                for element, c in self.counts[order_context].items():
                    order_counts[element] = c
                    newly_matched[element] = True
            
            # exclusion mechanism
            order_counts *= (~already_matched)
            newly_matched &= ~already_matched
            
            if order_counts.sum() > 0:
                already_matched |= newly_matched
                if already_matched.sum() == self.sequence_dim:
                    # all symbols have been encountered, don't need to escape
                    order_probs = order_counts / order_counts.sum()
                    probs[newly_matched] *= order_probs[newly_matched]
                    break
                else:
                    escape_count = newly_matched.sum() # partial exclusion: len(self.counts[order_context]), full exclusion: newly_matched.sum()
                    order_probs = order_counts / (order_counts.sum() + escape_count)
                    probs[newly_matched] *= order_probs[newly_matched]
                    # escape probability
                    probs[~already_matched] *= escape_count / (order_counts.sum() + escape_count)
            # if the counts are 0 at this level, the escape probability is 1, and multiplying by 1 does not change the current probability vector
        
        return probs
    
    
    def get_counts(self) -> dict[tuple[int, ...], dict[int, int]]:
        return self.counts
    
    
    def count_matched_orders(self, context: list[int] | tuple[int]) -> int:
        """
        For a given context, return the number of matched, non-zero contexts.
        """
        already_matched = np.zeros(self.sequence_dim, dtype=bool)
        count_matches = 0
        
        for order in range(self.k, 0, -1): 
            
            if len(context) < order:
                continue
                
            order_context = tuple(context[-order:]) if order > 0 else ()
            newly_matched = np.zeros(self.sequence_dim, dtype=bool)
            if order_context in self.counts:
                for element, c in self.counts[order_context].items():
                    newly_matched[element] = True
            
            # exclusion mechanism
            newly_matched &= ~already_matched
            
            if np.any(newly_matched):
                count_matches += 1
                already_matched |= newly_matched
                if already_matched.sum() == self.sequence_dim:
                    break

        return count_matches
    
    
    def prune_contexts(self, remaining_contexts: int):
        """
        Prune the model to only contain `remaining_contexts` number of contexts (`P( * | x_{1}, x_{2}, ...,  x_{k})`) that are not 0-order.
        Note that we remove the full frequencies for all symbols appearing after x_{1}, ...,  x_{k}.
        We keep the entries with the highest log-likelihood ratio:
        `-ln(L(D | M_0)) + ln(L(D | M_1)) = sum_{x_{k+1} in symbols} count(x_{1}, x_{2}, ...,  x_{k}, x_{k+1}) * ln(P(x_{k+1} | x_{1}, x_{2}, ...,  x_{k}) / P(x_{k+1} | x_{2}, ...,  x_{k}))`.
        
        Args:
            remaining_contexts (int): The number of contexts that remain, not counting the 0-order context.
        """
        # NOTE: Since we allocate fallback mass (smooth the distribution), the log-likelihood can get negative.
        # Otherwise it would be equal to counts(x_{1}, x_{2}, ...,  x_{k}) * KL(P(x_{k+1} | x_{1}, x_{2}, ...,  x_{k}) || P(x_{k+1} | x_{2}, ...,  x_{k})) and thus always non-negative.
        # Two alternatives could be: 
        # 1) Use counts(x_{1}, x_{2}, ...,  x_{k}) * KL(P(x_{k+1} | x_{1}, x_{2}, ...,  x_{k}) || P(x_{k+1} | x_{2}, ...,  x_{k})) on the smoothed distributions.
        # 2) Use counts(x_{1}, x_{2}, ...,  x_{k}) * KL(P(x_{k+1} | x_{1}, x_{2}, ...,  x_{k}) || P(x_{k+1} | x_{2}, ...,  x_{k})) on the unsmoothed distributions.
        
        # calculate the test statistics
        test_statistics: dict[tuple[int, ...], int] = {}
        learned_contexts = sorted(self.counts.keys(), key = lambda x: len(x), reverse = True)
        for context in learned_contexts:
            if len(context) == 0:
                continue
            p_0 = self.predict_probabilities(context[1:]) # null hypothesis
            p_1 = self.predict_probabilities(context)     # tested hypothesis
            n = np.zeros(self.sequence_dim) # how often do we encounter a symbol (in the training data) where the likelihood changes
            for element, c in self.counts[context].items():
                n[element] = c
            test_statistic = np.sum(n * np.log(p_1 / p_0))
            test_statistics[context] = test_statistic.item()
            
        # prune the model
        ranked_contexts = [context_stats[0] for context_stats in sorted(test_statistics.items(), key = lambda x: x[1])]
        for context in ranked_contexts:
            if(len(self.counts) <= remaining_contexts + 1): # +1 for 0-order
                break
            del self.counts[context]
    
  

def convert_list_dataset_to_tree_dataset(
    samples: list[dict[str, int | list[int] | list[bool]]],
    regular_sampling: bool,
    max_relative_distance: float,
    padding_id: int,
    treat_irregular_as_regular: bool = False,
    ensure_length: int = None,
    ensure_max_length: int = None,
    expected_num_classes: int = None,
    num_samples_random_left_padding: int = None,
) -> tuple[list[list[int]], list[list[float]], list[int]]:
    """Convert a sequence dataset given as lists to a tabular dataset to use trees on.

    Args:
        samples (list[dict[str, int | list[int] | list[bool]]]): List of samples represented as dictionaries. Each sample consists of at least
            - `'x'`: (list[int]) The generated sequence.
            - `'sampling_times'`: (list[float]) The sampling time associated with the sequence x. Can be None. If None, regular sampling has to be used.
            - `'target_mask'`: (list[bool]) Binary indicator list if an element is included as target. Can be None. If None, all elements are treated as targets.
        regular_sampling (bool): Whether sequences are regularly sampled.
        max_relative_distance (float): Maximum relative temporal distance to consider for the observations.
        padding_id (int): The id used for padding.
        treat_irregular_as_regular (bool, optional): Flag for experiments that treat irregular sequences as regularly sampled but use a lookback window (max_relative_distance) based on the sampling times (for comparability between models). Defaults to False.
        ensure_length (int, optional): If set, ensures that all samples from converted sequences have exactly this length (at most that many features), by left-cutting or left-padding all sequences.
        ensure_max_length (int, optional): If set, ensures that all samples from converted sequences have at most this length (at most that many features), by left-cutting all sequences that are too long.
        expected_num_classes (int, optional): If set, ensures that for every class in {0, 1, ..., expected_num_classes}, there is at least on entry in the dataset, by adding one dummy entry for every unseen class.
        num_random_left_padding (int, optional): If set, augments that dataset by creating samples where a random amount of context is removed (i.e., it creates samples by shortening sequences).

    Returns:
        tuple[list[list[int]],list[list[float]],list[int]]: tuple[list[list[int]],list[list[float]],list[int]]: A tuple with
            - The categorical lag features for each sample in the tabular dataset.
            - The time features for each sample in the tabular dataset (empty for regularly sampled data).
            - The categorical target for each sample in the tabular dataset.
    """
    if ensure_length is not None and ensure_max_length is not None:
        raise ValueError("Cannot set ensure_length and ensure_max_length simultaneously.")
    if treat_irregular_as_regular:
        assert regular_sampling, "regular_sampling should be True when treat_irregular_as_regular is True."
    if regular_sampling and not treat_irregular_as_regular:
        assert isinstance(max_relative_distance, int)
    time_features = []
    features = []
    targets = []
    
    for sample in samples:
        x: list[int] = sample['x']
        sampling_times: list[float] = sample['sampling_times']
        target_mask: list[bool] = sample['target_mask']
        
        if not regular_sampling or treat_irregular_as_regular:
            assert sampling_times is not None
            assert len(x) == len(sampling_times)
        if target_mask is not None:
            assert len(x) == len(target_mask)
        
        
        # TODO: keep track of the choice of skipping the first time step
        for i in range(1, len(x)):
            if target_mask is not None and not target_mask[i]:
                continue
            
            targets.append(x[i])
            if regular_sampling and not treat_irregular_as_regular:
                features.append(x[max(0,i-max_relative_distance):i])
                time_features.append([]) # not relevant
            else:
                # find the allowed window
                j = i - 1
                pred_time = sampling_times[i]
                while j >= 0:
                    if pred_time - sampling_times[j] > max_relative_distance:
                        break
                    j -= 1
                features.append(x[j+1:i])
                if not regular_sampling:  
                    time_diffs = pred_time - np.array(sampling_times[j+1:i])
                    time_features.append(list(time_diffs))
                else:
                    time_features.append([]) # not relevant
            
        
    if num_samples_random_left_padding is not None:
        random.seed(1445)
        MAX_RETRIES = 10
        original_size = len(features) 
        for _ in range(num_samples_random_left_padding):
            retries = 0
            while retries < MAX_RETRIES:
                i = random.randint(0, original_size - 1)
                if len(features[i]) <= 1: # cannot remove any context
                    retries += 1
                else:
                    k = random.randint(1, len(features[i]) - 1)
                    targets.append(targets[i])
                    features.append(features[i][-k:])
                    if time_features[i] == []:
                        time_features.append([])
                    else:
                        time_features.append(time_features[i][-k:])
                    break
    
    if expected_num_classes is not None:
        seen_classes = set(targets)
        for i in range(expected_num_classes):
            if i not in seen_classes:
                # make a dummy entry
                targets.append(i)
                features.append([])
                time_features.append([])        
      
    assert len(features) == len(time_features)              
    max_len = max([len(f) for f in features])
    if ensure_length is not None:
        max_len = max(ensure_length, max_len)          
    
    for i, (f, t) in enumerate(zip(features, time_features)):
        # left-pad the features, such that the closest sequence elements have the highest column
        f = [(elem + 1) if elem >= padding_id else elem for elem in f]
        if len(f) < max_len:
            f = (max_len - len(f)) * [padding_id] + f
        if ensure_length is not None:
            f = f[-ensure_length:]
        if ensure_max_length is not None:
            f = f[-ensure_max_length:]
            
        if not regular_sampling:
            if len(t) < max_len:
                t = (max_len - len(t)) * [max_relative_distance] + t
            if ensure_length is not None:
                t = t[-ensure_length:]
            if ensure_max_length is not None:
                t = t[-ensure_max_length:]
        features[i] = f
        time_features[i] = t
    
    return features, time_features, targets


def pad_same_length(x1: np.ndarray, x2: np.ndarray, padding_value) -> tuple[np.ndarray, np.ndarray]:
    """
    Given two 2-dimensional numpy array, ensure that both have the same second dimension.
    If one in smaller, left-pad if with the padding_value.
    """
    cols1 = x1.shape[1]
    cols2 = x2.shape[1]
    max_cols = max(cols1, cols2)
    pad1 = max_cols - cols1
    pad2 = max_cols - cols2
    
    x1 = np.pad(x1, ((0, 0), (pad1, 0)), mode='constant', constant_values=padding_value)
    x2 = np.pad(x2, ((0, 0), (pad2, 0)), mode='constant', constant_values=padding_value)
    
    return x1, x2


def xgboost_extract_tree_info(model: xgboost.XGBClassifier, num_classes: int) -> dict:
    """
    Extract per-tree leaf information from an XGBoost model.
    Args:
        model (xgboost.XGBClassifier): The trained classifier.
        num_classes (int): The number of prediction labels (classes).

    Returns:
        dict: {
            tree_id: {
                "num_leaves": int,
                "leaves": {
                    leaf_node_id: {
                        "depth": int,
                        "conditions": tuple[(feature, op, threshold)]
                    }
                }
            }
        }
    """
    try:
        best_iteration = model.best_iteration
    except AttributeError:
        best_iteration = None
        
    booster = model.get_booster()
    df = booster.trees_to_dataframe()
    children_to_parent = {}

    for _, row in df.iterrows():
        parent_id = row["Node"]
        
        if best_iteration is not None and row['Tree'] >= (best_iteration + 1) * num_classes:
            continue # model saves all trees, even the ones that would be removed by early stopping; disregard trees not used for prediction
        if row["Feature"] == "Leaf":
            continue
        
        if(not pd.isna(row["Split"])) and (row["Category"] is None):
            # numerical split
            children_and_operators = [("Yes", "<"), ("No", ">=")]
            kind = 'numerical'
        elif (pd.isna(row["Split"])) and (row["Category"] is not None) and isinstance(row["Category"], list):
            # categorical split
            children_and_operators = [("Yes", "in"), ("No", "not in")]
            kind = 'categorical'
        else:
            raise ValueError(f"Unexpected entry for a node: {row}")
        
        if row["Missing"] == row["Yes"]:
            missing_operator = children_and_operators[0][1]
        elif row["Missing"] == row["No"]:
            missing_operator = children_and_operators[1][1]
        else:
            raise ValueError("Missing child does not match Yes/No")
        
        
        for child_col, operator in children_and_operators:
            child = row[child_col]
            if pd.notna(child):
                # this is the original implementation that sorts alphabetically, then maps to int, which is unintuitive but still results in a unique representation
                # more intuitive: tuple(sorted(map(int, row["Category"]))) 
                split_condition = float(row["Split"]) if kind == 'numerical' else tuple(map(int, sorted(row["Category"])))
                tree_id, child_id = child.split('-')
                tree_id, child_id = int(tree_id), int(child_id)
                children_to_parent[(tree_id, child_id)] = (
                    tree_id,
                    parent_id,
                    row["Feature"],
                    operator,
                    split_condition,
                    missing_operator,
                )
    results = {}

    for tree_id in df["Tree"].unique():
        if best_iteration is not None and tree_id >= (best_iteration + 1) * num_classes:
            continue # model saves all trees, even the ones that would be removed by early stopping; disregard trees not used for prediction
        tree_df = df[df["Tree"] == tree_id]
        leaf_df = tree_df[tree_df["Feature"] == 'Leaf']

        leaves_info = {}

        for _, leaf in leaf_df.iterrows():
            node_id = leaf["Node"]
            depth = 0
            conditions = []

            # Walk leaf -> root
            current = node_id
            while (tree_id, current) in children_to_parent:
                depth += 1
                _, parent_node, feature, operator, split_condition, missing_operator = children_to_parent[(tree_id, current)]
                conditions.append(
                    (feature, operator, split_condition, missing_operator)
                )
                current = parent_node

            # Canonical ordering: by feature, then operator, then threshold
            conditions = tuple(sorted(conditions))

            leaves_info[node_id] = {
                "depth": depth,
                "conditions": conditions
            }

        results[tree_id] = {
            "num_leaves": len(leaf_df),
            "leaves": leaves_info
        }

    return results


def xgboost_get_constraint_tree_number(tree_info: dict, num_rules: int, num_classes: int) -> int:
    """
    Args:
        tree_info (dict): The dictionary produced by `xgboost_extract_tree_info()`.
        num_rules (int): The number of rules to constrain the model to.
        num_classes (int): The number of prediction labels (classes).

    Returns:
        int: The iteration range id (XGBoost parameter) for the smallest number of trees, starting from the first (fitted) one, that together have `>= num_rules` leaves that induce unique regions. 
            The returned number is that for the iteration range, not the number of trees, since XGBoost converts these internally.
    """
    tree_num = 0
    unique_leaf_regions = set()
    
    for i in range(len(tree_info)):
        if i % num_classes == 0:
            tree_num += 1
            
        if tree_info[i]["num_leaves"] == 1:
            continue # 1 leaf means no split, i.e., no condition/rule    
        leaves_info = tree_info[i]["leaves"]
        for leaf_info in leaves_info.values():
            unique_leaf_regions.add(leaf_info['conditions'])
        
        # NOTE: XGBoost builds one tree per class per iteration, we could count the number of leaves divided by the number of classes, but this gives it much more capactiy
        # code would be: if len(unique_leaf_regions) / num_classes >= num_rules:
        if len(unique_leaf_regions) >= num_rules:    
            break
    return tree_num
        
        
def build_catboost_binary_feature_table(features_info) -> list[dict]:
    """
    Reconstruct CatBoost's flattened binary feature table.

    Returns:
        list[dict]: indexable by split_index
    """
    binary_features = []

    # 1. Float feature borders
    for ff in features_info.get("float_features", []):
        feature_idx = ff["feature_index"]
        flat_feature_idx = ff["flat_feature_index"]
        for border in ff.get("borders", []):
            binary_features.append({
                "type": "float",
                "float_feature_index": feature_idx,
                "flat_feature_idx": flat_feature_idx,
                "border": border,
            })

    # 2. One-hot categorical features
    for cf in features_info.get("categorical_features", []):
        if "values" in cf:
            cat_idx = cf["feature_index"]
            flat_feature_idx = cf["flat_feature_index"]
            for value in cf["values"]:
                binary_features.append({
                    "type": "one_hot",
                    "cat_feature_index": cat_idx,
                    "flat_feature_idx": flat_feature_idx,
                    "hash_value": value,
                })

    # 3. CTR binary features
    for ctr in features_info.get("ctrs", []):
        for border in ctr["borders"]:
            binary_features.append({
                "type": "ctr",
                "elements": tuple(
                    (el["cat_feature_index"], el["combination_element"])
                    for el in ctr["elements"]
                ),
                "ctr_type": ctr["ctr_type"],
                "target_border_idx": ctr["target_border_idx"],
                "prior_denomerator": ctr["prior_denomerator"],
                "prior_numerator": ctr["prior_numerator"],
                "scale": ctr["scale"],
                "shift": ctr["shift"],
                "border": border,
            })

    return binary_features


def catboost_extract_tree_info(model: catboost.CatBoostClassifier, json_path: str, num_classes: int) -> dict:
    """
    Extract per-tree leaf information from a CatBoost model.
    Args:
        model (catboost.CatBoostClassifier): The trained classifier.
        json_path (str): The path to where the model is saved in json format. Expects that this path belongs to a json file generated from `model.save_model(...)`!!!
        num_classes (int): The number of prediction labels (classes).

    Returns:
        dict: {
            tree_id: {
                "num_leaves": int,
                "leaves": {
                    leaf_id: {
                        "depth": int,
                        "conditions": tuple[(feature, op, threshold, extra)]
                    }
                }
            }
        }

    Notes:
    - Trees are oblivious
    - Categorical splits are CTR-based numeric splits
    - 'feature' for CTR splits is a canonical string describing the CTR
    """
    # tmp_path = "_cat_boost.json"
    # model.save_model(tmp_path, format='json', pool=train_pool)
    # with open(tmp_path, "r") as f:
    #     model_json = json.load(f)
    # os.remove(tmp_path)
    
    with open(json_path, "r") as f:
        model_json = json.load(f)

    trees = model_json["oblivious_trees"]
    features_info = model_json["features_info"]
    binary_features = build_catboost_binary_feature_table(features_info)
    results = {}
    num_leaves_array = model.get_tree_leaf_counts()

    for tree_id, tree in enumerate(trees):
        if model.best_iteration_ is not None and tree_id > model.best_iteration_:
            break
        splits = tree["splits"]
        depth = len(splits) if splits is not None else 0
        if depth == 0:
            pass
        num_leaves = 2 ** depth
        
        leaves_info = {}

        # Each leaf corresponds to a binary path of length = depth
        for leaf_id, decisions in enumerate(product([0, 1], repeat=depth)):
            conditions = []

            for d, decision in enumerate(decisions):
                split = splits[-d-1] # splits are inverted in the json
                bf = binary_features[split["split_index"]]
                
                if split["split_type"] == "FloatFeature":
                    assert bf['type'] == 'float'
                    assert bf['float_feature_index'] == split['float_feature_index']
                    assert bf['border'] == split['border']
                    operator = ">" if decision else "<="
                    threshold = split["border"]
                    feature = f"float_{bf['float_feature_index']}"
                
                elif split["split_type"] == "OneHotFeature":
                    assert bf['type'] == 'one_hot'
                    assert bf['cat_feature_index'] == split['cat_feature_index']
                    assert bf['hash_value'] == split["value"]
                    operator = "==" if decision else "!="
                    threshold = split["value"]
                    feature = f"cat_{bf['cat_feature_index']}"
                    
                elif split["split_type"] == "OnlineCtr":
                    assert bf['type'] == 'ctr'
                    assert bf['target_border_idx'] == split["ctr_target_border_idx"]
                    assert bf['border'] == split['border']
                    operator = ">" if decision else "<="
                    threshold = split["border"]
                    feature = (
                        "CTR("
                        f"elements={bf['elements']}, "
                        f"type={bf['ctr_type']}, "
                        f"target_bin={bf['target_border_idx']}, "
                        f"prior_denominator={bf['prior_denomerator']}, " # seems to be a typo in CatBoost's .json
                        f"prior_numerator={bf['prior_numerator']}, "
                        f"scale={bf['scale']}, "
                        f"shift={bf['shift']}"
                        ")"
                    )
                    
                else:
                    raise ValueError(f"Unexpected split_type '{split["split_type"]}'.")

                conditions.append((feature, operator, threshold))
                
            leaf_weight = tree['leaf_weights'][leaf_id]
            if leaf_weight == 0:
                assert np.all(np.array(tree['leaf_values'][leaf_id * num_classes:(leaf_id + 1) * num_classes]) == 0)
            
            # Canonical ordering (important for rule identity)
            leaves_info[leaf_id] = {
                "depth": depth,
                "leaf_weight": leaf_weight,
                "conditions": tuple(sorted(conditions))
            }

        assert num_leaves == num_leaves_array[tree_id].item() == len(leaves_info)
        results[tree_id] = {
            "num_leaves": num_leaves,
            "leaves": leaves_info
        }

    return results


def catboost_get_constraint_tree_number(
    tree_info: dict,
    num_rules: int,
) -> int:
    """
    Args:
        tree_info (dict): The dictionary produced by `cat_boost_extract_tree_info()`.
        num_rules (int): The number of rules to constrain the model to.

    Returns:
        int: The iteration range id (CatBoost parameter) for the smallest number of trees, starting from the first (fitted) one, that together have `>= num_rules` leaves that induce unique regions. 
            We do not count leaves without observations in the data, at these have leaf-value 0 and thus do not contribute in classification.
    """
    tree_num = 0
    unique_leaf_regions = set()

    for i in range(len(tree_info)):
        
        tree_num += 1
        if tree_info[i]["num_leaves"] == 1:
            continue # 1 leaf means no split, i.e., no condition/rule
        leaves_info = tree_info[i]["leaves"]

        for leaf_info in leaves_info.values():
            if leaf_info["leaf_weight"] > 0: # leaves with 0 entries do not contribute (0 weights for all classes), thus they are not rules
                unique_leaf_regions.add(leaf_info["conditions"])

        if len(unique_leaf_regions) >= num_rules:
            break

    return tree_num

