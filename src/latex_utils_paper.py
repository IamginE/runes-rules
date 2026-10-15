import numpy as np
import os
import pickle
import torch
from typing import Literal
import yaml

from data_generation import LOGGING_DIR


COLORS_METHODS = {
    'rules': 'green',
    'gt': 'darkgreen',
    'ppm_markov': 'orange',
    'transformer': 'yellow',
    'catboost': 'purple',
    'xgboost': 'red'
}
MARKERS_METHODS = {
    'rules': '*',
    'gt': 'square*',
}

RESULT_DIR = f'{LOGGING_DIR}/synthetic'
PLOT_DIR_LATEX = 'plots_and_tables_tex'

DEFAULT_SETTING = {
    'sequence_dim': 128,
    'size_rule_set': 20,
    'num_samples': 1024,
    'percentage_rules_given': 0.3,
    'frequency_rule_consequent_given': 0.4,
    'rule_confidence_given': 0.85,
}

REAL_DATASETS      = ['Holidays',           'Chorales', 'Commits', 'Rolling Mill', 'Amazon']
REAL_DATASET_NAMES = ['holidays_irregular', 'chorales', 'commits', 'rolling_mill', 'amazon_purchases']


def get_result_dicts_across_settings_synthetic(
    dir: str, 
    default_setting: dict[str, int | float],
    regular_sampling: bool,
    vary_params: tuple[str, ...],
    file_results: str = 'aggregated_results_all.yaml',
    include_baselines: bool = True,
    file_results_baselines: str = 'aggregated_results_all_ppm_trees.yaml',
) -> tuple[dict[tuple, dict], list[tuple]]:
    """    
    Args:
        dir (str): The directory where the subfolders should contain the result files for each setting.
        default_setting (dict[str, int | float]): A dictionary with keys that specify the important default params and their default values.
            There has to be exactly one setting that matches these params, and there should not be two settings that have the same params if projected onto the given ones.
        regular_sampling (bool): Whether we are considering regular sampling setting.
        vary_params (tuple[str, ...]): A tuple that specify which parameters should be varied together to compare results.
            In other words, the setting will be the same as the specified default except for the parameters that are varied.
        file_results (str, optional): The yaml file name with the aggregated results for each setting. Defaults to 'aggregated_results_all.yaml'.
        include_baselines (bool, optional): Whether to include baselines. Only works if the baseline results exist. Defaults to True.
        file_results_baselines (str, optional): The yaml file name with the aggregated results for the baselines in each setting. Defaults to 'aggregated_results_all_ppm_trees.yaml'.
        
    Returns:
        tuple[dict[tuple,dict],list[tuple]]:
            - A dictionary mapping from the tuple that identifies a setting based on `vary_params` to the result dict.
            - The ordered list of the key tuples. 
    """ 
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
        
        if include_baselines:
            baseline_file = f'{d}/{file_results_baselines}'
            if not os.path.exists(baseline_file):
                print(f"Could not find baseline result file {file_results_baselines} in {d}. Skipping...")
                continue
            with open(baseline_file, 'r') as f:
                aggregated_results_dict_baseline = yaml.safe_load(f)
                setting_to_results[key_tuple] = aggregated_results_dict_baseline | aggregated_results_dict
        
    if not found_default_setting:
        raise FileNotFoundError("No file that matches the given default parameters was found.")
    
    # ==================== Identify the Important Files when Varying the Given Parameters ====================
    settings = {}
    ordered_keys_settings = []
    # Check if two settings are the same except for the parameters that are being varied
    for k in setting_to_results.keys():
        add_k = True
        for (param, v1), v2 in zip(default_setting.items(), k):
            if param not in vary_params:
                if v1 != v2:
                    add_k = False
                    break
        if add_k:
            settings[k] = setting_to_results[k]
            ordered_keys_settings.append(k)
    ordered_keys_settings.sort() 
    return settings, ordered_keys_settings


def tuple_x_names(params_to_vary: tuple[str, ...], setting_key: tuple):
    tmp_list = []
    for (param, v1), v2 in zip(DEFAULT_SETTING.items(), setting_key):
        if param in params_to_vary:
            tmp_list.append(v2)
    if len(tmp_list) == 1:
        return tmp_list[0]
    return tuple(tmp_list) 


def generate_tikz_singleplot_bars(
    data: list[list[tuple[float, float]]],
    label_names: list[str],
    colors: list[str],
    xlabel: str,
    ylabel: str,
    xticklabels: list[str],
    enlarge_x_limit: float,
    enlarge_y_limit: float,
    width: str,
    height: str,
    bar_width: float,
    error_line_width: float,
    legend_columns: int,
    output_file: str,
    x_tick_label_style: str = None,
    y_max: int = None,
    y_ticks: list[int] = None,
):  
    """
    Args:
        data (list[list[tuple[float,float]]]): 
            Data for all label entries. Each entry is a list of `(y_value, y_err)`.
        label_names (list[str]): The label names.
        colors (list[str]): The colors for each label.
        xlabel (str): The x-axis label.
        ylabel (str): The y-axis label.
        xticklabels (list[str]): The x-tick labels.
        enlarge_x_limit (float): How far to extend the x-axis in both directions.
        enlarge_y_limit (float): How far to extend the y-axis in positive direction.
        width (str): The width of the plot.
        height (str): The height of the plot.
        bar_width (float): The width of the bars in the bar plots in pt.
        error_line_width (float): The width of the error bars in pt.
        legend_columns (int): The number of columns in the legend.
        output_file (str): The file to write the tex code to.
        x_tick_label_style (str, optional): Direct Latex string for the x tick label style.
        y_max (int, optional): Max y value.
        y_ticks (list[int], optional): Y-tick positions.
    """
    assert len(data) == len(label_names) == len(colors)
    for label_data in data:
        assert len(label_data) == len(xticklabels)

    prefix = """
\\begin{figure*}
    \\centering
    \\begin{tikzpicture}
    """

    postfix = """
    \\end{tikzpicture}
    \\caption[TODO]{
        \\textit{TODO.}
        TODO
    }
    \\label{fig:todo}
\\end{figure*}
"""

    # ======== Axis Specification ========
    axis = f"""
        \\begin{{axis}}[
            width={width},
            height={height},
            clip=false,
            axis lines=left,
            axis line style={{line width=1pt}},
            x axis line style={{-}},
            tick style={{line width=0.8pt, black}},
            tick label style={{font=\\small}},
            label style={{font=\\small}},
            ymajorgrids,
            grid style={{black!30,line width=0.5pt,dashed}},
            ybar,
            ymin=0,
            enlarge x limits={enlarge_x_limit},
            enlarge y limits={{upper={enlarge_y_limit}}},
            yticklabel style={{/pgf/number format/fixed}},
            xlabel={{{xlabel}}},
            ylabel={{{ylabel}}},
            xtick=data,
            xticklabels={{{",".join(xticklabels)}}},"""

    if x_tick_label_style is not None:
        axis += f"""
            xticklabel style={x_tick_label_style},"""

    if y_max is not None:
        axis += f"""
            ymax={y_max},"""

    if y_ticks is not None:
        y_tick_str = ",".join(map(str, y_ticks))
        axis += f"""
            ytick={{{y_tick_str}}},"""

    axis += f"""
            scaled y ticks=false,
            legend columns={legend_columns},
            legend style={{
                font=\\small,
                legend cell align=left,
                at={{(0.5,1.05)}},
                anchor=south,
                /tikz/every even column/.append style={{column sep=0.5cm}}
            }},
            legend image code/.code={{\\draw [#1] (0cm,-0.075cm) rectangle (0.2cm,0.125cm);}}
        ]
        """
    # ======== Plot Data ========
    plots_code = []
    for label_idx, label_data in enumerate(data):
        coords = " ".join(
            f"({j},{val:.4f}) +- (0,{err:.4f})" 
            for j, (val, err) in enumerate(label_data)
        )
        color = colors[label_idx]
        plot_block = f"""
            \\addplot+[
                fill={color}, draw={color}, bar width={bar_width}pt, error bars/.cd, y dir=plus, y explicit, error mark=|,
                error bar style={{draw={color},line width={error_line_width}pt}}, error mark options={{draw={color},line width={error_line_width}pt}}
            ] coordinates {{{coords}}};"""
        plots_code.append(plot_block)

    # ======== Legend ========
    legend_code = "\\legend{" + ",".join(label_names) + "}"

    full_code = (
        prefix
        + axis
        + "\n".join(plots_code)
        + "\n\n"
        + legend_code
        + "\n        \\end{axis}\n"
        + postfix
    )
    
    with open(output_file, 'w') as f:
        f.write(full_code)
        
        
def generate_tikz_singleplot_bars_no_errors_subfig(
    data: list[list[float]],
    label_names: list[str],
    colors: list[str],
    xlabel: str,
    ylabel: str,
    xticklabels: list[str],
    enlarge_x_limit: float,
    enlarge_y_limit: float,
    width: str,
    height: str,
    bar_width: float,
    legend_columns: int,
    caption: str,
    is_first: bool,
    sharedlegend: str,
    x_tick_label_style: str = None,
    y_max: int = None,
    y_ticks: list[int] = None,
) -> str:  
    """
    Args:
        data (list[list[float]]): 
            Data for all label entries. Each entry is a list of `y_value`.
        label_names (list[str]): The label names.
        colors (list[str]): The colors for each label.
        xlabel (str): The x-axis label.
        ylabel (str): The y-axis label.
        xticklabels (list[str]): The x-tick labels.
        enlarge_x_limit (float): How far to extend the x-axis in both directions.
        enlarge_y_limit (float): How far to extend the y-axis in positive direction.
        width (str): The width of the plot.
        height (str): The height of the plot.
        bar_width (float): The width of the bars in the bar plots in pt.
        legend_columns (int): The number of columns in the legend.
        caption (str): The caption of the subfigure.
        is_first (bool): Whether this is the first subfigure in the plot (for the shared legend).
        sharedlegend (str): The name of the shared legend (for the ref in LaTeX).
        x_tick_label_style (str, optional): Direct Latex string for the x tick label style.
        y_max (int, optional): Max y value.
        y_ticks (list[int], optional): Y-tick positions.
    """
    assert len(data) == len(label_names) == len(colors)
    for label_data in data:
        assert len(label_data) == len(xticklabels)

    prefix = """
\\begin{subfigure}{0.48\\textwidth}
    \\begin{tikzpicture}
    """

    postfix = f"""
    \\end{{tikzpicture}}
    \\caption{{
        {caption}
    }}
    \\label{{fig:todo}}
\\end{{subfigure}}
"""

    # ======== Axis Specification ========
    axis = f"""
        \\begin{{axis}}[
            width={width},
            height={height},
            clip=false,
            axis lines=left,
            axis line style={{line width=1pt}},
            x axis line style={{-}},
            tick style={{line width=0.8pt, black}},
            tick label style={{font=\\small}},
            label style={{font=\\small}},
            ymajorgrids,
            grid style={{black!30,line width=0.5pt,dashed}},
            ybar,
            ymin=0,
            enlarge x limits={enlarge_x_limit},
            enlarge y limits={{upper={enlarge_y_limit}}},
            yticklabel style={{/pgf/number format/fixed}},
            xlabel={{{xlabel}}},
            ylabel={{{ylabel}}},
            xtick=data,
            xticklabels={{{",".join(xticklabels)}}},"""

    if x_tick_label_style is not None:
        axis += f"""
            xticklabel style={x_tick_label_style},"""

    if y_max is not None:
        axis += f"""
            ymax={y_max},"""

    if y_ticks is not None:
        y_tick_str = ",".join(map(str, y_ticks))
        axis += f"""
            ytick={{{y_tick_str}}},"""

    axis += """
            scaled y ticks=false,"""
            
    if is_first:
        axis += f"""
            legend to name={sharedlegend},
            legend columns={legend_columns},
            legend style={{
                font=\\small,
                legend cell align=left,
                at={{(0.5,1.05)}},
                anchor=south,
                /tikz/every even column/.append style={{column sep=0.5cm}}
            }},
            legend image code/.code={{\\draw [#1] (0cm,-0.075cm) rectangle (0.2cm,0.125cm);}}
        ]
        """
    else:
        axis += """
        ]
        """
    # ======== Plot Data ========
    plots_code = []
    for label_idx, label_data in enumerate(data):
        coords = " ".join(
            f"({j},{val:.4f})" 
            for j, val in enumerate(label_data)
        )
        color = colors[label_idx]
        plot_block = f"""
            \\addplot+[
                fill={color}, draw={color}, bar width={bar_width}pt
            ] coordinates {{{coords}}};"""
        plots_code.append(plot_block)
        

    # ======== Legend ========
    if is_first:
        legend_code = "\\legend{" + ",".join(label_names) + "}"

        full_code = (
            prefix
            + axis
            + "\n".join(plots_code)
            + "\n\n"
            + legend_code
            + "\n        \\end{axis}\n"
            + postfix
        )
    else:
        full_code = (
            prefix
            + axis
            + "\n".join(plots_code)
            + "\n"
            + "\n        \\end{axis}\n"
            + postfix
        )
    
    return full_code
        
        
def generate_tikz_singleplot_lines(
    data: list[list[tuple[float, float]]],
    label_names: list[str],
    colors: list[str],
    markers: list[str],
    marker_size: int,
    xlabel: str,
    ylabel: str,
    xticklabels: list[str],
    enlarge_x_limit: float,
    enlarge_y_limit: float,
    width: str,
    height: str,
    line_width: float,
    error_line_width: float,
    legend_columns: int,
    output_file: str,
    x_tick_label_style: str = None,
    y_max: int = None,
    y_ticks: list[int] = None,
):  
    """
    Args:
        data (list[list[tuple[float,float]]]): 
            Data for all label entries. Each entry is a list of `(y_value, y_err)`.
        label_names (list[str]): The label names.
        colors (list[str]): The colors for each label.
        markers (list[str]): The line markers for each label.
        marker_size (int): The size of the markers in pt.
        xlabel (str): The x-axis label.
        ylabel (str): The y-axis label.
        xticklabels (list[str]): The x-tick labels.
        enlarge_x_limit (float): How far to extend the x-axis in both directions.
        enlarge_y_limit (float): How far to extend the y-axis in positive direction.
        width (str): The width of the plot.
        height (str): The height of the plot.
        line_width (float): The width of the lines in the plots in pt.
        error_line_width (float): The width of the error bars in pt.
        legend_columns (int): The number of columns in the legend.
        output_file (str): The file to write the tex code to.
        x_tick_label_style (str, optional): Direct Latex string for the x tick label style.
        y_max (int, optional): Max y value.
        y_ticks (list[int], optional): Y-tick positions.
    """
    assert len(data) == len(label_names) == len(colors)
    for label_data in data:
        assert len(label_data) == len(xticklabels)

    prefix = """
\\begin{figure*}
    \\centering
    \\begin{tikzpicture}
    """

    postfix = """
    \\end{tikzpicture}
    \\caption[TODO]{
        \\textit{TODO.}
        TODO
    }
    \\label{fig:todo}
\\end{figure*}
"""

    # ======== Axis Specification ========
    axis = f"""
        \\begin{{axis}}[
            width={width},
            height={height},
            clip=false,
            axis lines=left,
            axis line style={{line width=1pt}},
            tick style={{line width=0.8pt, black}},
            tick label style={{font=\\small}},
            label style={{font=\\small}},
            ymajorgrids,
            grid style={{black!30,line width=0.5pt,dashed}},
            ymin=0,
            enlarge x limits={enlarge_x_limit},
            enlarge y limits={{upper={enlarge_y_limit}}},
            yticklabel style={{/pgf/number format/fixed}},
            xlabel={{{xlabel}}},
            ylabel={{{ylabel}}},
            xtick=data,
            xticklabels={{{",".join(xticklabels)}}},"""

    if x_tick_label_style is not None:
        axis += f"""
            xticklabel style={x_tick_label_style},"""

    if y_max is not None:
        axis += f"""
            ymax={y_max},"""

    if y_ticks is not None:
        y_tick_str = ",".join(map(str, y_ticks))
        axis += f"""
            ytick={{{y_tick_str}}},"""

    axis += f"""
            scaled y ticks=false,
            legend columns={legend_columns},
            legend style={{
                font=\\small,
                legend cell align=left,
                at={{(0.5,1.05)}},
                anchor=south,
                /tikz/every even column/.append style={{column sep=0.5cm}}
            }}
        ]
        """
    # ======== Plot Data ========
    plots_code = []
    for label_idx, label_data in enumerate(data):
        coords = " ".join(
            f"({j},{val:.4f}) +- (0,{err:.4f})" 
            for j, (val, err) in enumerate(label_data)
        )
        color = colors[label_idx]
        marker = markers[label_idx]
        plot_block = f"""
            \\addplot+[
                {color}, line width={line_width}pt, mark={marker}, mark size={marker_size}pt, mark options={{fill={color}}}, error bars/.cd, y dir=both, y explicit, error mark=|,
                error bar style={{draw={color},line width={error_line_width}pt}}, error mark options={{draw={color},line width={error_line_width}pt}}
            ] coordinates {{{coords}}};"""
        plots_code.append(plot_block)

    # ======== Legend ========
    legend_code = "\n            \\legend{" + ",".join(label_names) + "}"

    full_code = (
        prefix
        + axis
        + "\n".join(plots_code)
        + "\n"
        + legend_code
        + "\n        \\end{axis}\n"
        + postfix
    )
    
    with open(output_file, 'w') as f:
        f.write(full_code)
        
        
def plot_struct_sim_synthetic(filter_settings_noise = True):
    
    for vary_params, plot_name, xlabel in zip(
        [('frequency_rule_consequent_given', 'rule_confidence_given'), ('sequence_dim'), ('num_samples')], 
        ['noise', 'vocab_size', 'training_samples'],
        ['Signal-to-Noise Ratio $(r_\\mathrm{targ}, r_\\mathrm{cons})$', 'Vocabulary Size $D$', 'Number of Training Samples $N$']
    ):
        postfix = "_full" if not filter_settings_noise and plot_name == "noise" else ""
        output_file = f'{PLOT_DIR_LATEX}/synthetic_struct_sim_{plot_name}{postfix}.tex'
        
        # changes the order of they keys
        default_setting = DEFAULT_SETTING.copy()
        # v = default_setting.pop('frequency_rule_consequent_given')
        # default_setting['frequency_rule_consequent_given'] = v
        
        params_to_res_dict, ordered_param_values = get_result_dicts_across_settings_synthetic(
            RESULT_DIR,
            default_setting,
            regular_sampling = True,
            vary_params = vary_params,
        )
        if filter_settings_noise and vary_params == ('frequency_rule_consequent_given', 'rule_confidence_given'):
            # ordered_param_values = [t for t in ordered_param_values if tuple_x_names(vary_params, t) not in {(0.1, 0.05), (1.0, 0.05), (1.0, 0.1), (1.0, 0.4), (1.0, 0.9)}]
            ordered_param_values = [t for t in ordered_param_values if tuple_x_names(vary_params, t) not in {(0.05, 0.1), (0.05, 1.0), (0.1, 1.0), (0.4, 1.0), (0.9, 1.0)}]
        
        label_names = ['$\\mathrm{sim}(\\mathbf{R},\\mathbf{R}^*)$', '$\\mathrm{sim}(\\mathbf{R}^*,\\mathbf{R})$']
        colors = ['green', 'darkgreen']
        data = []
        for k in ['sim_model_gt', 'sim_gt_model']:
            data.append([
                (params_to_res_dict[t]['rules'][k][0], params_to_res_dict[t]['rules'][k][2]) 
                for t in ordered_param_values 
            ])

        x_tick_label_style = None
        if vary_params == ('frequency_rule_consequent_given', 'rule_confidence_given'):
            x_tick_lables = [f'{tuple_x_names(vary_params, t)}' for t in ordered_param_values]
            x_tick_label_style = '{rotate=45, anchor=east, font=\\tiny}'
        else:
            x_tick_lables = [f'{int(tuple_x_names(vary_params, t))}' for t in ordered_param_values]
        y_ticks = [round(x * 0.2, 2) for x in range(6)]
        y_max = 1
                
        generate_tikz_singleplot_bars(
            data = data,
            label_names = label_names,
            colors = colors,
            xlabel = xlabel,
            ylabel = 'Similarity',
            xticklabels = x_tick_lables,
            enlarge_x_limit = 0.1,
            enlarge_y_limit = 0.1,
            width = '5cm',
            height = '4cm',
            bar_width = 2,
            error_line_width = 0.9,
            legend_columns = 2,
            output_file = output_file,
            x_tick_label_style = x_tick_label_style,
            y_max = y_max,
            y_ticks = y_ticks
        )
        
         
        print(f"Successfully wrote plotting code to {output_file}.")
        
        
def plot_acc_synthetic(filter_settings_noise = True):
    
    for vary_params, plot_name, xlabel in zip(
        [('frequency_rule_consequent_given', 'rule_confidence_given'), ('sequence_dim'), ('num_samples')], 
        ['noise', 'vocab_size', 'training_samples'],
        ['Signal-to-Noise Ratio $(r_\\mathrm{targ}, r_\\mathrm{cons})$', 'Vocabulary Size $D$', 'Number of Training Samples $N$']
    ):
        postfix = "_full" if not filter_settings_noise and plot_name == "noise" else ""
        output_file = f'{PLOT_DIR_LATEX}/synthetic_acc_{plot_name}{postfix}.tex'
        params_to_res_dict, ordered_param_values = get_result_dicts_across_settings_synthetic(
            RESULT_DIR,
            DEFAULT_SETTING,
            regular_sampling = True,
            vary_params = vary_params,
        )
        
        # filter out specific param values, because they don't reveal much
        if filter_settings_noise and vary_params == ('frequency_rule_consequent_given', 'rule_confidence_given'):
            ordered_param_values = [t for t in ordered_param_values if tuple_x_names(vary_params, t) not in {(0.05, 0.1), (0.05, 1.0), (0.1, 1.0), (0.4, 1.0), (0.9, 1.0)}]
        
        label_names = ['\\textsc{PPM}',              '\\textsc{CatBoost}',      '\\textsc{Rules}',        'Transformer',                 '\\textsc{XGBoost}',      '$\\text{\\textsc{Rules}}^*$']
        colors =      [COLORS_METHODS['ppm_markov'], COLORS_METHODS['catboost'], COLORS_METHODS['rules'], COLORS_METHODS['transformer'], COLORS_METHODS['xgboost'], COLORS_METHODS['gt']]
        data = []
        for k in ['ppm_markov_model_best_val', 'catboost', 'rules', 'transformer', 'xgboost', 'rules_gt']:
            data.append([
                (params_to_res_dict[t][k]['test_data_acc'][0], params_to_res_dict[t][k]['test_data_acc'][2])
                for t in ordered_param_values 
            ])

        x_tick_label_style = None
        if vary_params == ('frequency_rule_consequent_given', 'rule_confidence_given'):
            x_tick_lables = [f'{tuple_x_names(vary_params, t)}' for t in ordered_param_values]
            x_tick_label_style = '{rotate=45, anchor=east, font=\\tiny}'
        else:
            x_tick_lables = [f'{int(tuple_x_names(vary_params, t))}' for t in ordered_param_values]
        y_ticks = [round(x * 0.1, 2) for x in range(5)] if filter_settings_noise else [round(x * 0.2, 2) for x in range(6)]
        y_max = 0.42 if filter_settings_noise else 1
        
        generate_tikz_singleplot_bars(
            data = data,
            label_names = label_names,
            colors = colors,
            xlabel = xlabel,
            ylabel = 'Test Accuracy',
            xticklabels = x_tick_lables,
            enlarge_x_limit = 0.1,
            enlarge_y_limit = 0.1,
            width = '11cm',
            height = '4cm',
            bar_width = 2,
            error_line_width = 0.9,
            legend_columns = 3,
            output_file = output_file,
            x_tick_label_style = x_tick_label_style,
            y_max = y_max,
            y_ticks = y_ticks
        )
        
         
        print(f"Successfully wrote plotting code to {output_file}.")


def latex_table_performance_real_data(split_into_two: bool) -> str | tuple[str,str]:
    
    columns = ['rules_processed_rules', 'runes', 'ppm_markov_model', 'catboost', 'xgboost', 'transformer']
    accuracy_ranks = {
        'rules_processed_rules': [],
        'runes': [],
        'ppm_markov_model': [],
        'catboost': [],
        'xgboost': [],
        'transformer': []
    }

    if split_into_two:
        table_str = r"""\begin{table}
    \centering
    \begin{tabular}{l rr r rr rr rr r}
        \toprule
                            & \multicolumn{2}{c}{\textsc{Rules}} & \multicolumn{1}{c}{\textsc{Runes}} & \multicolumn{2}{c}{\textsc{PPM}} & \multicolumn{2}{c}{\textsc{CatBoost}} & \multicolumn{2}{c}{\textsc{XGBoost}} & Transformer \\ 
        \cmidrule(lr){2-3} \cmidrule(lr){4-4} \cmidrule(lr){5-6}  \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-11}                
        \textbf{Datasets}    & $|\mathbf{R}|$     & Acc.                & Acc.                & $|\mathbf{R}|$     & Acc.                & $|\mathbf{R}|$     & Acc.                & $|\mathbf{R}|$     & Acc.                & Acc.                 \\  
        \midrule""" + "\n"
            
        table_str_2 = r"""\begin{table}
    \centering
    \begin{tabular}{l rrr rrr rrr rrr}
        \toprule
                            & \multicolumn{3}{c}{\textsc{Rules}} & \multicolumn{3}{c}{\textsc{PPM}} & \multicolumn{3}{c}{\textsc{CatBoost}} & \multicolumn{3}{c}{\textsc{XGBoost}} \\ 
        \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}            
        \textbf{Datasets}    """ + 4 * r"& $|\mathbf{R}|$     & $\lvert A \rvert$  & Acc. $\uparrow$     " + r"\\" + r""" 
        \midrule""" + "\n"
        
    else:
        table_str = r"""\begin{table}
    \centering
    \begin{tabular}{l rrr r rrr rrr rrr r}
        \toprule
                            & \multicolumn{3}{c}{\textsc{Rules}} & \multicolumn{1}{c}{\textsc{Runes}} & \multicolumn{3}{c}{\textsc{PPM}} & \multicolumn{3}{c}{\textsc{CatBoost}} & \multicolumn{3}{c}{\textsc{XGBoost}} & Transformer \\ 
        \cmidrule(lr){2-4} \cmidrule(lr){5-5} \cmidrule(lr){6-8}  \cmidrule(lr){9-11} \cmidrule(lr){12-14} \cmidrule(lr){15-15}"""
        
        table_str += r"""                 
        \textbf{Datasets}    & $|\mathbf{R}|$     & $\lvert A \rvert$  & Acc. $\uparrow$     & Acc. $\uparrow$     & $|\mathbf{R}|$     & $\lvert A \rvert$  & Acc. $\uparrow$     & $|\mathbf{R}|$     & $\lvert A \rvert$  & Acc. $\uparrow$     & $|\mathbf{R}|$     & $\lvert A \rvert$  & Acc. $\uparrow$     & Acc. $\uparrow$      \\  
        \midrule""" + "\n"
    

    for dataset, name in zip(REAL_DATASETS, REAL_DATASET_NAMES):
        
        with open(f'{LOGGING_DIR}/real/{name}/ppm_markov_model_aggregated_results.yaml', 'r') as f:
            result_logs_ppm = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/catboost_aggregated_results.yaml', 'r') as f:
            result_logs_catboost = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/xgboost_aggregated_results.yaml', 'r') as f:
            result_logs_xgboost = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/aggregated_results.yaml', 'r') as f:
            result_logs = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/rules_processed_rules_aggregated_results.yaml', 'r') as f:
            result_logs_processed_rules = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/runes_processed_rules_aggregated_results.yaml', 'r') as f:
            result_logs_residual_network = yaml.safe_load(f)
        
        with open(f'{LOGGING_DIR}/real/{name}/interpretability_results.yaml', 'r') as f:
            result_logs_interpretability = yaml.safe_load(f)
        
        accuracies = {}
        num_rules = {}
        antecedent_sizes = {}
        for key, logs in zip(columns, [
            result_logs_processed_rules,
            result_logs_residual_network,
            result_logs_ppm,
            result_logs_catboost,
            result_logs_xgboost,
            result_logs,
        ]):
            if key == 'ppm_markov_model':
                # get the order of the best model
                best_order = logs['model_list'][0][0]
                key_acc = f'{key}_{best_order}'
            else:
                key_acc = key
            test_acc = round(logs[key_acc]['test_data_acc'] * 100, 2)
            accuracies[key] = test_acc
            
            if key not in ['runes', 'transformer']:
                constraint = 'constraint' if key == 'rules_processed_rules' else 'unconstraint'
                num_rules[key] = result_logs_interpretability[key][constraint]['num_rules']
                mean_antecedent_size = result_logs_interpretability[key][constraint]['mean_rule_size']
                std_antecedent_size = result_logs_interpretability[key][constraint]['std_rule_size']
                antecedent_sizes[key] = (mean_antecedent_size, std_antecedent_size)
        
        for k1, acc in accuracies.items():
            num_bigger = 0
            for k2, acc_comp in accuracies.items():
                if k1 != k2 and acc < acc_comp:
                    num_bigger += 1
            accuracy_ranks[k1].append(num_bigger + 1)
             
        
        table_str += f"        {dataset:<21}"
        acc_list = [acc for k, acc in accuracies.items()]
        top_acc = max(acc_list)
        acc_list.remove(top_acc) # only removes best acc once
        top_2_acc = max(acc_list)
        
        for k in columns:        
            if k not in ['runes', 'transformer']:
                table_str += f"&{num_rules[k]:>18d}  "
                # table_str += f"&{antecedent_sizes[k][0]:>9.1f} $\\pm$ {antecedent_sizes[k][1]:>3.1f} "
                if not split_into_two:
                    table_str += f"&{antecedent_sizes[k][0]:>18.1f}  "
            acc = accuracies[k]
            if acc == top_acc:
                table_str += f"&      \\textbf{{{acc:>5.2f}}} "
            elif acc == top_2_acc:
                table_str += f"&   \\underline{{{acc:>5.2f}}} "
            else:
                table_str += f"&{acc:>19.2f}  "
                
        table_str += r" \\" + "\n"
        
        if split_into_two: 
            table_str_2 += f"        {dataset:<21}"
            for k in columns:
                if k in ['runes', 'transformer']: 
                    continue    

                table_str_2 += f"&{num_rules[k]:>18d}  "
                table_str_2 += f"&{antecedent_sizes[k][0]:>18.1f}  "
                acc = accuracies[k]
                table_str_2 += f"&{acc:>19.2f}  "
            table_str_2 += r" \\" + "\n"
    
    table_str += r"        \midrule" + "\n"    
    table_str += f"        {'Avg. Rank':<21}"
    for k in columns:
        mean_rank = sum(accuracy_ranks[k]) / len(accuracy_ranks[k])
        if k == 'runes' or k == 'transformer':
            table_str += f"&{mean_rank:>19.1f}  "
        else:
            factor = 1 if split_into_two else 2
            table_str += factor*f"&{'':>18s}  " + f"&{mean_rank:>19.1f}  "
    table_str += r" \\" + "\n"
    
    table_str += r"        \bottomrule" + "\n"
    table_str += r"    \end{tabular}" + "\n"
    table_str += r"""
    \caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{tab:todo}
\end{table}
""" 
    if split_into_two: 
        table_str_2 += r"        \bottomrule" + "\n"
        table_str_2 += r"    \end{tabular}" + "\n"
        table_str_2 += r"""
    \caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{tab:todo}
\end{table} 
"""      
        return table_str, table_str_2
    else:
        return table_str


def latex_table_residual_network_confusion_real_data() -> tuple[str, ...]:
    tables_datasets_test = []
    aggregated_row_1 = np.zeros(4)
    aggregated_row_2 = np.zeros(4)
    aggregated_fraction_data = np.zeros(2)
    count_datasets = 0
    
    for dataset, name in zip(REAL_DATASETS, REAL_DATASET_NAMES):
    # for dataset, name in zip(REAL_DATASETS, REAL_DATASET_NAMES):
        if name in ['commits']:
            continue
        
        prefix = 'test'
        table_changes_pred = r"""\begin{subtable}{0.48\textwidth}
    \centering
    \begin{tabular}{l rr rr}
        \toprule
                         & \multicolumn{4}{c}{\textsc{Rules}} \\ 
        \cmidrule(lr){2-5}
        \textsc{Runes}   & \multicolumn{2}{c}{Rule Active} & \multicolumn{2}{c}{No Rule Active}\\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5}
                         & right & wrong & right & wrong  \\
        \midrule
"""       
            
        with open(f'{LOGGING_DIR}/real/{name}/runes_processed_rules_aggregated_results.yaml', 'r') as f:
            result_logs = yaml.safe_load(f)
            
        both_models_correct_active = result_logs['runes'][f'{prefix}_both_models_correct_active']
        both_models_correct_inactive = result_logs['runes'][f'{prefix}_both_models_correct_inactive']
        both_models_incorrect_active = result_logs['runes'][f'{prefix}_both_models_incorrect_active']
        both_models_incorrect_inactive = result_logs['runes'][f'{prefix}_both_models_incorrect_inactive']
        
        correct_change_active = result_logs['runes'][f'{prefix}_correct_change_active']
        correct_change_inactive = result_logs['runes'][f'{prefix}_correct_change_inactive']
        incorrect_change_active = result_logs['runes'][f'{prefix}_incorrect_change_active']
        incorrect_change_inactive = result_logs['runes'][f'{prefix}_incorrect_change_inactive']
            
        count_active = both_models_correct_active + both_models_incorrect_active + correct_change_active + incorrect_change_active
        count_inactive = both_models_correct_inactive + both_models_incorrect_inactive + correct_change_inactive + incorrect_change_inactive
        total_count = count_active + count_inactive

        both_models_correct_active /= count_active
        both_models_incorrect_active /= count_active
        correct_change_active /= count_active
        incorrect_change_active /= count_active
        
        both_models_correct_inactive /= count_inactive
        both_models_incorrect_inactive /= count_inactive
        correct_change_inactive /= count_inactive
        incorrect_change_inactive /= count_inactive
            
        row1 = [both_models_correct_active, correct_change_active, both_models_correct_inactive, correct_change_inactive]
        row2 = [incorrect_change_active, both_models_incorrect_active, incorrect_change_inactive, both_models_incorrect_inactive]
        
        count_datasets += 1
        aggregated_row_1 += np.array(row1)
        aggregated_row_2 += np.array(row2)
        aggregated_fraction_data += np.array([count_active/total_count, count_inactive/total_count])
    
        rows = f"        right            "
        for entry in row1:
            rows += f"&{entry:>6.2f} "
        rows += r" \\" + "\n"
        
        rows += f"        wrong            "
        for entry in row2:
            rows += f"&{entry:>6.2f} "
        rows += r" \\" + "\n"

        rows += rf"""        \midrule
        Fraction of Data & \multicolumn{{2}}{{c}}{{{count_active / total_count:.2f}}} & \multicolumn{{2}}{{c}}{{{count_inactive / total_count:.2f}}} \\ \cmidrule(lr){{2-5}}       
        Total Count      & \multicolumn{{4}}{{c}}{{{int(total_count):d}}} \\""" + "\n"
        rows += r"        \bottomrule" + "\n"
        rows += r"    \end{tabular}" + "\n"
        rows += f"""
\\caption{{
    {dataset}
}}
\\label{{tab:todo}}
\\end{{subtable}}
"""
        
        table_changes_pred += rows
        tables_datasets_test.append(table_changes_pred)
                    
    combined_table_datasets = r"""\begin{table}
\centering
"""
    combined_table_datasets += "\\hfill \n".join(tables_datasets_test)
    combined_table_datasets += f"""
    \\caption[TODO]{{
        TODO
    }}
    \\label{{tab:todo}}
\\end{{table}}
"""

    # AGGREGATED TABLE
    aggregated_row_1 /= count_datasets
    aggregated_row_2 /= count_datasets
    aggregated_fraction_data /= count_datasets

    aggregated_table = r"""\begin{table}
    \centering
    \begin{tabular}{l rr rr}
        \toprule
                         & \multicolumn{4}{c}{\textsc{Rules}} \\ 
        \cmidrule(lr){2-5}
        \textsc{Runes}   & \multicolumn{2}{c}{Rule Active} & \multicolumn{2}{c}{No Rule Active}\\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5}
                         & right & wrong & right & wrong  \\
        \midrule
"""
    aggregated_table += f"        right            "
    for entry in aggregated_row_1:
        aggregated_table += f"&{entry.item():>6.2f} "
    aggregated_table += r" \\" + "\n"
    
    aggregated_table += f"        wrong            "
    for entry in aggregated_row_2:
        aggregated_table += f"&{entry.item():>6.2f} "
    aggregated_table += r" \\" + "\n"

    aggregated_table += rf"""        \midrule
        Fraction of Data & \multicolumn{{2}}{{c}}{{{aggregated_fraction_data[0].item():.2f}}} & \multicolumn{{2}}{{c}}{{{aggregated_fraction_data[1].item():.2f}}} \\ """ + "\n"
    aggregated_table += r"        \bottomrule" + "\n"
    aggregated_table += r"    \end{tabular}" + "\n"
    aggregated_table += """
\\caption{
    TODO
}
\\label{tab:todo}
\\end{table}
"""
    
    return aggregated_table, combined_table_datasets    


def table_real_dataset_stats():
    
    table = r"""\begin{table}
    \centering
    \small
    \begin{tabular}{l rrrrr c}
        \toprule
        \textbf{Datasets}   & $N$    & $L_\mathrm{tot}$ & $L_\mathrm{min}$ & $L_\mathrm{max}$ & $D$    & Regular \\
        \midrule  
"""
    for dataset, name in zip(REAL_DATASETS, REAL_DATASET_NAMES):
        metadata_train_val = torch.load(f'data/real/{name}/train_val.pt')['metadata']
        metadata_test = torch.load(f'data/real/{name}/test.pt')['metadata']
        
        assert metadata_train_val['sequence_dim'] == metadata_test['sequence_dim']
        d = metadata_train_val['sequence_dim']
        if dataset in ['Holidays', 'Commits']:
            # they only have one long sequence that is split across train and test set
            n = 1
        else:
            n = metadata_train_val['num_samples'] + metadata_test['num_samples']
        e = metadata_train_val['t_all'] + metadata_test['t_all']
        reg = r'$\checkmark$' if metadata_train_val['regular_sampling'] else r'\ding{55}'
        
        if dataset in ['Holidays', 'Commits']:
            table += f"        {dataset:<19} &{n:7d} &{e:>16d} &{'--':>17} &{'--':>17} &{d:>7d} & {reg:<15} \\\\ \n"
        else:
            min_length = metadata_train_val['min_sequence_length']
            max_length = metadata_train_val['max_sequence_length']
            table += f"        {dataset:<19} &{n:7d} &{e:>16d} &{min_length:>17d} &{max_length:>17d} &{d:>7d} & {reg:<15} \\\\ \n"
    
    table += r"        \bottomrule" + "\n"
    table += r"    \end{tabular}"
    table += r"""
    \caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{tab:todo}
"""
    table += r"\end{table}" + "\n"
    return table       



def latex_table_rule_processing_real() -> str:

    table_str = r"""\begin{table}
    \centering
    \begin{tabular}{l rrr rrr rrr}
        \toprule
                             & \multicolumn{3}{c}{Before} & \multicolumn{3}{c}{After} & \multicolumn{3}{c}{Difference} \\ 
        \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}"""
    
    table_str += r"""                 
        \textbf{Datasets}    & Acc.         & Avg. $\lvert A\rvert$     & Avg. $\lvert\Delta_i\rvert$ & Acc.         & Avg. $\lvert A\rvert$     & Avg. $\lvert\Delta_i\rvert$ & Acc.              & Avg. $\lvert A\rvert$ & Avg. $\lvert\Delta_i\rvert$ \\
        \midrule""" + "\n"
    

    for dataset, name in zip(REAL_DATASETS, REAL_DATASET_NAMES):
        
        
        with open(f'{LOGGING_DIR}/real/{name}/aggregated_results.yaml', 'r') as f:
            results_before_processing = yaml.safe_load(f)
        with open(f'{LOGGING_DIR}/real/{name}/rules_processed_rules_aggregated_results.yaml', 'r') as f:
            results_after_processing = yaml.safe_load(f)
        with open(f'{LOGGING_DIR}/real/{name}/rule_processing_stats.yaml', 'r') as f:
            rule_processing_stats = yaml.safe_load(f)
            
        
        entries_before = []
        entries_after  = []
        diffs = []
        
        for is_before in [True, False]:
            if is_before:
                results = results_before_processing
                key = 'before_processing'
                key_model = 'rules'
                entries = entries_before
            else:
                results = results_after_processing
                key = 'after_processing'
                key_model = 'rules_processed_rules'
                entries = entries_after
            
            test_acc = round(results[key_model]['test_data_acc'] * 100, 2)
            entries.append(test_acc)
  
            mean_ant_size = rule_processing_stats[key]['ant_size_mean']
            std_ant_size  = rule_processing_stats[key]['ant_size_std']
            entries.append((mean_ant_size, std_ant_size))
            
            mean_interval_width = rule_processing_stats[key]['interval_width_mean']
            std_interval_width  = rule_processing_stats[key]['interval_width_std']
            entries.append((mean_interval_width, std_interval_width))
            
       
        for e1, e2 in zip(entries_before, entries_after):
            if isinstance(e1, tuple) and isinstance(e2, tuple):
               diffs.append(e2[0] - e1[0])
            else:
                diffs.append(e2 - e1)
    
        table_str += f"        {dataset:<21}"
        
        for count_l, l in enumerate([entries_before, entries_after, diffs]):
            for i in range(len(l)):
                if isinstance(l[i], tuple):
                    table_str += f"&{l[i][0]:>10.1f} $\\pm$ {l[i][1]:>5.1f} "
                else:
                    if count_l == 2:
                        table_str += f"&{l[i]:>+18.2f} "
                    else:
                        table_str += f"&{l[i]:>13.2f} "
        table_str += r" \\" + "\n"
    table_str += r"        \bottomrule" + "\n"
    table_str += r"    \end{tabular}" + "\n"
    table_str += r"""
    \caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{tab:todo}
\end{table}
"""
    return table_str 



def plot_activation_histograms_real():
    
    output_file = f'{PLOT_DIR_LATEX}/activation_histograms.tex'
    sharedlegend = 'sharedlegend_2'
    
    plot_str = f"""\\begin{{figure}}[hb]
\\centering

\\ref{{{sharedlegend}}}
\\vspace{{0.2cm}}
    """
    
    
    for num_dataset, (dataset, name) in enumerate(zip(REAL_DATASETS, REAL_DATASET_NAMES)):
        
        with open(f'{LOGGING_DIR}/real/{name}/rule_processing_stats.yaml', 'r') as f:
            rule_processing_stats = yaml.safe_load(f)
            
        hist_before = rule_processing_stats['before_processing']['test_act_histogram']
        hist_after  = rule_processing_stats['after_processing']['test_act_histogram']
        max_x = None
        max_y = 0
        for i in range(len(hist_before)):
            if hist_before[i] > 0:
                max_x = i
            max_y = max(max_y, int(hist_before[i]))
        
        for i in range(len(hist_after)):
            if hist_after[i] > 0 and i > max_x:
                max_x = i
            max_y = max(max_y, int(hist_after[i]))

        data = [hist_before[:max_x + 1], hist_after[:max_x + 1]]
        
        subfigure = generate_tikz_singleplot_bars_no_errors_subfig(
            data = data,
            label_names = ['Before Processing', 'After Processing'],
            colors = ['green', 'darkgreen'],
            xlabel = 'Active Rules',
            ylabel = 'Number of Events',
            xticklabels = list(map(str, range(max_x + 1))),
            enlarge_x_limit = 0.1,
            enlarge_y_limit = 0.1,
            width = '8cm',
            height = '5cm',
            bar_width = 2,
            legend_columns = 2,
            y_max = max_y,
            y_ticks = [i * (max_y // 4) for i in range(5)],
            caption = dataset,
            is_first = num_dataset == 0,
            sharedlegend = sharedlegend
        )
        
        plot_str += (
            "\n"
            + subfigure
            + "\n"
        )
        
    
    plot_str += r"""\caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{fig:todo}
\end{figure}"""

    with open(output_file, 'w') as f:
        f.write(plot_str)
        
        
def latex_table_prediction_performance_real_data(kind: Literal['loss', 'acc']) -> str:
    
    columns = ['rules_processed_rules', 'runes', 'ppm_markov_model', 'catboost', 'xgboost', 'transformer']
    ranks = {
        'rules_processed_rules': [],
        'runes': [],
        'ppm_markov_model': [],
        'catboost': [],
        'xgboost': [],
        'transformer': []
    }

    table_str = r"""\begin{table}
    \centering
    \begin{tabular}{l rrr r rrr rrr rrr r}
        \toprule
        \textbf{Datasets}    & \multicolumn{1}{c}{\textsc{Rules}} & \multicolumn{1}{c}{\textsc{Runes}} & \multicolumn{1}{c}{\textsc{PPM}} & \multicolumn{1}{c}{\textsc{CatBoost}} & \multicolumn{1}{c}{\textsc{XGBoost}} & Transformer \\
        \midrule"""
    table_str += "\n"
    
    for dataset, name in zip(REAL_DATASETS, REAL_DATASET_NAMES):
        
        with open(f'{LOGGING_DIR}/real/{name}/ppm_markov_model_aggregated_results.yaml', 'r') as f:
            result_logs_ppm = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/catboost_aggregated_results.yaml', 'r') as f:
            result_logs_catboost = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/xgboost_aggregated_results.yaml', 'r') as f:
            result_logs_xgboost = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/aggregated_results.yaml', 'r') as f:
            result_logs = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/rules_processed_rules_aggregated_results.yaml', 'r') as f:
            result_logs_processed_rules = yaml.safe_load(f)
            
        with open(f'{LOGGING_DIR}/real/{name}/runes_processed_rules_aggregated_results.yaml', 'r') as f:
            result_logs_residual_network = yaml.safe_load(f)
        
        results = {}
        for key, logs in zip(columns, [
            result_logs_processed_rules,
            result_logs_residual_network,
            result_logs_ppm,
            result_logs_catboost,
            result_logs_xgboost,
            result_logs,
        ]):
            if key == 'ppm_markov_model':
                # get the order of the best model
                best_order = logs['model_list'][0][0]
                key_res = f'{key}_{best_order}'
            else:
                key_res = key
            
            if kind == 'acc':
                test_metric = round(logs[key_res]['test_data_acc'] * 100, 2)
            elif kind == 'loss':
                test_metric = round(logs[key_res]['test_data_loss'], 3)
            results[key] = test_metric
        
        for k1, res in results.items():
            num_better = 0
            for k2, res_comp in results.items():
                if kind =='acc' and k1 != k2 and res < res_comp:
                    num_better += 1
                elif kind =='loss' and k1 != k2 and res > res_comp:
                    num_better += 1
            ranks[k1].append(num_better + 1)
             
        
        table_str += f"        {dataset:<21}"
        res_list = [res for k, res in results.items()]
        top_res = max(res_list) if kind == 'acc' else min(res_list)
        res_list.remove(top_res) # only removes best result once
        top_2_res = max(res_list) if kind == 'acc' else min(res_list)
        
        for k in columns:        
            res = results[k]
            if res == top_res:
                table_str += f"&      \\textbf{{{res:>.2f}}} " if kind == 'acc' else f"&      \\textbf{{{res:>.3f}}} "
            elif res == top_2_res:
                table_str += f"&   \\underline{{{res:>.2f}}} " if kind == 'acc' else f"&   \\underline{{{res:>.3f}}} " 
            else:
                table_str += f"&{res:>19.2f}  " if kind == 'acc' else f"&{res:>19.3f}  "
                
        table_str += r" \\" + "\n"
    
    table_str += r"        \midrule" + "\n"    
    table_str += f"        {'Avg. Rank':<21}"
    for k in columns:
        mean_rank = sum(ranks[k]) / len(ranks[k])
        table_str += f"&{mean_rank:>19.1f}  "
    table_str += r" \\" + "\n"
    
    table_str += r"        \bottomrule" + "\n"
    table_str += r"    \end{tabular}" + "\n"
    table_str += r"""
    \caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{tab:todo}
\end{table}
"""
    return table_str

        
        
def mean_and_std_table(split_into_two: bool) -> str | tuple[str,str]:
    columns = ['rules',                'rules_processed_rules', 'runes_processed_rules', 'transformer']
    
    if split_into_two:
        table_str = r"""\begin{table}
    \centering
    \begin{tabular}{l r r r r}
        \toprule
                            & \multicolumn{1}{c}{\textsc{Rules} (no refinement)} & \multicolumn{1}{c}{\textsc{Rules}} & \multicolumn{1}{c}{\textsc{Runes}} & \multicolumn{1}{c}{Transformer} \\ 
        \midrule """ + "\n"
        table_str_2 = table_str
    
    else:
        table_str = r"""\begin{table}
    \centering
    \begin{tabular}{l rr rr rr rr}
        \toprule
                            & \multicolumn{2}{c}{\textsc{Rules} (no refinement)} & \multicolumn{2}{c}{\textsc{Rules}} & \multicolumn{2}{c}{\textsc{Runes}} & \multicolumn{2}{c}{Transformer} \\ 
        \cmidrule(lr){2-3} \cmidrule(lr){4-5}  \cmidrule(lr){6-7} \cmidrule(lr){8-9} """
            
        # table_str += r"""                 
        #     \textbf{Datasets}    """ + 4*r"& NLL $\downarrow$    & Acc. $\uparrow$     " + r"""\\  
        #     \midrule""" + "\n"
        table_str += r"""                 
        \textbf{Datasets}    """ + 4*r"& \multicolumn{1}{c}{NLL $\downarrow$} & \multicolumn{1}{c}{Acc. $\uparrow$}     " + r"""\\  
        \midrule""" + "\n"
        
    for dataset, name in zip(REAL_DATASETS, REAL_DATASET_NAMES):
        table_str += f"        {dataset:<21}"
        if split_into_two:
            table_str_2 += f"        {dataset:<21}"
        with open(f'./{LOGGING_DIR}/real/{name}/results_mean_std.yaml', 'r') as f:
            res = yaml.safe_load(f)
        for key in columns:
            table_str += f'& {res[key]['test_data_loss']['mean']:>6.3f} $\\pm$ {res[key]['test_data_loss']['std']:>6.3f} '
            if split_into_two:
                table_str_2 += f'& {res[key]['test_data_acc']['mean']*100:>6.2f} $\\pm$ {res[key]['test_data_acc']['std']*100:>6.2f} '
            else:
                table_str += f'& {res[key]['test_data_acc']['mean']*100:>6.2f} $\\pm$ {res[key]['test_data_acc']['std']*100:>6.2f} '
        table_str += r"\\" + "\n"
        if split_into_two:
            table_str_2 += r"\\" + "\n"
    
    table_str += r"        \bottomrule" + "\n"
    table_str += r"    \end{tabular}" + "\n"
    table_str += r"""
    \caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{tab:todo}
\end{table}
"""
    if split_into_two:
        table_str_2 += r"        \bottomrule" + "\n"
        table_str_2 += r"    \end{tabular}" + "\n"
        table_str_2 += r"""
    \caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{tab:todo}
\end{table}
"""
        return table_str, table_str_2
    else:
        return table_str


def latex_table_independent_training() -> str:
    columns = ['runes_processed_rules', 'runes_ablation']
    table_str = r"""\begin{table}
    \centering
    \begin{tabular}{l rr rr}
        \toprule
                             & \multicolumn{2}{c}{\textsc{Runes}} & \multicolumn{2}{c}{\textsc{Runes} (independent)} \\ 
        \cmidrule(lr){2-3} \cmidrule(lr){4-5} """
        
    table_str += r"""                 
        \textbf{Datasets}    """ + 2*r"& \multicolumn{1}{c}{NLL $\downarrow$} & \multicolumn{1}{c}{Acc. $\uparrow$}     " + r"""\\  
        \midrule""" + "\n"
        
    for dataset, name in zip(REAL_DATASETS, REAL_DATASET_NAMES):
        table_str += f"        {dataset:<21}"
        with open(f'./{LOGGING_DIR}/real/{name}/results_mean_std.yaml', 'r') as f:
            res = yaml.safe_load(f)
        for key in columns:
            table_str += f'& {res[key]['test_data_loss']['mean']:>6.3f} $\\pm$ {res[key]['test_data_loss']['std']:>6.3f} '
            table_str += f'& {res[key]['test_data_acc']['mean']*100:>6.2f} $\\pm$ {res[key]['test_data_acc']['std']*100:>6.2f} '
        table_str += r"\\" + "\n"
    
    table_str += r"        \bottomrule" + "\n"
    table_str += r"    \end{tabular}" + "\n"
    table_str += r"""
    \caption[TODO]{
        \textit{TODO.}
        TODO
    }
    \label{tab:todo}
\end{table}
"""
    return table_str
        

def plot_kl_vs_agreement_binned(dataset: str, metric: str, num_bins: int = 10, normalize_counts_within_bin = True):
    
    with open(f'./{LOGGING_DIR}/real/{dataset}/residual_network_behavior_individual.pkl', 'rb') as f:
        d = pickle.load(f)

    data_split = 'test'
    active_str = 'active'
    metric_name_map = {
        'entropy_rules': 'Entropy \\textsc{Rules}',
        'symmetric_kl': 'Symmetric KL, \\textsc{Rules} vs. \\textsc{Runes}'
    }
    
    cases = [k for k in d.keys() if k.startswith(f'{data_split}_{active_str}')]

    # compute global bins 
    all_values = np.concatenate([d[c][metric] for c in cases])
    bin_edges = np.linspace(np.min(all_values), np.max(all_values), num_bins + 1)

    # histogram counts per case
    hist_counts = {}
    if normalize_counts_within_bin:
        overall_counts = np.zeros(num_bins)
    for case in cases:
        values = d[case][metric]
        counts, _ = np.histogram(values, bins=bin_edges)
        hist_counts[case] = counts
        if normalize_counts_within_bin:
            overall_counts += counts
        
    if normalize_counts_within_bin:
        overall_counts[overall_counts == 0] = 1.0
        for case in cases:
            hist_counts[case] = hist_counts[case].astype(float) / overall_counts
        
    label_names = ['Both Correct', 'Correct Change', 'Incorrect Change', '{Both Incorrect, Agree}', '{Both Incorrect, Disagree}']
    colors =      ['darkgreen'   , 'green'         , 'red'             , 'blue'                   , 'orange'                    ]  
    data = []
    for case in [
        f'{data_split}_{active_str}_both_correct',
        f'{data_split}_{active_str}_correct_change',
        f'{data_split}_{active_str}_incorrect_change',
        f'{data_split}_{active_str}_both_agree_incorrect',
        f'{data_split}_{active_str}_both_disagree_incorrect', 
    ]: # ensure correct sorting
        data.append(list(map(lambda x: (x, 0), hist_counts[case].tolist())))

    x_tick_label_style ='{rotate=45, anchor=east, font=\\footnotesize}'
    x_tick_lables = [f"{{[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]}}" for i in range(len(bin_edges) - 1)]
    y_ticks = [round(x * 0.2, 2) for x in range(6)] if normalize_counts_within_bin else None
    y_max = 1 if normalize_counts_within_bin else None
    
    ylabel = "Interval Prob." if normalize_counts_within_bin else "Count"
    xlabel = metric_name_map[metric]
    output_file = f'{PLOT_DIR_LATEX}/{dataset}_{metric}_hist_hybrid.tex'
        
    generate_tikz_singleplot_bars(
        data = data,
        label_names = label_names,
        colors = colors,
        xlabel = xlabel,
        ylabel = ylabel,
        xticklabels = x_tick_lables,
        enlarge_x_limit = 0.1,
        enlarge_y_limit = 0.1,
        width = '11cm',
        height = '4cm',
        bar_width = 2,
        error_line_width = 0.9,
        legend_columns = 3,
        output_file = output_file,
        x_tick_label_style = x_tick_label_style,
        y_max = y_max,
        y_ticks = y_ticks
    )
    
        
    print(f"Successfully wrote plotting code to {output_file}.")
 
        
if __name__ == "__main__":
    if not os.path.exists(PLOT_DIR_LATEX):
        os.makedirs(PLOT_DIR_LATEX)
    plot_struct_sim_synthetic(True)
    plot_struct_sim_synthetic(False)
    plot_acc_synthetic(True)
    plot_acc_synthetic(False)
    
    with open(f'{PLOT_DIR_LATEX}/table_results_real.tex', 'w') as f:
        t1, t2 = latex_table_performance_real_data(split_into_two=True)
        f.write(t1 + "\n\n\n" + t2)
    
    t1, t2 = latex_table_residual_network_confusion_real_data()
    with open(f'{PLOT_DIR_LATEX}/confusion_residual_network_accumulated.tex', 'w') as f:
        f.write(t1)
    with open(f'{PLOT_DIR_LATEX}/confusion_residual_network_individual.tex', 'w') as f:
        f.write(t2)
    
    with open(f'{PLOT_DIR_LATEX}/table_dataset_stats.tex', 'w') as f:
        f.write(table_real_dataset_stats())
        
    with open(f'{PLOT_DIR_LATEX}/rule_processing_stats.tex', 'w') as f:
        f.write(latex_table_rule_processing_real())
    plot_activation_histograms_real()
    
    with open(f'{PLOT_DIR_LATEX}/loss_real_data.tex', 'w') as f:
        f.write(latex_table_prediction_performance_real_data('loss'))
    
    with open(f'{PLOT_DIR_LATEX}/mean_and_std_results.tex', 'w') as f:
        t1, t2 = mean_and_std_table(split_into_two=True)
        f.write(t1 + "\n\n\n" + t2)
        
    plot_kl_vs_agreement_binned('rolling_mill', 'symmetric_kl', num_bins=8)
        
    with open(f'{PLOT_DIR_LATEX}/independent_training.tex', 'w') as f:
        f.write(latex_table_independent_training())
    