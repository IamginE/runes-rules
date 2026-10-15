from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import yaml

from data_generation import (
    LOGGING_DIR,
)


def plot_runtimes():
    def plot_data(data):
        fig = plt.figure(figsize=(7, 5), dpi=300)
        for model, dim_dict in data.items():
            for dim, seq_dict in dim_dict.items():
                seqs = sorted(seq_dict.keys())
                means = [seq_dict[s][0] for s in seqs]
                stds = [seq_dict[s][1] for s in seqs]

                plt.errorbar(
                    seqs,
                    means,
                    yerr = stds,
                    label = f"{model}, $D={dim}$",
                    color = colors[dim],
                    linestyle = linestyles[model],
                    marker = markers[model],
                    capsize = 3
                )

        plt.xlabel("Sequence Length")
        plt.ylabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 1.8,
        "lines.markersize": 6.5,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "pdf.fonttype": 42,
        "text.usetex": True,
        "font.family": "serif",
    })

    color_list = [
        (0.68,  0.36,  0.51),   # purple
        (0.0,   0.447, 0.698),  # blue
        (0.96,  0.624, 0.0),    # orange
        (0.55,  0.71,  0.0),    # green
    ]

    with open(f"./{LOGGING_DIR}/runtime/wall_clock_times.yaml", "r") as f:
        data = yaml.safe_load(f)
    train_data = defaultdict(lambda: defaultdict(dict))
    infer_data = defaultdict(lambda: defaultdict(dict))

    for seq_len, dim_dict in data.items():
        for dim, models_and_mode in dim_dict.items():
            
            # Training (step time)
            if "rules_training" in models_and_mode:
                mean, std = models_and_mode["rules_training"]["step"]
                train_data["RULES"][dim][seq_len] = (mean, std)

            if "transformer_training" in models_and_mode:
                mean, std = models_and_mode["transformer_training"]["step"]
                train_data["Transformer"][dim][seq_len] = (mean, std)

            # Inference (forward time)
            if "rules_inference" in models_and_mode:
                mean, std = models_and_mode["rules_inference"]["forward"]
                infer_data["RULES"][dim][seq_len] = (mean, std)

            if "rules_inference_symbolic" in models_and_mode:
                mean, std = models_and_mode["rules_inference_symbolic"]["forward"]
                infer_data["RULES (S)"][dim][seq_len] = (mean, std)

            if "transformer_inference" in models_and_mode:
                mean, std = models_and_mode["transformer_inference"]["forward"]
                infer_data["Transformer"][dim][seq_len] = (mean, std)

    linestyles = {
        "RULES": "-",
        "RULES (S)": "--",
        "Transformer": ":"
    }

    markers = {
        "RULES": "o",
        "RULES (S)": "^",
        "Transformer": "s"
    }

    # Assign colors per dimensionality
    dims = sorted({dim for m in train_data.values() for dim in m.keys()})
    colors = {dim: color_list[i % len(color_list)] for i, dim in enumerate(dims)}

    # Create plots
    plot_data(train_data)
    plt.savefig(f"./{LOGGING_DIR}/runtime/figure_time_training.pdf", bbox_inches="tight")
    plt.close()
    plot_data(infer_data)
    plt.savefig(f"./{LOGGING_DIR}/runtime/figure_time_inference.pdf", bbox_inches="tight")
    plt.close()
    

def plot_entropy_vs_kl_residual_network(dataset: str):

    def case_label_and_marker_map(case_str: str) -> tuple[str,str]:
        stripped_str = '_'.join(case_str.split('_')[2:])
        label_str = None
        marker = None
        if stripped_str == 'both_correct':
            label_str = 'Both Correct'
            marker = 'o'
        elif stripped_str == 'both_agree_incorrect':
            label_str = 'Both Incorrect, Agree'
            marker = 'o'
        elif stripped_str == 'correct_change':
            label_str = 'Correct Change'
            marker = '^'
        elif stripped_str == 'incorrect_change':
            label_str = 'Incorrect Change'
            marker = '^'
        elif stripped_str == 'both_disagree_incorrect':
            label_str = 'Both Incorrect, Disagree'
            marker = '^'
        else:
            raise ValueError
        return label_str, marker
    
    colors = {
        'purple': (0.68, 0.36, 0.51),
        'blue': (0.0, 0.447, 0.698),
        'orange': (0.96, 0.624, 0.0),
        'green': (0.55, 0.71, 0.0),
        'yellow': (0.85, 0.694, 0.125),
        'darkgreen': (0.18, 0.35, 0.05),
        'red': (0.9, 0.17, 0.31),
        'cyan': (0.301, 0.745, 0.933),
    }
    
    DATASET_STYLE_MAP = {
        'amazon_purchases': 'density', 
        'chorales': 'scatter',
        'holidays_irregular': 'scatter',
        'rolling_mill': 'density',
    }
    assert dataset in DATASET_STYLE_MAP, f"Dataset '{dataset}' not available, should be from {list(DATASET_STYLE_MAP.keys())}."

    with open(f'./{LOGGING_DIR}/real/{dataset}/residual_network_behavior_individual.pkl', 'rb') as f:
        d = pickle.load(f)

    data_split = 'test'
    active_str = 'active'
    x_metric = 'entropy_rules'
    y_metric = 'symmetric_kl'

    x_metric_label = 'Entropy of RULES'
    y_metric_label = 'Symmetric KL'
    
    case_color_map = {
        f'{data_split}_active_both_correct': colors['darkgreen'],
        f'{data_split}_active_both_agree_incorrect': colors['blue'],
        f'{data_split}_active_correct_change': colors['green'],
        f'{data_split}_active_incorrect_change': colors['red'],
        f'{data_split}_active_both_disagree_incorrect': colors['orange'],
    }
    
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "lines.linewidth": 1.8,
        "lines.markersize": 7.5,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "pdf.fonttype": 42,
        "text.usetex": True,
        "font.family": "serif",
    })


    # global limits
    cases = [k for k in d.keys() if k.startswith(f'{data_split}_{active_str}')]
    all_x = np.concatenate([d[c][x_metric] for c in cases])
    all_y = np.concatenate([d[c][y_metric] for c in cases])
    assert len(all_x) == len(all_y)
    total = len(all_x)

    x_min, x_max = 0, np.max(all_x) + 0.2
    y_min, y_max = 0, np.max(all_y) + 0.2

    # layout
    fig, axes = plt.subplots(1, len(cases), figsize=(20,5), sharex=True, sharey=True, dpi=300)
    ax_num = -1
    legend_labels = []

    for case_name in cases:
        ax_num += 1
        ax = axes[ax_num]
        x_values = d[case_name][x_metric]
        y_values = d[case_name][y_metric]

        label_str, marker = case_label_and_marker_map(case_name)
        color = case_color_map[case_name]
        count = len(x_values)
            
        legend_labels.append(f'{label_str} ({count / total:.2f})')
        
        if DATASET_STYLE_MAP[dataset] == 'scatter':
            ax.scatter(
                x_values,
                y_values,
                label = label_str,
                alpha = 0.3,
                color = color,
                marker = marker,
            )
        elif DATASET_STYLE_MAP[dataset] == 'density':
            sns.kdeplot(
                x = x_values,
                y = y_values,
                ax = ax,
                fill = True,
                levels = 6,
                thresh = 0.05,
                cmap = sns.light_palette(color, as_cmap=True)
            )
        else:
            raise ValueError

        ax.grid(True)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        if ax_num == 0:
            ax.set_ylabel(y_metric_label)

    fig.supxlabel(x_metric_label)
   
    legend_lines = []
    for case_name in cases:
        label_str, marker = case_label_and_marker_map(case_name)
        color = case_color_map[case_name]
        if not DATASET_STYLE_MAP[dataset] == 'scatter':
            marker = 's'
        legend_lines.append(
            Line2D([], [],
                color = color,
                marker = marker,
                linestyle = 'None',
                markersize = 10,
                alpha = 1.0,
            )
        )
    
    leg = fig.legend(
        legend_lines,
        legend_labels,
        loc = 'upper center',
        ncol = len(legend_labels),
        frameon = True,
        bbox_to_anchor = (0.5, 1.1)  # move above plots
    )
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(1.2)
    leg.get_frame().set_boxstyle('Square')

    plt.tight_layout()
    plt.savefig(f"./{LOGGING_DIR}/real/{dataset}/residual_network_behavior_individual.pdf", bbox_inches="tight")
    plt.close()
    print(f"{dataset}, total events considered: {total}")
    
    
if __name__ == "__main__":
    plot_runtimes()
    plot_entropy_vs_kl_residual_network('chorales')
    plot_entropy_vs_kl_residual_network('holidays_irregular')
    plot_entropy_vs_kl_residual_network('rolling_mill')
    plot_entropy_vs_kl_residual_network('amazon_purchases')