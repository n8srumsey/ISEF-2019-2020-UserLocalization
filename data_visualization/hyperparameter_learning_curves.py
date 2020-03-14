"""
learning_curves.py
~~~~~~~~~~~~~~~~
Defines functions to print learning curves by epoch by hyperparameters.

"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

results_folder_path = ".\\results"
results = sorted(os.listdir(results_folder_path))

jsons = []
for file_name in results:
    file_path = os.path.join(results_folder_path, file_name)
    with open(file_path) as f:
        j = json.load(f)
    jsons.append(j)


def discrete_set(accs, key_name, key_values):
    plt.figure()

    key_values = [str(i) for i in key_values]

    cmap = cm.get_cmap('Set1')
    clr_val = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0, 0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67,
               0.78, 0.89, 1.0, 0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0]
    colors = [cmap(i) for i in clr_val]
    colors_mapping = {x: colors[i] for i, x in enumerate(set(key_values))}

    already_used_labels = set()
    for accuracy, val in zip(accs, key_values):
        if val in already_used_labels:
            plt.plot(accuracy, color=colors_mapping[val])
        else:
            plt.plot(accuracy, color=colors_mapping[val], label=val)
            already_used_labels.update({val})

    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.title("Model performance in function of the '{}' hyperparameter".format(key_name))
    plt.legend()
    plt.show()


def int_val(accs, key_name, key_values):
    global cmap, norm
    fig = plt.figure()
    orig_kv = list(key_values)

    tmp_kval = [k for k in key_values if k is not None]
    min_val = min(tmp_kval)
    max_val = max(tmp_kval)
    for i, kv in enumerate(key_values):
        if kv is None:
            key_values[i] = float(min_val)
            continue
        key_values[i] = float(kv)

    for accuracy, color, key_value in zip(accs, key_values, orig_kv):
        if type(color) is float:
            cmap = cm.get_cmap('viridis')
            norm = colors.Normalize(vmin=min_val, vmax=max_val)
            if key_name == "lr_rate":
                norm = colors.Normalize(vmin=min_val, vmax=0.005)
            rgba = cmap(norm(color))
            plt.plot(accuracy, color=rgba, label=str(key_value))
        else:
            plt.plot(accuracy, color="black")

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title("Model performance in function of the '{}' hyperparameter".format(key_name))

    fig.subplots_adjust(right=0.82, top=0.95)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
    num_ticks = 5
    yticklabels = [str(min_val + i*(max_val-min_val)/num_ticks)[:6] for i in range(num_ticks + 1)]
    if key_name == "nb_conv_filters" or key_name == "fc_nodes_1" or key_name == "fc_second_layer":
        yticklabels = [str(round(min_val + i * (max_val - min_val) / num_ticks))[:6] for i in range(num_ticks + 1)]
    if key_name == "lr_rate":
        yticklabels = [format(0.00005 + i * (max_val - min_val) / num_ticks, '.6f')[:7] for i in range(int(num_ticks + 1))]
    cbar_ax.set_yticklabels(yticklabels)
    cbar_ax.set_ylabel(key_name, rotation=270, labelpad=10)

    plt.show()


plot_function_map_from_key = {
    'activation': discrete_set,
    'conv_dropout': int_val,
    'conv_kernel_size': discrete_set,
    'fc_dropout_proba': int_val,
    'fc_nodes_1': int_val,
    'fc_second_layer': int_val,
    'l2_weight_reg': int_val,
    'lr_rate': int_val,
    'nb_conv_filters': int_val,
    'nb_conv_in_conv_pool_layers': discrete_set,
    'nb_conv_pool_layers': discrete_set,
    'optimizer': discrete_set,
    'pooling_type': discrete_set
}


def plot_val_accuracy_by_hyperparameters():
    for key, plot_func in plot_function_map_from_key.items():
        accs = [neural_net["history"]["val_accuracy"] for neural_net in jsons]
        key_values = [neural_net["space"][key] for neural_net in jsons]
        plot_func(accs, key, key_values)
