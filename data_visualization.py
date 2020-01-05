"""
data_visualization.py
~~~~~~~~~~~~~~~~~~
Visualizaes results of hyperopt optimization.

This is based off of Vooban's demonstration repo @ https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100
"""

import json
import os
import pickle
import pprint
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from hyperopt.plotting import main_plot_history, main_plot_histogram
from matplotlib import colors

pp = pprint.PrettyPrinter(indent=4, width=100)

results_folder_path = "results"
results = sorted(os.listdir(results_folder_path))

jsons = []
for file_name in results:
    file_path = os.path.join(results_folder_path, file_name)
    with open(file_path) as f:
        j = json.load(f)
    jsons.append(j)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def print_dict_json_keys():
    print("Here are some useful keys in our dict/json structure:")
    pp.pprint(list(jsons[0].keys()))
    pp.pprint(list(jsons[0]["history"].keys()))
    pp.pprint(jsons[0]["space"])


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_learning_curves_by_epoch():
    plt.figure(figsize=(16, 12))
    for neural_net in jsons:
        accuracy = [1.0 / 100] + neural_net["history"]["accuracy"]
        end_accuracy = neural_net["end_accuracy"]
        cmap = cm.get_cmap('jet')
        rgba = cmap(end_accuracy)
        plt.plot(accuracy, color=rgba)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title("Learning curves over time, lines colored according to best test accuracy")
    plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def discrete_set(accs, key_name, key_values):
    plt.figure(figsize=(16, 12))

    key_values = [str(i) for i in key_values]

    colors = ["red", "blue", "green", "cyan", "magenta", "yellow", "black"]
    colors_mapping = {x: colors[i] for i, x in enumerate(set(key_values))}

    already_used_labels = set()
    for accuracy, val in zip(accs, key_values):
        if val in already_used_labels:
            plt.plot(accuracy, color=colors_mapping[val])
        else:
            plt.plot(accuracy, color=colors_mapping[val], label=val)
            already_used_labels.update({val})

    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy on fine labels')
    plt.title("Model performance in function of the '{}' hyperparameter".format(key_name))
    plt.legend()
    plt.show()


def int_val(accs, key_name, key_values):
    plt.figure(figsize=(16, 12))
    orig_kv = list(key_values)

    tmp_kval = [k for k in key_values if k is not None]
    min_val = min(tmp_kval)
    max_val = max(tmp_kval)
    for i, kv in enumerate(key_values):
        if kv is None:
            key_values[i] = "black"
            continue
        kv -= min_val
        kv = kv / (max_val - min_val)
        key_values[i] = float(kv)

    for accuracy, color, key_value in zip(accs, key_values, orig_kv):
        if type(color) is float:
            cmap = cm.get_cmap('jet')
            rgba = cmap(color)
            plt.plot(accuracy, color=rgba, label=str(key_value))
        else:
            plt.plot(accuracy, color="black")

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title("Model performance in function of the '{}' hyperparameter".format(key_name))
    plt.legend()
    plt.show()


plot_function_map_from_key = {
    'activation': discrete_set,
    'conv_dropout': int_val,
    'conv_kernel_size': discrete_set,
    'fc_dropout_proba': int_val,
    'fc_nodes_1': int_val,
    'fc_second_layer': int_val,
    'l2_weight_reg_mult': int_val,
    'lr_rate': int_val,
    'nb_conv_filters': int_val,
    'nb_conv_in_conv_pool_layers': discrete_set,
    'nb_conv_pool_layers': discrete_set,
    'optimizer': discrete_set,
    'pooling_type': discrete_set
}


def plot_accuracy_by_hyperparameters():
    for key, plot_func in plot_function_map_from_key.items():
        accs = [
            [1.0 / 100] + neural_net["history"]["val_accuracy"] for neural_net in jsons
        ]
        key_values = [
            neural_net["space"][key] for neural_net in jsons
        ]
        plot_func(accs, key, key_values)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def plot_trials_data():
    trials = pickle.load(open("trials_history.pkl", "rb"))

    print("Now plotting with some built-in functions.")
    print("Remember that the loss is the negative of the test accuracy on fine labels.\n")

    main_plot_history(trials)
    main_plot_histogram(trials)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


int_params_names_to_correlate = [
    'activation',
    'conv_dropout',
    'conv_kernel_size',
    'fc_dropout_proba',
    'fc_nodes_1',
    'fc_second_layer',
    'l2_weight_reg_mult',
    'lr_rate',
    'nb_conv_filters',
    'nb_conv_in_conv_pool_layers',
    'nb_conv_pool_layers',
    'optimizer',
    'pooling_type'
]

params_values = [[neural_net["space"][p] for neural_net in jsons] for p in int_params_names_to_correlate]
best_accs = [neural_net["best_accuracy"] for neural_net in jsons]


def scatterplot_matrix_colored(params_names, params_values, best_accs, blur=False):
    """Scatterplot colored according to the Z values of the points."""

    nb_params = len(params_values)
    best_accs = np.array(best_accs)
    norm = colors.Normalize(vmin=best_accs.argmin(), vmax=best_accs.argmax())

    fig, ax = plt.subplots(nb_params, nb_params, figsize=(16, 16))  # , facecolor=bg_color, edgecolor=fg_color)

    for i in range(nb_params):
        p1 = params_values[i]
        for j in range(nb_params):
            p2 = params_values[j]

            axes = ax[i, j]
            # Subplot:
            if blur:
                s = axes.scatter(p2, p1, s=400, alpha=.1,
                                 c=best_accs, cmap='viridis', norm=norm)
                s = axes.scatter(p2, p1, s=200, alpha=.2,
                                 c=best_accs, cmap='viridis', norm=norm)
                s = axes.scatter(p2, p1, s=100, alpha=.3,
                                 c=best_accs, cmap='viridis', norm=norm)
            s = axes.scatter(p2, p1, s=15,
                             c=best_accs, cmap='plasma', norm=norm)

            # Labels only on side subplots, for x and y:
            if j == 0:
                axes.set_ylabel(params_names[i], rotation=0)
            else:
                axes.set_yticks([])

            if i == nb_params - 1:
                axes.set_xlabel(params_names[j], rotation=90)
            else:
                axes.set_xticks([])

    fig.subplots_adjust(right=0.82, top=0.95)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(s, cax=cbar_ax)

    plt.suptitle(
        'Scatterplot matrix of tried values in the search space over different params, colored in function of best '
        'test accuracy')
    plt.show()


def plot_scatterplot_matrices():
    scatterplot_matrix_colored(int_params_names_to_correlate, params_values, best_accs, blur=True)
    scatterplot_matrix_colored(int_params_names_to_correlate, params_values, best_accs, blur=False)


# print_dict_json_keys()
# plot_learning_curves_by_epoch()
# plot_accuracy_by_hyperparameters()
# plot_trials_data()
# plot_scatterplot_matrices()
