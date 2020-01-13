"""
scatterplot_matrices.py
~~~~~~~~~~~~~~~~
Defines functions to print scatterplots of hyperparameter data.

"""
import json
import os
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib import colors

results_folder_path = "./results"
results = sorted(os.listdir(results_folder_path))

jsons = []
for file_name in results:
    file_path = os.path.join(results_folder_path, file_name)
    with open(file_path) as f:
        j = json.load(f)
    jsons.append(j)

int_params_names_to_correlate = [
    'activation',
    'conv_dropout',
    'conv_kernel_size',
    'fc_dropout_proba',
    'fc_nodes_1',
    'fc_second_layer',
    'l2_weight_reg',
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
    # Scatterplot colored according to the Z values of the points.

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
