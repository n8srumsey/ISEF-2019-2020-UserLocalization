"""
scatterplot_matrices.py
~~~~~~~~~~~~~~~~
Defines functions to print scatterplots of hyperparameter data.

"""
import json
import os
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib import colors, cm

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
best_accs = [neural_net["history"]["val_accuracy"][-1] for neural_net in jsons]


def scatterplot_matrix_colored(params_names, params_space_values, best_metric):
    # Scatterplot colored according to the Z values of the points.

    nb_params = len(params_space_values)
    best_metric = np.array([float(i) for i in best_metric])
    norm = colors.Normalize(vmin=best_metric.argmin(), vmax=best_metric.argmax())

    fig, ax = plt.subplots(nb_params - 1, nb_params - 1, figsize=(16, 16))  # , facecolor=bg_color, edgecolor=fg_color)
    cmap = cm.get_cmap('viridis')

    for r in range(nb_params - 1):
        p1 = params_space_values[r + 1]
        for c in range(nb_params - 1):
            p2 = params_space_values[c]
            axes = ax[r, c]
            if r >= c:
                # Subplot
                s = axes.scatter(p2, p1, s=10, c=cmap(best_metric), cmap=cmap, norm=norm)

                if r == 5:
                    axes.set_yticks([0.000, 0.003])
                    axes.set_ylim(-0.0007, 0.0037)
                if r == 6:
                    axes.set_yticks([0.000, 0.001])
                    axes.set_ylim(-0.00005, 0.0012)
                if c == 6:
                    axes.set_xticks([0.000, 0.003])
                    axes.set_xlim(-0.0007, 0.0037)
                if c == 7:
                    axes.set_xticks([0.000, 0.001])
                    axes.set_xlim(-0.00005, 0.0012)

                if r == 1:
                    axes.set_yticks([3, 5, 7])
                    axes.set_ylim(2, 8)
                if r == 3:
                    axes.set_yticks([0, 500, 1000])
                    axes.set_ylim(-100, 1100)
                if r == 4:
                    axes.set_yticks([0, 500, 1000])
                    axes.set_ylim(-100, 1100)
                if c == 2:
                    axes.set_xticks([3, 5, 7])
                    axes.set_xlim(2, 8)
                if c == 4:
                    axes.set_xticks([0, 500, 1000])
                    axes.set_xlim(-100, 1100)
                if c == 5:
                    axes.set_xticks([0, 500, 1000])
                    axes.set_xlim(-100, 1100)

                if r == 8:
                    axes.set_yticks([1, 2])
                    axes.set_ylim(0.5, 2.5)
                if r == 9:
                    axes.set_yticks([2, 3])
                    axes.set_ylim(1.5, 3.5)
                if c == 9:
                    axes.set_xticks([1, 2])
                    axes.set_xlim(0.5, 2.5)
                if c == 10:
                    axes.set_xticks([2, 3])
                    axes.set_xlim(1.5, 3.5)

                # Labels only on side subplots, for x and y:
                if c == 0:
                    if r == 0:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=40)
                    if r == 1:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=47)
                    if r == 2:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=45)
                    if r == 3:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=37)
                    if r == 4:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=42)
                    if r == 5:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=40)
                    if r == 6:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=25)
                    if r == 7:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=55)
                    if r == 8:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=85)
                    if r == 9:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=62)
                    if r == 10:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=41)
                    if r == 11:
                        axes.set_ylabel(params_names[r + 1], rotation=0, labelpad=40)

                else:
                    axes.set_yticks([])

                if r == nb_params - 2:
                    axes.set_xlabel(params_names[c], rotation=0)
                else:
                    axes.set_xticks([])

            else:
                fig.delaxes(ax[r, c])

    fig.subplots_adjust(right=0.82, top=0.95)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Validation Dataset Accuracy', rotation=270, labelpad=10)


    plt.suptitle(
        'Scatterplot Matrix of all Hyperparameter Values Selected During Optimization Phase')
    plt.show()


def plot_scatterplot_matrices():
    scatterplot_matrix_colored(int_params_names_to_correlate, params_values, best_accs)
