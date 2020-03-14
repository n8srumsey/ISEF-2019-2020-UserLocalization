"""
learning_curves.py
~~~~~~~~~~~~~~~~
Defines functions to print learning curves by epoch.

"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

from utils import load_jsons

results_folder_path = ".\\results"
results = sorted(os.listdir(results_folder_path))

jsons = []
for file_name in results:
    file_path = os.path.join(results_folder_path, file_name)
    with open(file_path) as foo:
        j = json.load(foo)
    jsons.append(j)


def plot_learning_curves(retrained=False):
    global jsons
    if retrained:
        results_folder_path = ".\\results-retrained"
        results = sorted(os.listdir(results_folder_path))
        jsons = []
        for file_name in results:
            file_path = os.path.join(results_folder_path, file_name)
            with open(file_path) as foo:
                j = json.load(foo)
            jsons.append(j)

    fig, ax = plt.subplots(2, 1, constrained_layout=False)  # , facecolor=bg_color, edgecolor=fg_color)
    axs1 = ax[0]
    axs2 = ax[1]
    cmap = cm.get_cmap('jet')
    norm = colors.Normalize(vmin=0, vmax=1)
    if retrained:
        norm = colors.Normalize(vmin=0.97, vmax=.998)

    for neural_net in jsons:
        accuracy = [float(i) for i in neural_net["history"]["accuracy"]]  # convert strings to floats
        val_accuracy = neural_net["history"]["val_accuracy"]

        rgba = cmap(norm(val_accuracy[-1]))

        axs2.plot(val_accuracy, color=rgba)
        axs1.plot(accuracy, color=rgba)

    axs1.set_xlabel('Epoch')
    axs1.set_ylabel('Training Accuracy')
    axs1.set_title("Training Accuracy vs. Epoch")

    axs2.set_xlabel('Epoch')
    axs2.set_ylabel('Validation Accuracy')
    axs2.set_title("Validation Accuracy vs. Epoch")

    axs1.set_xticks([0, 4, 8, 12, 16, 20, 24])
    axs1.set_xlim(0, 23)
    axs2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs2.set_ylim(0, 1)

    axs2.set_xticks([0, 4, 8, 12, 16, 20, 24])
    axs2.set_xlim(0, 23)
    axs2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs2.set_ylim(0, 1)

    if retrained:
        axs1.set_xticks([6*i for i in range(int(72/6))])
        axs1.set_xlim(0, 71)
        axs1.set_yticks([0.95, 0.96, 0.97, 0.98, .99, 1.0])
        axs1.set_ylim(0.95, 1)
        axs2.set_xticks([6*i for i in range(int(72/6))])
        axs2.set_xlim(0, 71)
        axs2.set_yticks([0.95, 0.94, 0.96, .97, 0.98, .99, 1.0])
        axs2.set_ylim(0.95, 1)

    plt.suptitle("Optimization Phase Accuracies on Training and Validation Datasets")
    if retrained:
        plt.suptitle("Retraining Phase Accuracies on Training and Validation Datasets")

    plt.tight_layout(pad=2.0)

    fig.subplots_adjust(right=0.82, top=0.87)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cbar_ax)
    cbar_ax.set_ylabel('Accuracy', rotation=270, labelpad=10)
    if retrained:
        cbar_ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar_ax.set_yticklabels(['0.95', '0.96', '0.97', '0.98', '0.99', '1.00'])



    plt.show()
