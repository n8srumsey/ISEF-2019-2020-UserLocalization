"""
learning_curves.py
~~~~~~~~~~~~~~~~
Defines functions to print learning curves by epoch.

"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm


results_folder_path = ".\\results"
results = sorted(os.listdir(results_folder_path))

jsons = []
for file_name in results:
    file_path = os.path.join(results_folder_path, file_name)
    with open(file_path) as foo:
        j = json.load(foo)
    jsons.append(j)


def plot_learning_curves_by_epoch():
    f, (axs1, axs2) = plt.subplots(2)
    for neural_net in jsons:
        accuracy = [float(i) for i in neural_net["history"]["accuracy"]] # convert strings to floats
        val_accuracy = neural_net["history"]["val_accuracy"]
        test_accuracy = neural_net["end_accuracy"]

        cmap = cm.get_cmap('jet')
        rgba = cmap(test_accuracy)

        axs2.plot(val_accuracy, color=rgba)
        axs1.plot(accuracy, color=rgba)

    axs1.set_xlabel('Epoch')
    axs1.set_ylabel('Accuracy')
    axs1.set_title("Accuracy vs. Epoch")

    axs2.set_xlabel('Epoch')
    axs2.set_ylabel('Validation Accuracy')
    axs2.set_title("Validation Accuracy vs. Epoch")

    plt.suptitle("Learning curves over time, lines colored according to test accuracy")
    plt.tight_layout(h_pad=.5)
    plt.show()

    """Plot loss learning curves"""
    f, (axs1, axs2) = plt.subplots(2)
    for neural_net in jsons:
        loss = neural_net["history"]["loss"]
        val_loss = neural_net["history"]["val_loss"]
        test_loss = neural_net["end_loss"]

        cmap = cm.get_cmap('jet')  # fixme should there be a universal cmap for all data representation?
        rgba = cmap(test_loss)

        axs1.plot(loss, color=rgba)
        axs2.plot(val_loss, color=rgba)

    axs1.set_ylim([0, 30])  # fixme
    axs2.set_ylim([0, 30])  # fixme

    axs1.set_xlabel('Epoch')
    axs1.set_ylabel('Loss')
    axs1.set_title("Loss vs. Epoch")

    axs2.set_xlabel('Epoch')
    axs2.set_ylabel('Validation Loss')
    axs2.set_title("Validation Loss vs. Epoch")

    plt.suptitle("Learning curves over time, lines colored according to test loss")
    plt.show()

