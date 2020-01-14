"""
trials_data.py
~~~~~~~~~~~~~~~~
Defines functions to print trials data.

"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

trials = pickle.load(open("./trials_history.pkl", "rb"))


def accuracy_loss_distance_vs_iteration():
    # fixme iterations start at 0
    val_acc_list = [i["result"]["history"]["val_accuracy"][-1] for i in trials]
    val_loss_list = [i["result"]["history"]["val_loss"][-1] for i in trials]
    distance_error_list = [i["result"]["euclidean_distance_error"] for i in trials]

    f, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(val_acc_list, linestyle='None', marker='.')
    ax1.axhline(max(val_acc_list))
    ax1.set_title('Validation Accuracy vs. Iteration')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_xlabel('Iteration')

    ax2.plot(val_loss_list, linestyle='None', marker='.')
    ax2.axhline(min(val_loss_list))
    ax2.set_ylim([0, 50])
    ax2.set_title('Validation Loss vs. Iteration')
    ax2.set_ylabel('Validation Loss')
    ax2.set_xlabel('Iteration')

    ax3.plot(distance_error_list, linestyle='None', marker='.')
    ax3.axhline(min(distance_error_list))
    ax3.set_ylim(0)
    ax3.set_title('Euclidean Distance Error vs. Iteration')
    ax3.set_ylabel('Euclidean Distance Error')
    ax3.set_xlabel('Iteration')

    plt.suptitle(" Validation Accuracy, Validation Loss, and Euclidean Distance Error vs. Iteration")
    plt.tight_layout(h_pad=.10)
    plt.subplots_adjust(top=0.85)
    plt.show()


def best_val_accuracy_and_loss_histogram():
    n_bins = 30
    val_acc_list = [i["result"]["history"]["val_accuracy"][-1] for i in trials]
    val_loss_list = [i["result"]["history"]["val_loss"][-1] for i in trials]
    val_loss_list_limited = [i for i in val_loss_list if i < 50]  # fixme
    distance_error_list = [i["result"]["euclidean_distance_error"] for i in trials]
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2, tight_layout=True)
    ax1.hist(val_acc_list, bins=n_bins)
    ax1.set_title('Validation Accuracy Frequency')
    ax1.set_xlabel('Validation Accuracy')
    ax1.set_ylabel('Frequency')
    ax2.hist(val_acc_list, bins=n_bins, weights=np.ones(len(val_acc_list)) / len(val_acc_list))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_title('Validation Accuracy Frequency')
    ax2.set_xlabel('Validation Accuracy')
    ax2.set_ylabel('Frequency')

    ax3.hist(val_loss_list_limited, bins=n_bins)
    ax3.set_title('Validation Loss Frequency')
    ax3.set_xlabel('Validation Loss')
    ax3.set_ylabel('Frequency')
    ax4.hist(val_loss_list_limited, bins=n_bins,
             weights=np.ones(len(val_loss_list_limited)) / len(val_loss_list_limited))
    ax4.yaxis.set_major_formatter(PercentFormatter(1))
    ax4.set_title('Validation Loss Frequency')
    ax4.set_xlabel('Validation Loss')
    ax4.set_ylabel('Frequency')

    before_shift_range = max(distance_error_list) - min(distance_error_list)
    after_shift_range = max(distance_error_list)
    range_scalar = before_shift_range / after_shift_range
    ax5.hist(distance_error_list, bins=int(n_bins * range_scalar + 3))
    ax5.set_title('Euclidean Distance vs. Frequency')
    ax5.set_xlabel('Euclidean Distance Error')
    ax5.set_ylabel('Frequency')
    ax5.set_xlim(0)
    ax6.hist(distance_error_list, bins=int(n_bins * range_scalar + 3),
             weights=np.ones(len(distance_error_list)) / len(distance_error_list))
    ax6.yaxis.set_major_formatter(PercentFormatter(1))
    ax6.set_title('Euclidean Distance vs. Frequency')
    ax6.set_xlabel('Euclidean Distance Error')
    ax6.set_ylabel('Frequency')
    ax6.set_xlim(0)

    plt.suptitle('Validation Accuracy, Validation Loss, and Euclidean Distance Error Frequencies', y=1)
    plt.tight_layout(h_pad=.25)
    plt.show()


def plot_trials_data():
    # main_plot_history(trials)
    # main_plot_histogram(trials)
    accuracy_loss_distance_vs_iteration()
    best_val_accuracy_and_loss_histogram()
