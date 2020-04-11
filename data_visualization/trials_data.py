"""
trials_data.py
~~~~~~~~~~~~~~~~
Defines functions to print trials data.

"""
import math
import pickle

import matplotlib.pyplot as plt
import numpy
import numpy as np
from matplotlib.ticker import PercentFormatter

trials = pickle.load(open("./trials_history.pkl", "rb"))


def accuracy_loss_distance_vs_iteration():
    # fixme iterations start at 0
    data = [i["result"]["history"]["val_accuracy"][-1] for i in trials]

    f, (ax1) = plt.subplots(1)
    ax1.plot(data, linestyle='None', marker='.')
    #    ax1.axhline(max(val_acc_list))
    ax1.set_title('Validation Accuracy vs. Bayesian Optimization Iteration')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_xlabel('Iteration')

    xdata = [i for i in range(len(data))]

    y = [math.log1p(i) for i in data]

    z = numpy.polyfit(xdata, data, 1)
    p = numpy.poly1d(z)
    ax1.plot(p(xdata), "r--")

    plt.tight_layout(h_pad=.10)
    plt.subplots_adjust(top=0.85)
    plt.show()


def best_val_accuracy_and_loss_histogram():
    n_bins = 30
    val_acc_list = [i["result"]["history"]["val_accuracy"][-1] for i in trials]
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    ax1.hist(val_acc_list, bins=n_bins)
    ax1.set_title('Validation Accuracy Frequency')
    ax1.set_xlabel('Validation Accuracy')
    ax1.set_ylabel('Frequency')
    ax2.hist(val_acc_list, bins=n_bins, weights=np.ones(len(val_acc_list)) / len(val_acc_list))
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_title('Validation Accuracy Frequency')
    ax2.set_xlabel('Validation Accuracy')
    ax2.set_ylabel('Frequency')

    plt.suptitle('\nValidation Accuracy Frequencies')
    plt.show()


def plot_trials_data():
    # main_plot_history(trials)
    # main_plot_histogram(trials)

    accuracy_loss_distance_vs_iteration()
    best_val_accuracy_and_loss_histogram()
