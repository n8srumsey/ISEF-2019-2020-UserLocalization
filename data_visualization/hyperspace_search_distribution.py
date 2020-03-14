"""
hyperspace_search_distribution.py
Defines function to display hyper-parameter search distributions across iterations.

"""
import pickle
import matplotlib.pyplot as plt
import numpy

trials = pickle.load(open("./trials_history.pkl", "rb"))

boxplot_or_not = {
    'conv_dropout': False,
    'conv_kernel_size': False,
    'fc_dropout_proba': True,
    'fc_nodes_1': True,
    'fc_second_layer': True,
    'l2_weight_reg': True,
    'lr_rate': False,
    'nb_conv_filters': True,
    'nb_conv_in_conv_pool_layers': False,
    'nb_conv_pool_layers': False
}


def boxplots():
    hyper_param_dict = {
        'conv_dropout': [],
        'conv_kernel_size': [],
        'fc_dropout_proba': [],
        'fc_nodes_1': [],
        'fc_second_layer': [],
        'l2_weight_reg': [],
        'lr_rate': [],
        'nb_conv_filters': [],
        'nb_conv_in_conv_pool_layers': [],
        'nb_conv_pool_layers': []
    }

    for trial in trials:
        for key in hyper_param_dict.keys():
            hyper_param_dict[key].append(trial['result']['space'][key])

    for key in hyper_param_dict.keys():
        data = list(hyper_param_dict[key])
        for i in range(len(data)):
            if data[i] is None:
                data[i] = 0  # fixme
        if boxplot_or_not[key]:
            plt.boxplot(data)
            plt.xticks([])
            plt.ylabel(key)
            plt.title("{} Search Distribution".format(key))
            plt.show()


def categorical(axs, j, k, data, xdata, key):
    axs[j][k].plot(data, marker='.', linestyle='None')
    axs[j][k].set_title("{} Search Distribution".format(key))
    axs[j][k].set_xlabel("Iteration")
    axs[j][k].set_ylabel(key)
    if key == "conv_kernel_size":
        axs[j][k].set_yticks([3, 5, 7])
    if key == "nb_conv_in_conv_pool_layers":
        axs[j][k].set_yticks([1, 2])
    if key == "nb_conv_pool_layers":
        axs[j][k].set_yticks([2, 3])


def interval(axs, j, k, data, xdata, key):
    axs[j][k].plot(data, marker='.', linestyle='None')
    axs[j][k].set_title("{} Search Distribution".format(key))
    axs[j][k].set_xlabel("Iteration")
    axs[j][k].set_ylabel(key)
    z = numpy.polyfit(xdata, data, 1)
    p = numpy.poly1d(z)
    axs[j][k].plot(xdata, p(xdata), "r--")


def scatter_vs_iteration():
    hyper_param_dict = {
        'activation': [],
        'conv_dropout': [],
        'conv_kernel_size': [],
        'fc_dropout_proba': [],
        'fc_nodes_1': [],
        'fc_second_layer': [],
        'l2_weight_reg': [],
        'lr_rate': [],
        'nb_conv_filters': [],
        'nb_conv_in_conv_pool_layers': [],
        'nb_conv_pool_layers': [],
        'optimizer': [],
        'pooling_type': []
    }
    hyper_param_type = {
        'activation': categorical,
        'conv_dropout': interval,
        'conv_kernel_size': categorical,
        'fc_dropout_proba': interval,
        'fc_nodes_1': interval,
        'fc_second_layer': interval,
        'l2_weight_reg': interval,
        'lr_rate': interval,
        'nb_conv_filters': interval,
        'nb_conv_in_conv_pool_layers': categorical,
        'nb_conv_pool_layers': categorical,
        'optimizer': categorical,
        'pooling_type': categorical
    }

    for trial in trials:
        for key in hyper_param_dict.keys():
            hyper_param_dict[key].append(trial['result']['space'][key])

    f, axs = plt.subplots(4, 4, constrained_layout=True)

    j = 0
    k = 0
    for key in hyper_param_dict.keys():
        data = list(hyper_param_dict[key])
        xdata = [i for i in range(len(data))]
        for i in range(len(data)):
            if data[i] is None:
                data[i] = 0
        hyper_param_type[key](axs, j, k, data, xdata, key)

        k += 1
        if k == 4:
            j += 1
            k = 0
    # fixme | add axes titles and suptitle
    f.delaxes(axs[3][1])
    f.delaxes(axs[3][2])
    f.delaxes(axs[3][3])
    plt.suptitle("Hyperparameter Value Selection by Iteration")
    plt.show()
