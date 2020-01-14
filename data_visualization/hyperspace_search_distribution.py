"""
hyperspace_search_distribution.py
Defines function to display hyper-parameter search distributions across iterations.

"""
import pickle
import matplotlib.pyplot as plt

trials = pickle.load(open("./trials_history.pkl", "rb"))


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
        'nb_conv_pool_layers': [],
    }

    for trial in trials:
        for key in hyper_param_dict.keys():
            hyper_param_dict[key].append(trial['result']['space'][key])

    for key in hyper_param_dict.keys():
        data = list(hyper_param_dict[key])
        for i in range(len(data)):
            if data[i] is None:
                data[i] = 0  # fixme
        plt.boxplot(data)
        plt.title("{} Search Distribution".format(key))
        plt.show()


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

    for trial in trials:
        for key in hyper_param_dict.keys():
            hyper_param_dict[key].append(trial['result']['space'][key])

    f, axs = plt.subplots(4, 4)

    j = 0
    k = 0
    for key in hyper_param_dict.keys():
        data = list(hyper_param_dict[key])
        for i in range(len(data)):
            if data[i] is None:
                data[i] = 0
        axs[j][k].plot(data, marker='.', linestyle='None')
        axs[j][k].set_title("{} Search Distribution".format(key))

        k += 1
        if k == 4:
            j += 1
            k = 0
    # fixme | add axes titles and suptitle
    f.delaxes(axs[3][1])
    f.delaxes(axs[3][2])
    f.delaxes(axs[3][3])
    plt.show()
