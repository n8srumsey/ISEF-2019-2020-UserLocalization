"""
learning_curves.py
~~~~~~~~~~~~~~~~
Defines functions to print learning curves by epoch by hyperparameters.

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
    with open(file_path) as f:
        j = json.load(f)
    jsons.append(j)


def discrete_set(accs, key_name, key_values):
    plt.figure()

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
    plt.ylabel('Validation accuracy')
    plt.title("Model performance in function of the '{}' hyperparameter".format(key_name))
    plt.legend()
    plt.show()


def int_val(accs, key_name, key_values):
    plt.figure()
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


def discrete_set_loss(accs, key_name, key_values):
    plt.figure()

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
    plt.ylabel('Validation Loss')
    plt.ylim([0, 30])  # fixme
    plt.title("Model performance in function of the '{}' hyperparameter".format(key_name))
    plt.legend()
    plt.show()


def int_val_loss(accs, key_name, key_values):
    plt.figure()
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
    plt.ylabel('Validation Loss')
    plt.ylim([0, 30])  # fixme
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
    'l2_weight_reg': int_val,
    'lr_rate': int_val,
    'nb_conv_filters': int_val,
    'nb_conv_in_conv_pool_layers': discrete_set,
    'nb_conv_pool_layers': discrete_set,
    'optimizer': discrete_set,
    'pooling_type': discrete_set
}

plot_function_map_from_key_loss = {
    'activation': discrete_set_loss,
    'conv_dropout': int_val_loss,
    'conv_kernel_size': discrete_set_loss,
    'fc_dropout_proba': int_val_loss,
    'fc_nodes_1': int_val_loss,
    'fc_second_layer': int_val_loss,
    'l2_weight_reg': int_val_loss,
    'lr_rate': int_val_loss,
    'nb_conv_filters': int_val_loss,
    'nb_conv_in_conv_pool_layers': discrete_set_loss,
    'nb_conv_pool_layers': discrete_set_loss,
    'optimizer': discrete_set_loss,
    'pooling_type': discrete_set_loss
}


def plot_val_accuracy_by_hyperparameters():
    for key, plot_func in plot_function_map_from_key.items():
        accs = [neural_net["history"]["val_accuracy"] for neural_net in jsons]
        key_values = [neural_net["space"][key] for neural_net in jsons]
        plot_func(accs, key, key_values)


def plot_val_loss_by_hyperparameters():
    for key, plot_func in plot_function_map_from_key_loss.items():
        losses = [neural_net["history"]["val_loss"] for neural_net in jsons]
        key_values = [neural_net["space"][key] for neural_net in jsons]
        plot_func(losses, key, key_values)
