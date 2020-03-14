"""
statistical-analysis.py
~~~~~~~~~~~~~~~~~~

Performs and records results of statistical analyses.
"""
import os
import scipy.stats as stats
import pprint
from utils import load_jsons

pp = pprint.PrettyPrinter(indent=4, width=100)

results_folder_path = "../results"
results = sorted(os.listdir(results_folder_path))

jsons = load_jsons()

# Calculate t-test of training versus validation accuracy and loss
t_statistics = [[], []]
p_values = [[], []]
accuracies = []
val_accuracies = []
losses = []
val_losses = []

for json in jsons:
    accuracy = [float(i) for i in json['history']['accuracy'][0:]]
    val_accuracy = [float(i) for i in json['history']['val_accuracy'][0:]]
    loss = [float(i) for i in json['history']['loss'][0:]]
    val_loss = [float(i) for i in json['history']['val_loss'][0:]]

    accuracies.append(accuracy)
    val_accuracies.append(val_accuracy)
    losses.append(loss)
    val_losses.append(val_loss)

    acc_t_stat, acc_p_values = stats.ttest_ind(accuracy, val_accuracy, None, False, 'raise')
    loss_t_stat, loss_p_values = stats.ttest_ind(loss, val_loss, None, False, 'raise')

    t_statistics[0].append(acc_t_stat)
    t_statistics[1].append(loss_t_stat)
    p_values[0].append(acc_p_values)
    p_values[1].append(loss_p_values)


def rmse(pred_list, true_list):
    if len(pred_list[0]) == len(true_list[0]):
        return [((sum([(truth - prediction) ** 2 for truth, prediction in zip(pred, true)])) / len(pred)) ** 0.5
                for pred, true in zip(pred_list, true_list)]


def mae(pred_list, true_list):
    if len(pred_list[0]) == len(true_list[0]):
        return [(sum([(abs(truth - prediction)) for truth, prediction in zip(pred, true)])) / len(pred)
                for pred, true in zip(pred_list, true_list)]


rmse_acc = rmse(val_accuracies, accuracies)
rmse_loss = rmse(val_losses, losses)
mae_acc = mae(val_accuracies, accuracies)
mae_loss = mae(val_losses, losses)

if __name__ == '__main__':
    print('T-statistic: [accuracy, loss]')
    pp.pprint(t_statistics[0])
    pp.pprint(t_statistics[1])
    print('\nP-values: [accuracy, loss]')
    pp.pprint(p_values[0])
    pp.pprint(p_values[1])

    print('\nRoot Square Mean Error: [accuracy, loss]')
    pp.pprint(rmse_acc)
    pp.pprint(rmse_loss)

    print('\n Mean Absolute Error:m [accuracy, loss]')
    pp.pprint(mae_acc)
    pp.pprint(mae_loss)

    print('\nAccuracy:')
    data = ([t_statistics[0], p_values[0], rmse_acc, mae_acc])
    i = 0
    names = ['T-stat:', 'P-value:', 'RMSE:', 'MAE:']
    for line in data:
        line = [str(item).ljust(25, ' ') for item in line]
        string = ""
        for item in line:
            string += item
        print(names[i].ljust(15, ' ') + string)
        i += 1

    data = [t_statistics[1], p_values[1], rmse_loss, mae_loss]
    print('\nLoss:')
    i = 0
    names = ['T-stat:', 'P-value:', 'RMSE:', 'MAE:']
    for line in data:
        line = [str(item).ljust(25, ' ') for item in line]
        string = ""
        for item in line:
            string += item
        print(names[i].ljust(15, ' ') + string)
        i += 1
