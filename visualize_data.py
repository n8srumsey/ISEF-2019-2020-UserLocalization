"""
visualize_data.py
~~~~~~~~~~~~~~~~~~
Visualizaes results of hyperopt optimization.
"""

import json
import os
import pprint

from data_visualization import hyperparameter_learning_curves as hlc
from data_visualization import hyperspace_search_distribution as hsd
from data_visualization import learning_curves as lc
from data_visualization import scatterplot_matrices as sm
from data_visualization import trials_data as td

pp = pprint.PrettyPrinter(indent=4, width=100)

results_folder_path = "results"
results = sorted(os.listdir(results_folder_path))

jsons = []
for file_name in results:
    file_path = os.path.join(results_folder_path, file_name)
    with open(file_path) as f:
        j = json.load(f)
    jsons.append(j)


def print_dict_json_keys():
    print("Here are some useful keys in the dict/json structure:")
    pp.pprint(list(jsons[0].keys()))
    pp.pprint(list(jsons[0]["history"].keys()))
    pp.pprint(jsons[0]["space"])


if __name__ == "__main__":
    learn_curves, hyper_learn_curves, trials_data, scatter_matrix, hyperspace_distribution = \
        True, False, False, False, False
    # print_dict_json_keys()
    if learn_curves:
        lc.plot_learning_curves(retrained=False)
        lc.plot_learning_curves(retrained=True)
    if hyper_learn_curves:
        hlc.plot_val_accuracy_by_hyperparameters()
        # hlc.plot_val_loss_by_hyperparameters()
    if trials_data:
        td.plot_trials_data()
    if scatter_matrix:
        sm.plot_scatterplot_matrices()
    if hyperspace_distribution:
        # hsd.boxplots()
        hsd.scatter_vs_iteration()
