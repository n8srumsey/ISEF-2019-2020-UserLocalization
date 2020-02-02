"""
utils.py
~~~~~~~~~~~~~~~~~~
Contains functions to save, load, and print the training results with JSON utils.

This is based off of Vooban's demonstration repo @ https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100
"""

import json
import os

RESULTS_DIR = "results/"
RETRAINED_RESULTS_DIR = "results-retrained/"


# For standard training and optimization steps
def print_json(result):
    """Pretty-print a jsonable structure (e.g.: result)."""
    print(json.dumps(
        result,
        default=str, sort_keys=True,
        indent=4, separators=(',', ': ')
    ))


def save_json_result(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    with open(os.path.join(RESULTS_DIR, result_name), 'w') as f:
        json.dump(
            result, f,
            default=str, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join(RESULTS_DIR, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(f.read())


def load_best_hyperspace():
    """Loads the hyperspace which yielded the best accuracy."""
    results = [
        f for f in list(sorted(os.listdir(RESULTS_DIR))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result(best_result_name)["space"]

  
def load_jsons():
    results_folder_path = "results"
    results = sorted(os.listdir(results_folder_path))
    jsons = []
    for file_name in results:
        file_path = os.path.join(results_folder_path, file_name)
        with open(file_path) as f:
            j = json.load(f)
        jsons.append(j)
    return jsons
  

# For retraining of best model hyper-parameters
def save_json_result_retrained(model_name, result):
    """Save json to a directory and a filename."""
    result_name = '{}.txt.json'.format(model_name)
    if not os.path.exists(RETRAINED_RESULTS_DIR):
        os.makedirs(RETRAINED_RESULTS_DIR)
    with open(os.path.join(RETRAINED_RESULTS_DIR, result_name), 'w') as f:
        json.dump(
            result, f,
            default=str, sort_keys=True,
            indent=4, separators=(',', ': ')
        )


def load_json_result_retrained(best_result_name):
    """Load json from a path (directory + filename)."""
    result_path = os.path.join(RETRAINED_RESULTS_DIR, best_result_name)
    with open(result_path, 'r') as f:
        return json.JSONDecoder().decode(f.read())


def load_best_hyperspace_retrained():
    """Loads the hyperspace which yielded the best accuracy."""
    results = [
        f for f in list(sorted(os.listdir(RETRAINED_RESULTS_DIR))) if 'json' in f
    ]
    if len(results) == 0:
        return None

    best_result_name = results[-1]
    return load_json_result_retrained(best_result_name)["space"]


