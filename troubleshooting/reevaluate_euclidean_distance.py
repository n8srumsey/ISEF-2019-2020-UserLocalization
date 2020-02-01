"""
reevaluate_euclidean_distance.py
~~~~~~~~~~~~~~~~~~

Fixes the error in euclidean distance evaluation.
"""
import os

from neural_net import euclidean_distance_metric, build_model
from utils import load_jsons
from visualize_data import results_folder_path

jsons = load_jsons()
for json in jsons:
    model_uuid = json["model_uuid"]
    space = json["space"]

    # build model
    model = build_model(space)

    # load saved final weights
    weights_path = format("weights/%s.hdf5", model_uuid)
    model.load_weights(weights_path)

    # evaluate euclidean distance
    metric_distance = euclidean_distance_metric(model)

    # update euclidean distance metric in json file
    file_name = json["model_name"]

    file_path = os.path.join(results_folder_path, file_name)
    with open(file_path) as f:
        j = json.load(f)
        j["euclidean_distance_error"] = metric_distance
        f.seek(0)  # rewind
        json.dump(j, f)
        f.truncate()

    print(metric_distance)  # for trouble shooting purposes
