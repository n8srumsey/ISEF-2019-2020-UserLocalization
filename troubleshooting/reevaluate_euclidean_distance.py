"""
reevaluate_euclidean_distance.py
~~~~~~~~~~~~~~~~~~

Fixes the error in euclidean distance evaluation.
"""
import os
import json
from neural_net import euclidean_distance_metric, build_model
from utils import load_jsons

results_folder_path = "../results"

i = 1
jsons = load_jsons()

num_jsons = len(jsons)


if __name__ == '__main__':
    """Loads the final weights of each model and then re-evaluates the euclidean distance error of each model, in order 
    to have the correct and accurate metric values"""
    for json_file in jsons:
        print("Reevaluating model %d/" % i, num_jsons)

        model_uuid = json_file["model_uuid"]
        space = json_file["space"]

        # build model
        model = build_model(space)

        # load saved final weights
        weights_path = "weights/%s.hdf5" % model_uuid
        model.load_weights(weights_path)

        # evaluate euclidean distance
        metric_distance = euclidean_distance_metric(model)
        json_file["euclidean_distance_error"] = metric_distance

        # update euclidean distance metric in json file
        file_name = json_file["model_name"] + '.txt.json'

        file_path = os.path.join(results_folder_path, file_name)
        with open(file_path, 'w') as f:
            f.seek(0)  # rewind
            json.dump(json_file, f)
            f.truncate()
            f.close()

        print("Metric distance of model %d/" % i, "%d: " % num_jsons + str(metric_distance))
        i += 1
