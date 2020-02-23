"""
reevaluate_euclidean_distance.py
~~~~~~~~~~~~~~~~~~

Fixes the error in euclidean distance evaluation.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from keras_preprocessing.image import img_to_array, ImageDataGenerator, load_img
import cv2

from neural_net import build_model, dataset_input_resize
from utils import load_jsons

datagen = ImageDataGenerator(rescale=1. / 225)
datagenerator = datagen.flow_from_directory('../data/validation/', class_mode='categorical',
                                            target_size=dataset_input_resize, batch_size=1, shuffle=False)

# Setup necessary for euclidean distance metric
dict_coordinate_to_index = datagenerator.class_indices
coordinate_names = []
for coordinate in list(dict_coordinate_to_index.keys()):
    index_y = coordinate.find('Y')
    x_value = int(coordinate[1:index_y])
    y_value = int(coordinate[(index_y + 1):])
    coordinate_names.append([x_value, y_value])


def euclidean_distance_metric_individual(model):
    rootdir = '../data/validation'

    metric_distance_list = []

    j = 0

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # load the image
            image = cv2.imread(os.path.join(subdir, file))
            res = cv2.resize(image, dsize=dataset_input_resize, interpolation=cv2.INTER_CUBIC)
            swap = np.swapaxes(res, 0, 1)

            # img = load_img(res)
            # im = img_to_array(img)
            samples = np.expand_dims(swap, 0)
            it = datagen.flow(samples, batch_size=1)
            batch = it.next()

            # predict class
            pred = model.predict_classes(x=batch)
            index_pred = int(np.argmax(pred))
            pred_x = coordinate_names[pred[0]][0]
            pred_y = coordinate_names[pred[0]][1]

            # get truth
            index_true = dict_coordinate_to_index[Path(file).stem.split('(')[0]]
            true_x = coordinate_names[index_true][0]
            true_y = coordinate_names[index_true][1]

            # print predicted and actual coordinates
            """if pred_x != true_x or pred_y != true_y:
                print("      Predicted Coordinate: (%d, %d) \n      Actual Coordinates: (%d, %d)" % (pred_x, pred_y,
                                                                                                     true_x, true_y))"""

            """if pred_x == 0 and pred_y == 0:
                print(j)
                j += 1"""

            # save result to list
            metric_distance_list.append(math.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2))

    # find average euclidean distance error
    resultant_metric_distance = sum(metric_distance_list) / len(metric_distance_list)

    return resultant_metric_distance


def euclidean_distance_metric_datagen(model):
    metric_distance_list = []

    true_max_index_list = datagenerator.classes

    j = 0

    pred = model.predict_generator(datagenerator)
    print(pred.shape)  # 37317, 103

    for prediction in pred:
        # interpret prediction
        index_pred = int(np.argmax(prediction))
        index_true = true_max_index_list[j]
        pred_x = coordinate_names[index_pred][0]
        pred_y = coordinate_names[index_pred][1]
        true_x = coordinate_names[index_true][0]
        true_y = coordinate_names[index_true][1]

        """# print predicted and actual coordinates
        if pred_x != true_x or pred_y != true_y:
            print("      Predicted Coordinate: (%d, %d) \n      Actual Coordinates: (%d, %d)" % (pred_x, pred_y,
                                                                                                 true_x, true_y))"""

        # save result to list
        metric_dist = math.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
        metric_distance_list.append(metric_dist)

        j += 1

    # find average euclidean distance error
    resultant_metric_distance = sum(metric_distance_list) / len(metric_distance_list)

    return resultant_metric_distance


def euclidean_distance_metric_individual_generator(model):
    rootdir = '../data/validation'

    metric_distance_list = []

    true_max_index_list = datagenerator.classes

    j = 0

    # pred = model.predict_generator(datagenerator)
    # print(pred.shape)  # 37317, 103

    for subdir, dirs, files in os.walk(rootdir):
        for _ in files:
            # load the image
            im_arr = datagenerator.next()[0]

            # predict class
            pred = model.predict_classes(x=im_arr)

            # interpret prediction
            index_pred = int(np.argmax(pred))
            index_true = true_max_index_list[i]
            pred_x = coordinate_names[index_pred][0]
            pred_y = coordinate_names[index_pred][1]
            true_x = coordinate_names[index_true][0]
            true_y = coordinate_names[index_true][1]

            # print predicted and actual coordinates
            """if pred_x != true_x or pred_y != true_y:
                print("      Predicted Coordinate: (%d, %d) \n      Actual Coordinates: (%d, %d)" % (pred_x, pred_y,
                                                                                                     true_x, true_y))"""

            # save result to list
            metric_distance_list.append(math.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2))

            j += 1

    # find average euclidean distance error
    resultant_metric_distance = sum(metric_distance_list) / len(metric_distance_list)

    return resultant_metric_distance


i = 1

if __name__ == '__main__':
    """Loads the final weights of each model and then re-evaluates the euclidean distance error of each model, in order 
    to have the correct and accurate metric values"""
    do_w_retrained_models = True

    if do_w_retrained_models:
        results_folder_path = "../results-retrained"
        results = sorted(os.listdir(results_folder_path))
        jsons = []
        for file_name in results:
            file_path = os.path.join(results_folder_path, file_name)
            with open(file_path) as f:
                j = json.load(f)
            jsons.append(j)
    else:
        results_folder_path = "../results"
        jsons = load_jsons()

    num_jsons = len(jsons)

    for json_file in jsons:
        print("Reevaluating model %d/" % i + str(num_jsons))
        print("Model UUID: " + json_file['model_uuid'])

        model_uuid = json_file["model_uuid"]
        space = json_file["space"]

        # build model
        model = build_model(space)

        # load saved final weights
        if do_w_retrained_models:
            weights_path = "../weights-retrained/%s.hdf5" % model_uuid
        else:
            weights_path = "../weights/%s.hdf5" % model_uuid

        model.load_weights(weights_path)

        # evaluate euclidean distance
        metric_distance = euclidean_distance_metric_individual(model)
        json_file["euclidean_distance_error"] = metric_distance

        # update euclidean distance metric in json file
        file_name = json_file["model_name"] + '.txt.json'

        file_path = os.path.join(results_folder_path, file_name)
        with open(file_path, 'w') as f:
            f.seek(0)  # rewind
            json.dump(json_file, f)
            f.truncate()
            f.close()

        print("Metric distance of model %d/" % i + "%d: " % num_jsons + str(metric_distance))
        i += 1
