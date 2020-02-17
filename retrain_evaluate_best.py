"""
retrain_evaluate_best.py
~~~~~~~~~~~~~~~~~~
Optimizes the specified neural_net model using hyperopt, based off of the designated search space.

This is based off of Vooban's demonstration repo @ https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100
"""
import json
import os
import traceback

import keras.backend as K
from keras.utils import plot_model

from neural_nets_retrain import build_and_train, build_model
from utils import print_json, save_json_result_retrained, load_best_hyperspace_retrained, load_jsons


def plot(hype_space, file_name_prefix):
    """Plot a model from it's hyperspace."""
    model = build_model(hype_space)
    plot_model(
        model,
        to_file='model-visualizations-retrained/{}.png'.format(file_name_prefix),
        show_shapes=True
    )
    print("Saved model visualization to model-visualizations-retrained/{}.png.".format(file_name_prefix))
    K.clear_session()
    del model


def plot_best_model():
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace_retrained()
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    # Print best hyperspace and save model png
    print("Best hyperspace yet:")
    print_json(space_best_model)
    plot(space_best_model, "retrained_model_best")


def train_cnn(hype_space, model_uuid_real):
    """Build a convolutional neural network and train it."""
    try:
        model, _, result, model_uuid = build_and_train(hype_space, model_uuid_real)

        # Save training results to disks with unique filenames
        save_json_result_retrained(model_uuid, result)

        # Save .png plot of the model according to its hyper-parameters
        plot(result['space'], model_uuid)

        K.clear_session()
        del model

        return result

    except Exception as error:
        K.clear_session()
        err_string = str(error)
        print(err_string)
        traceback_string = str(traceback.format_exc())
        print(traceback_string)
        print("\n\n")
        return {
            'err': err_string,
            'traceback': traceback_string
        }


if __name__ == "__main__":
    """Iterate through the 10 best models found in the Bayesian Hyper-parameter Optimization"""

    print("\nResults will be saved in the folder named 'results-retrained/'")

    # The following code is used to find and store the ten best models
    results_folder_path = "results"
    results = sorted(os.listdir(results_folder_path))

    jsons = load_jsons()

    jsons_best = sorted(range(len(jsons)), key=lambda i: jsons[i]["history"]["val_accuracy"][-1], reverse=True)[:10]
    # jsons_best = sorted(range(len(jsons)), key=lambda i: jsons[i]["history"]["val_loss"][-1], reverse=True)[:10]
    best_jsons = [jsons[i] for i in jsons_best]

    i = 1
    for json in best_jsons:
        hyperspace = json["space"]
        # Optimize a new model with the TPE Algorithm:
        print("\n\nTRAINING NEXT MODEL: {}/10".format(str(i)))
        try:
            train_cnn(hyperspace, json["model_uuid"])
            print("\n-TRAINING STEP COMPLETE-\n")
            i += 1
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)

    print("\n\nRETRAINING OF TEN BEST MODEL HYPER-PARAMETERS COMPLETE")
