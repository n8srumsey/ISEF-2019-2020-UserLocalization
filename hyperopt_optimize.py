"""
hyperopt_optimize.py
~~~~~~~~~~~~~~~~~~
Optimizes the specified neural_net model using hyperopt, based off of the designated search space.

This is based off of Vooban's demonstration repo @ https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100
"""

import pickle
import traceback

import keras.backend as K
from hyperopt import hp, tpe, fmin, Trials, STATUS_FAIL
from keras.utils import plot_model

from neural_net import build_and_train, build_model
from utils import print_json, save_json_result, load_best_hyperspace

space = {
    # Select learning rate
    'lr_rate': hp.loguniform('lr_rate', -9.8, -3.0),
    # L2 regularization weight decay:
    'l2_weight_reg': hp.loguniform('l2_weight_reg', -8.6, -6.0),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Activations that are used everywhere
    'activation': hp.choice('activation', ['relu', 'elu']),

    # Select the number of filters that each convolutional layer will have
    'nb_conv_filters': hp.qloguniform('nb_conv_filters', 2.0, 3.5, 1),
    # The kernel_size for convolutions:
    'conv_kernel_size': hp.choice('conv_kernel_size', [3, 5, 7]),
    # Number of conv+pool layers stacked:
    'nb_conv_pool_layers': hp.choice('nb_conv_pool_layers', [2, 3]),
    # Number of convolutional layers in conv+pool layers:
    'nb_conv_in_conv_pool_layers': hp.choice('nb_conv_in_conv_pool_layers', [1, 2]),
    # Use SpatialDropout2D after conv layers (and if so, what rate)
    'conv_dropout': hp.choice('conv_dropout', [None, hp.uniform('conv_dropout_probability', 0.0, 0.5)]),
    # The type of pooling used at each sub-sampling step:
    'pooling_type': hp.choice('pooling_type', [
        'max',  # Max pooling
        'avg',  # Average pooling
    ]),

    # Uniform distribution in finding appropriate dropout values for FC layers
    'fc_dropout_proba': hp.uniform('fc_dropout_proba', 0.0, 0.5),
    # Amount of fully-connected units after convolution feature map
    'fc_nodes_1': hp.qloguniform('fc_nodes_1', 4.8, 7.0, 1),
    # Use one more FC layer before output
    'fc_second_layer': hp.choice(
        'fc_second_layer', [None, hp.qloguniform('fc_nodes_2', 4.8, 7.0, 1)]
    )
}


def plot(hyperspace, file_name_prefix):
    """Plot a model from it's hyperspace."""
    model = build_model(hyperspace)
    plot_model(
        model,
        to_file='model-visualizations/{}.png'.format(file_name_prefix),
        show_shapes=True
    )
    print("Saved model visualization to model-visualizations/{}.png.".format(file_name_prefix))
    K.clear_session()
    del model


def plot_best_model():
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace()
    if space_best_model is None:
        print("No best model to plot. Continuing...")
        return

    # Print best hyperspace and save model png
    print("Best hyperspace yet:")
    print_json(space_best_model)
    plot(space_best_model, "model_best")


def optimize_cnn(hype_space):
    """Build a convolutional neural network and train it."""
    try:
        model, model_name, result, model_uuid = build_and_train(hype_space, save_best_weights=True)

        # Save training results to disks with unique filenames
        save_json_result(model_name, result)

        # Save .png plot of the model according to its hyper-parameters
        plot(result['space'], model_uuid)

        K.clear_session()
        del model

        return result

    except Exception as err:
        try:
            K.clear_session()
        except:
            pass
        err_str = str(err)
        print(err_str)
        traceback_str = str(traceback.format_exc())
        print(traceback_str)
        print("\n\n")
        return {
            'status': STATUS_FAIL,
            'err': err_str,
            'traceback': traceback_str
        }


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    print("Attempting to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("trials_history.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new Trials().")

    best = fmin(
        optimize_cnn,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("trials_history.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")


if __name__ == "__main__":
    """Run the optimisation forever (and saves results)."""

    print("Will train many models, one after the other.")

    print("\nResults will be saved in the folder named 'results/'. "
          "Results are continuously saved into a 'trials_history.pkl'"
          "file, too. Re-running optimize.py will resume the "
          "meta-optimization.\n")

    while True:

        # Optimize a new model with the TPE Algorithm:
        print("\n\nOPTIMIZING NEW MODEL:")
        try:
            run_a_trial()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)

        # Re-plot best model since it may have changed:
        print("PLOTTING BEST MODEL:")
        plot_best_model()
