"""
neural_net.py
~~~~~~~~~~~~~~~~~~
Contains functions to build and train a keras CNN based on passed hyperparameter arguments.

This is based off of Vooban's demonstration repo @ https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100
"""
import os
import uuid
import keras
from hyperopt import STATUS_OK
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout, SpatialDropout2D, Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import K  # import keras.backend as K
from keras.optimizers import Adam, Nadam, RMSprop
from keras_preprocessing.image import ImageDataGenerator
from utils import print_json

# Set directory to save model weights to
WEIGHTS_DIR = "weights/"

"""Setup data generator"""
# resize images to be reduce number of parameters, to increase speed of training
dataset_input_resize = (192, 108)
dataset_input_shape = (192, 108, 3)
# instantiate ImageDataGenerator
datagen = ImageDataGenerator(rescale=1. / 225, rotation_range=10)
train_it = datagen.flow_from_directory('data/train/', class_mode='categorical', target_size=dataset_input_resize,
                                       batch_size=16, shuffle=True)
val_it = datagen.flow_from_directory('data/validation/', class_mode='categorical', target_size=dataset_input_resize,
                                     batch_size=16)
test_it = datagen.flow_from_directory('data/test/', class_mode='categorical', target_size=dataset_input_resize,
                                      batch_size=16)
# Define number of classes to
num_classes = 103

# Set training constants
EPOCHS = 3

OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}

"""
list_coordinate_names = [dI for dI in os.listdir('data/train') if os.path.isdir(os.path.join('data/train', dI))]
coordinate_tuple_names = []
for coordinate in list_coordinate_names:
    index_y = coordinate.find('Y')
    x_value = int(coordinate[1:index_y])
    y_value = int(coordinate[(index_y+1):])
    coordinate_tuple_names.append((x_value, y_value))
"""


def build_and_train(hype_space, save_best_weights=False):
    """Build the deep CNN model and train it."""
    # setup Keras to learning phase - learn
    K.set_learning_phase(1)
    K.set_image_data_format('channels_last')

    # Build the model according to the hyper-parameter space passed.
    model = build_model(hype_space)

    # Set model_uuid
    model_uuid = str(uuid.uuid4())[:5]

    # Create callbacks list to add to as according to constructor parameters
    callbacks = []

    # Weight saving callback:
    if save_best_weights:
        weights_save_path = os.path.join(
            WEIGHTS_DIR, '{}.hdf5'.format(model_uuid))
        print("Model's weights will be saved to: {}".format(weights_save_path))
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

        # Add weights saving callback to model's callbacks
        callbacks.append(keras.callbacks.ModelCheckpoint(
            weights_save_path,
            monitor='val_accuracy',
            save_best_only=True, mode='max'))

    # Train net:
    print("\nBegin training of model:")
    history = model.fit_generator(
        train_it, validation_data=val_it,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
    ).history

    # Test net:
    print("\nBegin evaluation of model:")
    K.set_learning_phase(0)
    score = model.evaluate_generator(test_it, verbose=1)
    max_acc = max(history['accuracy'])

    # fixme: model predict to return 'distance' metric goes here

    # Define model name
    model_name = "model_{}_{}".format(str(max_acc), model_uuid)
    print("Model name: {}".format(model_name))

    # Note: to restore the model, you'll need to have a keras callback to
    # save the best weights and not the final weights. Only the result is
    # saved.
    print(history.keys())
    print(history)
    print(score)
    result = {
        # We plug "-accuracy" as a
        # minimizing metric named 'loss' by Hyperopt.
        'loss': -history['val_accuracy'][-1],  # fixme | is this the loss metric I really want?
        'real_loss': score[0],
        # Stats:
        'best_loss': min(history['loss']),
        'best_accuracy': max(history['accuracy']),
        'end_loss': score[0],
        'end_accuracy': score[1],
        # Misc:
        'model_name': model_name,
        'model_uuid': model_uuid,
        'space': hype_space,
        'history': history,
        'status': STATUS_OK
    }
    print("\nRESULT:")
    print_json(result)

    return model, model_name, result, model_uuid


def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)
    model = Sequential()

    # Define parameters to goverrn construction of model according to hype_space
    n_filters = int(round(hype_space['nb_conv_filters']))
    if hype_space['nb_conv_in_conv_pool_layers'] == 2:
        two_conv_layers = True
    else:
        two_conv_layers = False

    # first conv+pool layer
    model.add(first_convolution(n_filters, hype_space))
    if hype_space['conv_dropout'] is not None:
        model.add(conv_dropout(hype_space))
    if two_conv_layers:
        model.add(convolution(n_filters, hype_space))
        if hype_space['conv_dropout'] is not None:
            model.add(conv_dropout(hype_space))
    model.add(pooling(hype_space))

    # adds additional conv+pool layers based on hype_space
    for _ in range(hype_space['nb_conv_pool_layers'] - 1):
        model.add(convolution(n_filters, hype_space))
        if hype_space['conv_dropout'] is not None:
            model.add(conv_dropout(hype_space))
        if two_conv_layers:
            model.add(convolution(n_filters, hype_space))
            if hype_space['conv_dropout'] is not None:
                model.add(conv_dropout(hype_space))
        model.add(pooling(hype_space))

    # fully connected layers
    model.add(Flatten())

    model.add(keras.layers.core.Dense(
        units=int(round(hype_space['fc_nodes_1'])),
        activation=hype_space['activation'],
        kernel_regularizer=keras.regularizers.l2(hype_space['l2_weight_reg'])
    ))
    model.add(fc_dropout(hype_space))

    if hype_space['fc_second_layer'] is not None:
        model.add(keras.layers.core.Dense(
            units=int(round(hype_space['fc_second_layer'])),
            activation=hype_space['activation'],
            kernel_regularizer=keras.regularizers.l2(hype_space['l2_weight_reg'])))
        model.add(fc_dropout(hype_space))

    model.add(Dense(num_classes, activation='softmax'))

    # Finalize and compile model:
    model.compile(
        optimizer=OPTIMIZER_STR_TO_CLASS[hype_space['optimizer']](
            lr=hype_space['lr_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def fc_dropout(hype_space):
    """Add dropout after a layer."""
    return Dropout(rate=hype_space['fc_dropout_proba'])


def conv_dropout(hype_space):
    """Add dropout after a layer."""
    return SpatialDropout2D(rate=hype_space['conv_dropout'])


def convolution(n_filters, hype_space):
    """Basic convolution layer, parametrized by the hype_space."""
    k = int(round(hype_space['conv_kernel_size']))
    return Conv2D(
        filters=n_filters, kernel_size=(k, k), strides=(1, 1),
        padding='same', activation=hype_space['activation'],
        kernel_regularizer=keras.regularizers.l2(hype_space['l2_weight_reg']))


def first_convolution(n_filters, hype_space):
    """Basic convolution layer, parametrized by the hype_space, with input shape defined."""
    k = int(round(hype_space['conv_kernel_size']))
    return Conv2D(
        filters=n_filters, kernel_size=(k, k), strides=(1, 1),
        padding='same', activation=hype_space['activation'],
        kernel_regularizer=keras.regularizers.l2(hype_space['l2_weight_reg']),
        input_shape=dataset_input_shape)


def pooling(hype_space):
    """Deal with pooling in convolution steps."""
    if hype_space['pooling_type'] == 'avg':
        return AveragePooling2D(pool_size=(2, 2))

    else:  # 'max'
        return MaxPooling2D(pool_size=(2, 2))


"""
def metric_distance(y_true, y_pred):
    print(K.int_shape(y_true))
    print(K.int_shape(y_pred))
    pred_index_tensor = K.argmax(y_pred, axis=-1)
    true_index_tensor = K.argmax(y_true, axis=-1)

    with tf.compat.v1.keras.backend.get_session().as_default():
        pred_index = pred_index_tensor.eval()
        true_index = true_index_tensor.eval()

    metric_distance_list = []
    for prediction, truth in pred_index, true_index:
        pred_x = coordinate_tuple_names[prediction][0]
        pred_y = coordinate_tuple_names[prediction][1]
        true_x = coordinate_tuple_names[truth][0]
        true_y = coordinate_tuple_names[truth][1]
        metric_distance_list.append(math.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2))
    metric_distance_numpy = numpy.asarray(metric_distance_list).transpose()

    with tf.compat.v1.keras.backend.get_session().as_default():
        tensor = K.constant(metric_distance_numpy)

    return tensor
"""
