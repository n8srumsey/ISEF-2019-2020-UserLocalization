## ISEF-2019-2020-UserLocalization

This repository contains the code developed and used for my 2019-2020 ISEF Project.

# Abstract
Indoor localization has become a burgeoning field of study. Visual localization offers advantages over
other
solutions in that it does not require substantial infrastructure, and can be employed using readily
available technologies, such as the camera interface on a user's smartphone. I developed an image
-
based localization algorithm employing a Convolutional Neu
ral Network (CNN) to classify the location of
images in an indoor space. I evaluated CNN performance on training, validation, and testing datasets to
determine if overfitting occurred. Overfitting indicates that a CNN performs well on a training data set
b
ut has difficulty in classifying location in practical applications. The hyperparameters, which govern the
training process of the CNN, were tuned with Bayesian Optimization to find the hyperparameter values
which maximized CNN localization accuracy. The s
earch for the best hyperparameter values was divided
into two phases: (1) the optimization phase, in which the Bayesian Optimization algorithm selects the
most promising hyperparameters, evaluates the performance of CNN built with those hyperparameters,
an
d then updates future selections according to the previous hyperparameters' performance; and (2)
the retraining phase, where the ten highest
-
accuracy CNNs were retrained over additional epochs, as
compared to the optimization phase, to maximize their perf
ormances. The highest
-
accuracy CNN of the
retraining phase achieved a location classification accuracy of 99.63%, convincingly showing that image
localization with a Convolutional Neural Network and Bayesian Optimization hyperparameter tuning can
localize
in an indoor environment with a high level of accuracy on par with other state
-
of
-
the
-
art
localization solutions.
