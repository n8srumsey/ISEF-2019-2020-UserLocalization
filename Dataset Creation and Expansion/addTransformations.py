"""
addTransformations.py
~~~~~~~~~~~~~~~~~~

A module to add transformations to images in order to artificially expand the dataset.

Methodology is based off of J. Brownlee's machine-learning blog: Machine Learning Mastery
https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
"""
import os

from PIL import Image
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# load the directory
director_in_string = 'C:\\Users\\natha\\Documents\\Nate ISEF 2019-2020\\Source\\Source Images\\'
target_dir_str = "C:\\Users\\natha\\Documents\\Nate ISEF 2019-2020\\data\\Transformed Images\\"

def transform_directory(directory_string, target_dir, rotation=15):
    n = 0
    directory = os.fsencode(directory_string)
    for file in os.listdir(directory):
        n += 1
        print("Transformed file  " + str(n) + "/186,534")

        filename = str(os.fsdecode(file))
        img = load_img(director_in_string + filename)

        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator

        # data_gen = ImageDataGenerator(width_shift_range=[-15, 15])
        data_gen = ImageDataGenerator(rotation_range=rotation)

        # prepare iterator
        it = data_gen.flow(samples, batch_size=1)
        # generate samples and plot
        for i in range(4):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            image2 = Image.fromarray(image)
            image2_filename = target_dir + filename[:-4] + '(' + str(i) + ')'
            image2.save(image2_filename + ".jpg")


transform_directory(director_in_string, target_dir_str, 15)
