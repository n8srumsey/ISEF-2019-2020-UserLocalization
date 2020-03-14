"""
datasetOrganizer.py
~~~~~~~~~~~~~~~~~~

A module to organize files in a single directory into a dataset organized for image processing in Keras.
"""

import os
from PIL import Image

directory_in_string = 'C:\\Users\\natha\\Documents\\Nate ISEF 2019-2020\\Source\\Source Images\\'
target_directory_in_string = 'C:\\Users\\natha\\Documents\\Nate ISEF 2019-2020\\' \
                             'ISEF-2019-2020-UserLocalization-Preparation\\data\\'


def return_num_in_category(directory_string):
    amount_each_coordinate = {}
    directory = os.fsencode(directory_string)
    for file1 in os.listdir(directory):
        filename1 = str(os.fsdecode(file1))
        index_end_coordinate_name1 = filename1.find("(")
        file_coordinate1 = filename1[:index_end_coordinate_name1]
        if file_coordinate1 not in amount_each_coordinate:
            amount_each_coordinate.update({file_coordinate1: 1})
        else:
            previous_total = amount_each_coordinate.get(file_coordinate1)
            amount_each_coordinate.update({file_coordinate1: (previous_total + 1)})

    return amount_each_coordinate


def return_list_categories(directory_string):
    list_coordinates = []
    directory = os.fsencode(directory_string)
    for file1 in os.listdir(directory):
        filename1 = str(os.fsdecode(file1))
        index_end_coordinate_name1 = filename1.find("(")
        file_coordinate1 = filename1[:index_end_coordinate_name1]
        if file_coordinate1 not in list_coordinates:
            list_coordinates.append(file_coordinate1)

    return list_coordinates


list_of_coordinates = return_list_categories(directory_in_string)
dict_coordinate_amounts = return_num_in_category(directory_in_string)

source_directory = os.fsencode(directory_in_string)

for nameCoordinate in list_of_coordinates:
    os.mkdir(target_directory_in_string + "train\\" + nameCoordinate + "\\")
    os.mkdir(target_directory_in_string + "validation\\" + nameCoordinate + "\\")
    os.mkdir(target_directory_in_string + "test\\" + nameCoordinate + "\\")

for coordinateName in list_of_coordinates:
    total_images = dict_coordinate_amounts.get(coordinateName)
    i = 0
    j = 0
    print("Starting organization for " + coordinateName)
    for file in os.listdir(source_directory):
        filename = str(os.fsdecode(file))
        index_end_coordinate_name = filename.find("(")
        file_coordinate = filename[:index_end_coordinate_name]

        if file_coordinate == coordinateName:
            if j == 5:
                j = 0
            i += 1
            j += 1
            print("    - [" + file_coordinate + "] Finished file " + str(i) + " out of " + str(total_images))
            if j == 1 or j == 3 or j == 5:
                image = Image.open(directory_in_string + filename)
                image.save(target_directory_in_string + "train\\" + file_coordinate + "\\" + filename)
            else:
                if j == 2:
                    image = Image.open(directory_in_string + filename)
                    image.save(target_directory_in_string + "validation\\" + file_coordinate + "\\" + filename)
                else:
                    image = Image.open(directory_in_string + filename)
                    image.save(target_directory_in_string + "test\\" + file_coordinate + "\\" + filename)
