"""
datasetCoordinateInfoGatherer.py
~~~~~~~~~~~~~~~~~~

A module to assist in organizing files in a single directory into a dataset organized for image processing in Keras.
"""
import os

director_in_string = 'C:\\Users\\natha\\Documents\\Nate ISEF 2019-2020\\Source\\Source Images\\'


def return_num_in_category(directory_string):
    amount_each_coordinate = {}
    directory = os.fsencode(directory_string)
    for file in os.listdir(directory):
        filename = str(os.fsdecode(file))
        index_end_coordinate_name = filename.find("(")
        file_coordinate = filename[:index_end_coordinate_name]
        if file_coordinate not in amount_each_coordinate:
            amount_each_coordinate.update({file_coordinate: 1})
        else:
            previous_total = amount_each_coordinate.get(file_coordinate)
            amount_each_coordinate.update({file_coordinate: (previous_total + 1)})

    return amount_each_coordinate


def return_list_categories(directory_string):
    list_coordinates = []
    directory = os.fsencode(directory_string)
    for file in os.listdir(directory):
        filename = str(os.fsdecode(file))
        index_end_coordinate_name = filename.find("(")
        file_coordinate = filename[:index_end_coordinate_name]
        if file_coordinate not in list_coordinates:
            list_coordinates.append(file_coordinate)

    return list_coordinates


print(return_num_in_category(director_in_string))
print(return_list_categories(director_in_string))
