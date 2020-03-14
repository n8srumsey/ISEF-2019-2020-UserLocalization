"""
videoSplitter.py
~~~~~~~~~~~~~~~~~~

A module to split the frames of the source videos into individual images.

Code for splitting videos into frames is adapted from Geeks for Geeks website:
https://www.geeksforgeeks.org/extract-images-from-video-in-python/
"""

# Importing all necessary libraries
import cv2
import os

# Sets the directory of source folder of .mp4 videos to split into individual images.
directory_in_str = 'C:\\Users\\Wildcat\\ISEF20192020\\data\\SourceVideoLowResolutionMP4'
directory = os.fsencode(directory_in_str)

# Initializes 'cam' variable which will store each video
cam = cv2.VideoCapture(
    'C:\\Users\\Wildcat\\ISEF20192020\\data\\SourceVideoLowResolution\\X0Y0.mp4')

"""
Splits videos (.mp4) into frames given the input of the file name.

It takes the argument of the name of a file (string). It does not return any value. 

It saves all images in a folder labeled 'data' into the same directory as this python script (videoSplitter.py)
"""


def split_frames(name_file):
    # Read the video from specified path
    cam = cv2.VideoCapture(directory_in_str + "\\" + name_file)

    try:
        # creating a folder named data; if doesn't exist, create it
        if not os.path.exists('data'):
            os.makedirs('data')

        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')

    # frame of video
    currentframe = 0

    # loop to iterate through each frame of video and save images
    while True:
        # reading from frame
        ret, frame = cam.read()
        name_of_file = name_file[0:len(name_file) - 4]

        if ret:
            # if there is video still left, continue creating images
            name = name_of_file + '(' + str(currentframe) + ').jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will show how many frames are created
            currentframe += 1

        # if there are not any frames of video left, break
        else:
            break


# iterates through each .mp4 file of specified directory and splits each by frame
for file in os.listdir(directory):
    filename = os.fsdecode(file)

    # if file is a .mp4 file
    if filename.endswith(".mp4"):
        split_frames(filename)
        continue

    # if file is not a .mp4 file
    else:
        continue

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
