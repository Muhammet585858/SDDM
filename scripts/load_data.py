# -*- coding: utf-8 -*-
"""
Script for loading the data.
Contains functions LoadTrainBatch and LoadTestBatch.

Adjusted from "driving_data.py" in github/autopilot
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import random
from scipy import ndimage
import cv2
import csv

# image file names
images = []
# steering angles
angles = []

# points to the end of the last batch
train_batch_pointer = 0
test_batch_pointer = 0

# read data.txt
# with open("data/data.txt") as f:
#     for line in f:
#         image, angle = line.split()
#         images.append("data/" + image)
#         # steering wheel angle in radians
#         angles.append(float(angle) * scipy.pi / 180)

with open("/mnt/data1/self-driving-car/datasets/larger/output/interpolated.csv") as f:
    reader = csv.reader(f, delimiter=",")
    for line in reader:
        if line[4] == "center_camera":
            image, angle = line[5], line[6]
            images.append("/mnt/data1/self-driving-car/datasets/larger/output/" + str(image))
            angles.append(angle)

# shuffle images and angles
temp_tuples = list(zip(images, angles))
random.shuffle(temp_tuples)
images, angles = zip(*temp_tuples)

# get number of images
data_size = len(images)
print data_size

# use the first 80% as training data
train_images = images[:int(data_size * 0.8)]
train_angles = angles[:int(data_size * 0.8)]

# use the last 20% as test data
test_images = images[-int(data_size * 0.2):]
test_angles = angles[-int(data_size * 0.2):]

# get number of images in training and test sets
train_images_size = len(train_images)
test_images_size = len(test_images)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    images_out = []
    angles_out = []
    for i in range(batch_size):
        # read image and cut off top 150px (horizon)
        img = scipy.misc.imread(train_images[(train_batch_pointer + i) % train_images_size])[-150:]
        # resize image (66x200 px)
        img2 = cv2.resize(img.astype('float32'), dsize=(200,66))
        # normalize [0,1] and save
        images_out.append(img2/255.0)
        # images_out.append(scipy.misc.imresize(scipy.misc.imread(train_images[(train_batch_pointer + i) % train_images_size])[-150:], [66, 200]) / 255.0)
        # i used cv2.resize instead of the schipy function because this is more
        # reliable according to this post https://github.com/scipy/scipy/issues/4458
        angles_out.append([train_angles[(train_batch_pointer + i) % train_images_size]])
    train_batch_pointer += batch_size
    return images_out, angles_out

LoadTrainBatch(1)

def LoadTestBatch(batch_size):
    global test_batch_pointer
    images_out = []
    angles_out = []
    for i in range(batch_size):
        img = scipy.misc.imread(test_images[(test_batch_pointer + i) % test_images_size])[-150:]
        images_out.append((cv2.resize(img.astype('float32'), dsize=(200,66)))/255.0)
        angles_out.append([test_angles[(test_batch_pointer + i) % test_images_size]])
    test_batch_pointer += batch_size
    return images_out, angles_out
