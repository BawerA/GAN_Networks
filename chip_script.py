#!/usr/bin/env python
# coding: utf-8

from keras.models import load_model
import os
import keras
from keras.layers import Dense, Input, Reshape, Flatten, Dropout
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (Conv2D, Input, Reshape, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout)
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image, ImageDraw
import skimage.filters as filters
import io
import glob
from tqdm import tqdm
import logging
import argparse
import os
import json
import wv_util as wv
#import tfr_util as tfr
#import aug_util as aug
import csv
import random
import six
import cv2
import glob
import mahotas as mh
#from chainer.dataset import dataset_mixin
from tqdm import tqdm
import time



def get_labels(fname="xView_train.geojson"):
    with open(fname) as f:
        data = json.load(f)
    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))    
    for i in range(len(data['features'])):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            coords[i] = val
        else:
            chips[i] = 'None'    
    return coords, chips, classes


def get_chips():
    print("\nStarting to chip the xView dataset.\n")
    thechips = []
    theclasses = []
    thecoords = []
    thecoords, thechips, theclasses = get_labels()
    per = 1
    X_data = []
    files2 = glob.glob ("/root/Desktop/seniordesign/chippedimages/*.tif")#                     Change this to ur own directory
    files = glob.glob ("/root/Desktop/seniordesign/testing_images/*.tif")#                     Change this to ur own directory
    for myFile in files:
        t = 0
        print('\nChipping image at this location: ', myFile)
        image = cv2.imread (myFile)
        #X_data.append (image) #                                                         https://stackoverflow.com/questions/37747021/create-numpy-array-of-images
        chipped_img, chipped_box, chipped_classes = wv.chip_image(img = image, coords = thecoords, classes=theclasses, shape=(224,224))
        numberOfChips = chipped_img.shape[0]
        print("This image created %d chips." % chipped_img.shape[0]) 
        while t < numberOfChips:
            #print(t + 1)
            os.chdir(r"/root/Desktop/seniordesign/chippedimages") #       Change this to ur own directory
            mh.imsave('%d.tif' % per, chipped_img[t])
            os.chdir(r"/root/Desktop/seniordesign") #                     Change this to ur own directory
            t += 1
            per += 1

    os.chdir(r"/root/Desktop/seniordesign/chippedimages") #        Change this to ur own directory
    for myFile in files2:
        chipimage = mh.imread(myFile)
        X_data.append(chipimage)


    npchipped = np.array([np.array(Image.open(myFile)) for myFile in files2]) ### This puts all of the images in to one nparray, ( i think the block of code above does the same thing)
#                                                                   https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array

    npchipped2 = np.array(X_data)  #  nchipped2 is the numpy array, I use it down below 
    #npchipped and npchipped are the same
    return npchipped2


