#!/usr/bin/env python
# coding: utf-8

#Testing chipping part#!/usr/bin/env python
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
from numpy import random
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
    files2 = glob.glob ("/media/root/New Volume/chipped_images1/*.tif")#                     Change this to ur own directory
    files = glob.glob ("/media/root/New Volume/*.tif")#                     Change this to ur own directory
    for myFile in files:
        t = 0
        print('\nChipping image at this location: ', myFile)
        image = cv2.imread (myFile)
        #X_data.append (image) #                                                         https://stackoverflow.com/questions/37747021/create-numpy-array-of-images
        chipped_img, chipped_box, chipped_classes = wv.chip_image(img = image, coords = thecoords, classes=theclasses, shape=(256,256))
        numberOfChips = chipped_img.shape[0]
        print("This image created %d chips." % chipped_img.shape[0]) 
        while t < numberOfChips:
            #print(t + 1)
            os.chdir(r"/media/root/New Volume/chipped_images1") #       Change this to ur own directory
            mh.imsave('%d.tif' % per, chipped_img[t])
            os.chdir(r"/media/root/New Volume") #                     Change this to ur own directory
            t += 1
            per += 1

    os.chdir(r"/media/root/New Volume/chipped_images1") #        Change this to ur own directory
    for myFile in files2:
        chipimage = mh.imread(myFile)
        X_data.append(chipimage)


    npchipped = np.array([np.array(Image.open(myFile)) for myFile in files2]) ### This puts all of the images in to one nparray, ( i think the block of code above does the same thing)
#                                                                   https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array

    npchipped2 = np.array(X_data)  #  nchipped2 is the numpy array, I use it down below 
    #npchipped and npchipped are the same
    return npchipped2, numberOfChips


def chip_images():
    print("\nStarting to chip the xView dataset.\n")
    thechips = []
    theclasses = []
    thecoords = []
    thecoords, thechips, theclasses = get_labels()
    per = 1
    X_data = []
    files2 = glob.glob ("/media/root/New Volume/chipped_images1/*.tif")#                     Change this to ur own directory
    files = glob.glob ("/media/root/New Volume/testing_images/*.tif")#                     Change this to ur own directory
    for myFile in files:
        t = 0
        print('\nChipping image at this location: ', myFile)
        image = cv2.imread (myFile)
        #X_data.append (image) #                                                         https://stackoverflow.com/questions/37747021/create-numpy-array-of-images
        chipped_img, chipped_box, chipped_classes = wv.chip_image(img = image, coords = thecoords, classes=theclasses, shape=(256,256))
        numberOfChips = chipped_img.shape[0]
        print("This image created %d chips." % chipped_img.shape[0]) 
        while t < numberOfChips:
            #print(t + 1)
            os.chdir(r"/media/root/New Volume/chipped_images1") #       Change this to ur own directory
            mh.imsave('%d.tif' % per, chipped_img[t])
            os.chdir(r"/media/root/New Volume") #                     Change this to ur own directory
            t += 1
            per += 1




def get_numpy():
    os.chdir(r"/media/root/New Volume/chipped_images1")
    files2 = glob.glob ("/media/root/New Volume/chipped_images1/*.tif")# #        Change this to ur own directory
    image_data = []
    for myFile in files2:
        chipimage = mh.imread(myFile)
        image_data.append(chipimage)


    npchipped = np.array([np.array(Image.open(myFile)) for myFile in files2]) ### This puts all of the images in to one nparray, ( i think the block of code above does the same thing)
#                                                                   https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array
    npchipped2 = np.array(image_data) 
    numberOfChips = len(image_data) 
    os.chdir(r"/media/root/New Volume")
    #numpy_data = TemporaryFile()
    np.save('numpy_data', npchipped2)
    print("Npz saved. Nice job homie")
    #np_data = np.load('numpy_data.npy')
    return numberOfChips

def get_array():
    print("Getting the numpy array...")




def numpyify(batch_size):
    os.chdir(r"/media/root/New Volume/carlos")   
    path1 = glob.glob("/media/root/New Volume/carlos/folder1/*.tif")
    path2 = glob.glob("/media/root/New Volume/carlos/folder2/*.tif")
    path3 = glob.glob("/media/root/New Volume/carlos/folder3/*.tif")
    path4 = glob.glob("/media/root/New Volume/carlos/folder4/*.tif")
    path5 = glob.glob("/media/root/New Volume/carlos/folder5/*.tif")
    path6 = glob.glob("/media/root/New Volume/carlos/folder6/*.tif")
    path7 = glob.glob("/media/root/New Volume/carlos/folder7/*.tif")
    path8 = glob.glob("/media/root/New Volume/carlos/folder8/*.tif")
    path9 = glob.glob("/media/root/New Volume/carlos/folder9/*.tif")
    path10 = glob.glob("/media/root/New Volume/carlos/folder10/*.tif")
    path11 = glob.glob("/media/root/New Volume/carlos/folder11/*.tif")
    path12 = glob.glob("/media/root/New Volume/carlos/folder12/*.tif") 
    path_array = [path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11, path12]
    image_data = []
    x = 0 
    path_choice = random.choice(path_array)
    for x in range(x, batch_size):
        chipimage = mh.imread(random.choice(path_choice))
        print(random.choice(path_choice))
        image_data.append(chipimage)
    npchipped2 = np.array(image_data) 
    numberOfChips = len(image_data) 
    return npchipped2






#path5 = random.choice(os.listdir(r"/media/root/New Volume/carlos/folder2"))

'''
def path_file():
    x = 0
    y = 1
    z = 2
    path1 = x
    path2 = y
    path3 = z
    files1 = glob.glob("/media/root/New Volume/carlos/folder2/*.tif")
    path_array = [path1, path2, path3]
    for i in range(0,5):
        screen_print = mh.imread(random.choice(files1))
        print(screen_print)
path_file()
'''



'''
def numpyify(batch_size):
    os.chdir(r"/root/Desktop/seniordesign/chippedimages")
    files2 = glob.glob ("/root/Desktop/seniordesign/chippedimages/*.tif")# #        Change this to ur own directory
    image_data = []
    x = 0
    for x in range(x, batch_size):
        chipimage = mh.imread(random.choice(files2))
        print(random.choice(files2))
        image_data.append(chipimage)
    npchipped2 = np.array(image_data)  
    numpyify_array = random.choices(npchipped2, k=batch_size)
    return numpyify_array

    '''

#get_numpy()
#chip_images()
#print('\n\nChipping comeplete.\n')




