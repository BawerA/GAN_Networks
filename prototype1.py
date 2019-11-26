#!/usr/bin/env python
# coding: utf-8

from keras.models import load_model
import os
import tensorflow.keras
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Dropout
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (Conv2D, Input, Reshape, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout)
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
import time
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
import chipchip as chip_

start = time.time()
#######################################
xview_numpy_array = chip_.get_chips() #   <<<<< Our beautiful numpy array *********
#######################################
#rank_check = tf.zeros(xview_numpy_array)
#rank_check = tf.rank(xview_numpy_array)
#print("\nThe rank of our numpy tensor: \n", tf.rank(xview_numpy_array))
print("\nThe rank of our numpy array:  ", xview_numpy_array.ndim)#I'm pretty sure this is the rank*

#print("\n \n", xview_numpy_array)   << This will print the weights of the numpy array

print("\n Our numpy info: (Batch size(number of chips), width, height, channel) of the numpy array: \n", xview_numpy_array.shape)
#The first number in the parenthesis is the batch size. The batch size is just the number of chips in the folder
#The second and third values represent the dimensions of the chipped images
#The fourth value represents the channels of the images. 3 stands for RGB or color images. If it was 1, then that means the images are greyscale.


def build_generator():
    model = models.Sequential()
    #First Layer:  (Convolutional Layer)
    #model.add(LeakyReLU(0.2))
    #model.add(BatchNormalization())
    #model.add(Reshape(56,56,112))
    model.add(Conv2D(filters= 32, kernel_size= (5,5), data_format='channels_last', input_shape=(224, 224, 3), padding='same'))
    model.add(BatchNormalization())#Makes the training a little easier, standardizes the activations from a prior layer to have a zero mean and unit variance. 
    """
    For Conv2D()'s parameters, use this link >>>>>>>>>>> https://missinglink.ai/guides/keras/keras-conv2d-working-cnn-2d-convolutions-keras/#section4
    Things online say to start the filter with 32 and then double it each time Conv2D() is called (Example: next would be 64, then 128, 256)
    kernel size is the area of pixels that is scanned, this link explains it more >>> https://missinglink.ai/guides/keras/keras-conv2d-working-cnn-2d-convolutions-keras/
    Fix the filters ^^^^^^^^^^^^^^^^ causing error i think< I this this might be fixed now
    """

    #  Below is the Pooling Layer
    model.add(MaxPooling2D())
    model.add(UpSampling2D())
    #  Another convolutional layer
    model.add(Conv2D(filters= 64, kernel_size= (5,5), padding='same'))
    model.add(MaxPooling2D())
    model.add(UpSampling2D())

    #  Below is the Fully-Connected or Core Layer
    #model.add(Flatten())
    model.add(Dropout(0.5))

    #    Uncomment  line 75 and 76
    
    model.add(Conv2D(3, kernel_size= (5,5), padding='same'))

    print("\n\t Generator's Layer information:\n")
    model.summary()
    return model

def build_discriminator():
    model = models.Sequential()
    #model.add(Reshape((224,) ,input_shape=(224,224)))
    model.add(Conv2D(filters= 32, kernel_size= (5,5), data_format='channels_last',input_shape=(224, 224, 3),  padding='same'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(0.2))
    #Maybe change the filter size to the last put filter, dunno yet
    model.add(MaxPooling2D())
    model.add(UpSampling2D())
    #model.add(Dropout(0.5))
    #model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same'))
    #model.add(LeakyReLU(0.2))
    #model.add(Dropout(0.5))
    #model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same'))
    #model.add(LeakyReLU(0.2))
    #                                     Uncomment every layer above this line
    """model.add(Flatten())
    model.add(Dropout(0.5))    
    model.add(UpSampling2D())
    model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same'))"""
    #model.add(UpSampling2D())
    # Look at this link:
    #  https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
   # tf.keras.backend.expand_dims(model, axis=-1)

    print("\n\tDisrciminator's Layer information:\n")
    model.summary()
    return model




def build_combined(generator, discriminator):
    model = models.Sequential()
    model.add(generator)        
    discriminator.trainable = False#True
    model.add(discriminator)
    #input_layer = Input((3,))
    #en_image = generator(input_layer)
    #gen_image = generator()    
    #dis_output = discriminator(gen_image)
    #model = models.Model(input_layer, dis_output)
    print("\n\n\tCombined Model Summary: \n\n")
    model.summary()
    return model

def prepare_images_for_disc(images):   #Prob will get rid of this method
    # convert from unit8 to float32
    images = images.astype('float32')
    # scale from [0,255] to [-1,1]
    images = (images - 127.5) / 127.5
    return images




#Build the stuff here:
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
combined_model = build_combined(generator, discriminator)                          
combined_model.compile(optimizer='adam', loss='binary_crossentropy') 
#model.compile(optimizer='adam', )

            
scaled_numpy = prepare_images_for_disc(xview_numpy_array)
print("\nScaled numpy: \n", scaled_numpy.shape)


#Training Here:
batch_size = 50
noise = np.random.uniform(-1.0, 1.0, size=[516,224,224,3])


generated_images = generator.predict(noise)
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for i in range(0, xview_numpy_array.shape[0], batch_size):
    #fake = generator.predict(noise)
    real = xview_numpy_array[i:i+batch_size].reshape(-1, 224, 224, 3)

#Tensorboard and other plots:




"""
Links that I used:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ndim.html
https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
https://stackoverflow.com/questions/43743593/keras-how-to-get-layer-shapes-in-a-sequential-model
https://www.tensorflow.org/guide/tensor
https://www.tensorflow.org/api_docs/python/tf/rank
https://stackoverflow.com/questions/57893542/convolution2d-depth-of-input-is-not-a-multiple-of-input-depth-of-filter
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py#L138



 "ValueError: Input tensor must be of rank 3, 4 or 5 but was 2."
https://www.tensorflow.org/api_docs/python/tf/rank   <<<<<<<<<<<<<<<<<<<<<<<

https://www.tensorflow.org/guide/tensor****************************

https://github.com/tensorflow/tensorflow/issues/9243
"""
end = time.time()
print("\n\nTime it took to run the code(in seconds): ", end - start)
