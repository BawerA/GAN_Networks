#!/usr/bin/env python
# coding: utf-8


# Testing protoype or final code
#!/usr/bin/env python
# coding: utf-8

from keras.models import load_model
import os
import tensorflow.keras
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Dropout
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (Conv1D ,Conv2D, Input, Reshape, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout)
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
import matplotlib
import glob
import mahotas as mh
#from chainer.dataset import dataset_mixin
from tqdm import tqdm
import team_gmu_train
import team_gmu_chip


start = time.time()
#######################################
xview_numpy_array, numberOfChips = team_gmu_chip.get_numpy() #   <<<<< Our numpy array *********
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

'''
def build_generator():
    model = models.Sequential()
    #First Layer:  (Convolutional Layer)
    #model.add(LeakyReLU(0.2))
    #model.add(BatchNormalization())
    #model.add(Reshape(56,56,112))
    model.add(Conv2D(filters= 32, kernel_size= (5,5), data_format='channels_last', input_shape=(1,516), padding='same'))
    model.add(BatchNormalization())#Makes the training a little easier, standardizes the activations from a prior layer to have a zero mean and unit variance. 
    """
    For Conv2D()'s parameters, use this link >>>>>>>>>>> https://missinglink.ai/guides/keras/keras-conv2d-working-cnn-2d-convolutions-keras/#section4
    Things online say to start the filter with 32 and then double it each time Conv2D() is called (Example: next would be 64, then 128, 256)
    kernel size is the area of pixels that is scanned, this link explains it more >>> https://missinglink.ai/guides/keras/keras-conv2d-working-cnn-2d-convolutions-keras/
    Fix the filters ^^^^^^^^^^^^^^^^ causing error i think< I this this might be fixed now
    """

    #  Below is the Pooling Layer
    model.add(MaxPooling2D(padding='same'))
    model.add(UpSampling2D())
    #  Another convolutional layer
    model.add(Conv2D(filters= 64, kernel_size= (5,5), padding='same'))
    model.add(MaxPooling2D(padding='same'))
    model.add(UpSampling2D())

    #  Below is the Fully-Connected or Core Layer
    #model.add(Flatten())
    model.add(Dropout(0.5))

    #    Uncomment  line 75 and 76
    
    model.add(Conv2D(3, kernel_size= (5,5), padding='same'))

    print("\n\t Generator's Layer information:\n")
    model.summary()
    return model
'''

def build_generator():
    
    # Depth is successively halved
    depth = 256
    dim = 8
    
    # Input layer (100-d noise vector) fully-connected to 256 * 8 * 8 = 16,384 node dense layer
    model = models.Sequential()
    model.add(Dense(depth*dim*dim, input_dim=100))
    model.add(LeakyReLU(0.02))
    model.add(BatchNormalization())
    
    # Reshape to a 8x8x256 tensor
    model.add(Reshape((dim, dim, depth)))
    
    # Upsample to 16x16x128 convolutional block
    model.add(UpSampling2D())    
    model.add(Conv2D(int(depth/2), (5, 5), padding='same'))
    model.add(LeakyReLU(0.02))
    model.add(BatchNormalization())
    
    # Upsample to 32x32x64 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), (5, 5), padding='same'))
    model.add(LeakyReLU(0.02))
    model.add(BatchNormalization())
    
    # Upsample to 64x64x32 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/8), (5, 5), padding='same'))
    model.add(LeakyReLU(0.02))
    model.add(BatchNormalization())
    
    # Upsample to 128x128x16 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/16), (5, 5), padding='same'))
    model.add(LeakyReLU(0.02))
    model.add(BatchNormalization())
    
    # Upsample to 256x256x3 convolutional block with sigmoid activation
    model.add(UpSampling2D())
    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    
    return model




def build_discriminator():
    model = models.Sequential()
    #model.add(Flatten(input_shape=(24, 1)))
    #model.add(Reshape((224,) ,input_shape=(224,224)))
    # Original Con2d line of code:       model.add(Conv2D(filters= 32, kernel_size= (5,5), data_format='channels_last',input_shape=(224, 224, 3),  padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(5,5), data_format='channels_last', input_shape=(250, 1, 3),  padding='same'))
    model.add(BatchNormalization()) 
    model.add(LeakyReLU(0.2))
    #Maybe change the filter size to the last put filter, dunno yet
    model.add(MaxPooling2D(padding='same'))
    model.add(UpSampling2D())
    #model.add(Dropout(0.5))
    #model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same'))
    #model.add(LeakyReLU(0.2))
    #model.add(Dropout(0.5))
    #model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same'))
    #model.add(LeakyReLU(0.2))
    #                                     Uncomment every layer above this line
    #model.add(Flatten())
    """
    model.add(Dropout(0.5))    
    model.add(UpSampling2D())
    model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same'))"""
    #model.add(UpSampling2D())
    # Look at this link:
    #  https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
   # tf.keras.backend.expand_dims(model, axis=-1)

    #model.add(Flatten()) **************<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< commented this and replaced it with dropout0.5 below
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='sigmoid'))  # 32 is the 'units' parameter, output shape will have 32 dimensions, dont know what to put for units. Just put 32,
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense   <<<<  Dense() documentation


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




#Build the stuff here:
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')
combined_model = build_combined(generator, discriminator)                          
combined_model.compile(optimizer='adam', loss='binary_crossentropy') 

#Training Here:

batch_size = 200 #increments
total_epochs = 10
loss = []
fake_images = []
the_variable = 0
for epoch in range(total_epochs):
    for i in range(0, xview_numpy_array.shape[0], batch_size):
        #print(i) # Prints at each batch size until the i reaches the batch size. Then restarts from 0 to the batch size again, this occurs for however many epochs
        #noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 224, 224, 3]) #'batch_size' here was originally "numberOfChips"
        #generator = build_generator()
        noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
        os.chdir(r"/root/Desktop/seniordesign/fake_images")
        fake_images.append(generator.predict(noise)[the_variable])
        plt.imsave('generated_%d.png' % the_variable, arr= fake_images[the_variable])
        os.chdir(r"/root/Desktop/seniordesign")                         #Uncomment this
        real = xview_numpy_array[i:i+batch_size].reshape(-1, 256, 256, 3)              #Uncomment this
        shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        np.random.shuffle(shuffle_idx)                                                 #Uncomment this
        print('shuffled %d ' % the_variable)
        
    
        x = np.vstack([noise, real])[shuffle_idx]   #Keep this commented, error here
        y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        
        the_variable += 1
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x, y)
        print("Created d loss")
        g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))


print("Number of times this forloop runs : ", the_variable)

end = time.time()
print("\n\nTime it took to run the code(in seconds): ", end - start)
