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
import glob
import mahotas as mh
#from chainer.dataset import dataset_mixin
from tqdm import tqdm
import chipchip as chip_
import cs_chip as chip_2

start = time.time()
#######################################
xview_numpy_array, numberOfChips = chip_2.get_numpy() #   <<<<< Our numpy array *********
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
#model.compile(optimizer='adam', )

            
#scaled_numpy = prepare_images_for_disc(xview_numpy_array)
#print("\nScaled numpy: \n", scaled_numpy.shape)


#Training Here:

batch_size = 250 #increments
#valid = np.ones((batch_size, 1))
#fake = np.zeros((batch_size, 1))
total_epochs = 3
loss = []
#range(start, stop, step)  step=incrementation
the_variable = 0
for epoch in range(total_epochs):
    for i in range(0, xview_numpy_array.shape[0], batch_size):
        #print(i) # Prints at each batch size until the i reaches the batch size. Then restarts from 0 to the batch size again, this occurs for however many epochs
        the_variable += 1
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 224, 224, 3]) #'batch_size' here was originally "numberOfChips"
        print('created noise', the_variable)          #Uncomment this
        generated_images = generator.predict(noise)         
        #print('generated noise')                           #Uncomment this
        real = xview_numpy_array[i:i+batch_size]#.reshape(-1, 224, 224, 3)              #Uncomment this
        #print("created real")
        shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        np.random.shuffle(shuffle_idx)                                                 #Uncomment this
        #print('shuffled')
        x = np.vstack([noise, real])[shuffle_idx]   #Keep this commented, error here
        y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        #print("X and Y: ", x)
        #print(y)
        #print("X: ", x.shape[0])
        #print("Y: ", y.shape[0])
       # print("created x and y")
        os.chdir(r"/root/Desktop/seniordesign/generated_images")
        random_input = np.random.uniform(-1.0, 1.0, size=[batch_size, 224, 224, 3])
        fake = generator.predict(random_input)
        mh.imsave('%d' % the_variable, fake[the_variable])
        os.chdir(r"/root/Desktop/seniordesign")
        #discriminator.trainable = True
        #d_loss = discriminator.train_on_batch(x, y)
        #print("Created d loss")

        #g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))
        #print("g loss created")


print("Number of times this forloop runs : ", the_variable)


#Tensorboard and other plots:

tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create graph
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
c = tf.add(a, b, name="addition")
# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# launch the graph in a session
with tf.Session() as Session1:
    # or creating the writer inside the session
    writer = tf.summary.FileWriter('./xView_Graphs', Session1.graph)
    print(Session1.run(c)) # this will print 5 

#After graphs are saved in 'xView_Graphs directory/folder, run this command in terminal: tensorboard --logdir==”./xView_Graphs” --port 6006'
#and then use the generated link


"""
Links that I used:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.ndim.html
https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
https://stackoverflow.com/questions/43743593/keras-how-to-get-layer-shapes-in-a-sequential-model
https://www.tensorflow.org/guide/tensor
https://www.tensorflow.org/api_docs/python/tf/rank
https://stackoverflow.com/questions/57893542/convolution2d-depth-of-input-is-not-a-multiple-of-input-depth-of-filter
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py#L138
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
https://stackoverflow.com/questions/49079115/valueerror-negative-dimension-size-caused-by-subtracting-2-from-1-for-max-pool


https://itnext.io/how-to-use-tensorboard-5d82f8654496    <TensorBoard

 "ValueError: Input tensor must be of rank 3, 4 or 5 but was 2."
https://www.tensorflow.org/api_docs/python/tf/rank   <<<<<<<<<<<<<<<<<<<<<<<

https://www.tensorflow.org/guide/tensor

https://github.com/tensorflow/tensorflow/issues/9243



use matplot to save fake images


"""
end = time.time()
print("\n\nTime it took to run the code(in seconds): ", end - start)
