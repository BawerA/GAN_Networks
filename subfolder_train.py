#!/usr/bin/env python
# coding: utf-8


# Testing protoype or final code
#!/usr/bin/env python
# coding: utf-8

from keras.models import load_model
import os
import tensorflow.keras
#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Dropout
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (Conv1D ,Conv2D, Input, Reshape, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout)
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
import random
import subfolder_chip
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph() #Testing this thing out


#tf.disable_v2_behavior()

start = time.time()
#######################################
#numberOfChips = team_gmu_chip.get_numpy() #   <<<<< Our numpy array *********
#xview_numpy_array = np.load('numpy_data.npy')
#######################################
#print("\nThe rank of our numpy array:  ", xview_numpy_array.ndim)#I'm pretty sure this is the rank*



#print("\n Our numpy info: (Batch size(number of chips), width, height, channel) of the numpy array: \n\n", xview_numpy_array.shape)
#The first number in the parenthesis is the batch size. The batch size is just the number of chips in the folder
#The second and third values represent the dimensions of the chipped images
#The fourth value represents the channels of the images. 3 stands for RGB or color images. If it was 1, then that means the images are greyscale.


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
    #model.add(Reshape((256,256,3)))
    
    print("\n\n\n\t Generator's Layer information:\n")
    model.summary()    
    return model




def build_discriminator():
    model = models.Sequential()

    depth = 64 
    dropout = 0.4
    dim = 7


    model.add(Conv2D(depth*1, 5, strides=5, input_shape=(256, 256, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(depth*2, 5, strides=5, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(depth*4, 5, strides=5, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(depth*8, 5, strides=5, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(depth*16, 5, strides=5, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(depth*32, 5, strides=5, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # 32 is the 'units' parameter, output shape will have 32 dimensions, dont know what to put for units. Just put 32,
    print("\n\tDisrciminator's Layer information:\n")
    model.summary()
    return model




def build_combined(generator, discriminator):
    model = models.Sequential()
    model.add(generator)        
    discriminator.trainable = True
    model.add(discriminator)
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

batch_size = 100 #increments   https://github.com/tensorflow/tensorflow/issues/18736   CALLED Mini batchsize gradient decent 
total_epochs = 1
loss = []
#temp_array = []
#xview_numpy_array = []
fake_images = []
the_variable = 0
for epoch in range(total_epochs):
    for i in range(0, 100, batch_size):
        #print(i) # Prints at each batch size until the i reaches the batch size. Then restarts from 0 to the batch size again, this occurs for however many epochs
        xview_numpy_array = subfolder_chip.numpyify(batch_size)
        noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))# Random noise vector
        fake = generator.predict(noise)
        
        os.chdir(r"/Users/Bawer/Desktop/tensorboard/fake_images")

        fake_images.append(generator.predict(noise)[the_variable])# Sending noise vector to genereator

        plt.imsave('generated_%d.png' % the_variable, arr= fake_images[the_variable])
        os.chdir(r"/Users/Bawer/Desktop/tensorboard")                         #Uncomment this
        real1 = np.array(xview_numpy_array)
        real = real1[i:i+batch_size].reshape(-1, 256, 256, 3)              #Uncomment this

        shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        np.random.shuffle(shuffle_idx)                                                 #Uncomment this
        print('Image Generated: %d ' % the_variable)
        
        
        x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
        y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        

        the_variable += 1
        discriminator.trainable = False#True
        d_loss = discriminator.train_on_batch(x, y)
        print("Added Disc. Loss")
        g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))
        print("Added Gen. Loss")
        loss.append([d_loss, g_loss])


print("Number of times this forloop runs : ", the_variable)

##This is where Saad was trying to get Tensorboard to work with our coder

loss = np.array(loss)
plt.figure()
plt.plot(loss[:, 0], label='Discriminator Loss')
plt.plot(loss[:, 1], label='Generator Loss')
plt.legend()
os.chdir(r"/Users/Bawer/Desktop/tensorboard/fake_images")
plt.savefig("Loss_Graph.png")

d_loss = np.array(d_loss)


# create the scalar variable
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
#y_scalar = tf.get_variable('y_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))

# ____step 1:____ create the scalar summary
first_summary = tf.summary.scalar(name='My_first_scalar_summary', tensor=x_scalar)
#second_summary = tf.summary.scalar(name='My_second_scalar_summary', tensor=y_scalar)
init = tf.global_variables_initializer()

print("\nTesting tensorboard\n")
with tf.Session() as sess:
    writer = tf.summary.FileWriter('/Users/Bawer/Desktop/tensorboard/fake_images/logs/plot_1', sess.graph) 


    for step in range(100):
        # loop over several initializations of the variable
        sess.run(init)
        # ____step 3:____ evaluate the scalar summary
        summary = sess.run(first_summary)
        # ____step 4:____ add the summary to the writer (i.e. to the event file)
        writer.add_summary(summary, step)
        writer.flush()
    print('Done with writing the scalar summary')


print('yo')


end = time.time()
print("\n\nTime it took to run the code(in seconds): ", end - start)


#Y is Loss
#X is Epochs

