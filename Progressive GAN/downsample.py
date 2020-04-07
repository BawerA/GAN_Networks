#!/usr/bin/env python
# coding: utf-8


# Testing protoype or final code
#!/usr/bin/env python
# coding: utf-8

from keras.models import load_model
import os
import tensorflow.keras
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, Dropout
from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import (Conv1D ,Conv2D, Conv2DTranspose, Input, Reshape, AveragePooling2D, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout)
#import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import Image, ImageDraw
import skimage.filters as filters
import io
import glob
from tqdm import tqdm
import logging
import argparse
from numpy import random
import os
import json
import wv_util as wv
from numpy import random
import time
#import tfr_util as tfr
#import aug_util as aug
import csv
#import random
import six
import cv2
import matplotlib
import glob
import mahotas as mh
#from chainer.dataset import dataset_mixin
from tqdm import tqdm
#import random
import subfolder_chip
import down_chip
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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




def build_generator_1():
    
    # Depth is successively halved
    depth = 256
    dim = 8
    
    # Input layer (100-d noise vector) fully-connected to 256 * 8 * 8 = 16,384 node dense layer
    model = models.Sequential()
    model.add(Dense(8*16*128, input_dim=100))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    model.add(Reshape((dim, dim, depth)))
    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    #model.add(Reshape((256,256,3)))
    
    print("\n\n\n\t 1st Generator's Layer information:\n")
    model.summary()    
    return model

def build_generator_2():
    
    # Depth is successively halved
    depth = 256
    dim = 8
    
    # Input layer (100-d noise vector) fully-connected to 256 * 8 * 8 = 16,384 node dense layer
    model = models.Sequential()
    model.add(Dense(depth*dim*dim, input_dim=100))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    # Reshape to a 8x8x256 tensor
    model.add(Reshape((dim, dim, depth)))
    
    # Upsample to 16x16x128 convolutional block
    model.add(UpSampling2D())    
    model.add(Conv2D(int(depth/2), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())



    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    #model.add(Reshape((256,256,3)))
    
    print("\n\n\n\t 2nd Generator's Layer information:\n")
    model.summary()    
    return model

def build_generator_3():
    
    # Depth is successively halved
    depth = 256
    dim = 8
    
    # Input layer (100-d noise vector) fully-connected to 256 * 8 * 8 = 16,384 node dense layer
    model = models.Sequential()
    model.add(Dense(depth*dim*dim, input_dim=100))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    # Reshape to a 8x8x256 tensor
    model.add(Reshape((dim, dim, depth)))
    
    # Upsample to 16x16x128 convolutional block
    model.add(UpSampling2D())    
    model.add(Conv2D(int(depth/2), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    # Upsample to 32x32x64 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    #model.add(Reshape((256,256,3)))
    
    print("\n\n\n\t 3rd Generator's Layer information:\n")
    model.summary()    
    return model


def build_generator_4():
    
    # Depth is successively halved
    depth = 256
    dim = 8
    
    # Input layer (100-d noise vector) fully-connected to 256 * 8 * 8 = 16,384 node dense layer
    model = models.Sequential()
    model.add(Dense(depth*dim*dim, input_dim=100))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    # Reshape to a 8x8x256 tensor
    model.add(Reshape((dim, dim, depth)))
    
    # Upsample to 16x16x128 convolutional block
    model.add(UpSampling2D())    
    model.add(Conv2D(int(depth/2), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    # Upsample to 32x32x64 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    # Upsample to 64x64x32 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/8), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    

    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    #model.add(Reshape((256,256,3)))
    
    print("\n\n\n\t 4th Generator's Layer information:\n")
    model.summary()    
    return model


def build_generator_5():
    
    # Depth is successively halved
    depth = 256
    dim = 8
    
    # Input layer (100-d noise vector) fully-connected to 256 * 8 * 8 = 16,384 node dense layer
    model = models.Sequential()
    model.add(Dense(depth*dim*dim, input_dim=100))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    # Reshape to a 8x8x256 tensor
    model.add(Reshape((dim, dim, depth)))
    
    # Upsample to 16x16x128 convolutional block
    model.add(UpSampling2D())    
    model.add(Conv2D(int(depth/2), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    # Upsample to 32x32x64 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    # Upsample to 64x64x32 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/8), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    # Upsample to 128x128x16 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/16), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    #model.add(Reshape((256,256,3)))
    
    print("\n\n\n\t 5th Generator's Layer information:\n")
    model.summary()    
    return model


def build_generator_6():
    
    # Depth is successively halved
    depth = 256
    dim = 8
    
    # Input layer (100-d noise vector) fully-connected to 256 * 8 * 8 = 16,384 node dense layer
    model = models.Sequential()
    model.add(Dense(depth*dim*dim, input_dim=100))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    

    # Reshape to a 8x8x256 tensor
    model.add(Reshape((dim, dim, depth)))
    
    # Upsample to 16x16x128 convolutional block
    model.add(UpSampling2D())    
    model.add(Conv2D(int(depth/2), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    
    # Upsample to 32x32x64 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    
    # Upsample to 64x64x32 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/8), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    
    
    # Upsample to 128x128x16 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/16), (5, 5), padding='same'))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    
    
    # Upsample to 256x256x3 convolutional block with sigmoid activation
    model.add(UpSampling2D())
    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    #model.add(Reshape((256,256,3)))
    
    print("\n\n\n\t 7th Generator's Layer information:\n")
    model.summary()    
    return model



def build_discriminator_1():
    model = models.Sequential()

    depth = 64 
    dropout = 0.4
    dim = 7


    model.add(Conv2D(int(depth*1), 5, strides=1, input_shape=(8, 8, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))


    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # 32 is the 'units' parameter, output shape will have 32 dimensions, dont know what to put for units. Just put 32,
    print("\n\t 1st Disrciminator's Layer information:\n")
    model.summary()
    return model


def build_discriminator_2():
    model = models.Sequential()

    depth = 64 
    dropout = 0.4
    dim = 7


    model.add(Conv2D(int(depth*1), 5, strides=1, input_shape=(16, 16, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*2), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # 32 is the 'units' parameter, output shape will have 32 dimensions, dont know what to put for units. Just put 32,
    print("\n\t 2nd Disrciminator's Layer information:\n")
    model.summary()
    return model



def build_discriminator_3():
    model = models.Sequential()

    depth = 64 
    dropout = 0.4
    dim = 7


    model.add(Conv2D(int(depth*1), 5, strides=1, input_shape=(32, 32, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*2), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*4), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # 32 is the 'units' parameter, output shape will have 32 dimensions, dont know what to put for units. Just put 32,
    print("\n\t 3rd Disrciminator's Layer information:\n")
    model.summary()
    return model


def build_discriminator_4():
    model = models.Sequential()

    depth = 64 
    dropout = 0.4
    dim = 7


    model.add(Conv2D(int(depth*1), 5, strides=1, input_shape=(64, 64, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(int(depth*2), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*4), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*8), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # 32 is the 'units' parameter, output shape will have 32 dimensions, dont know what to put for units. Just put 32,
    print("\n\t 4th Disrciminator's Layer information:\n")
    model.summary()
    return model


def build_discriminator_5():
    model = models.Sequential()

    depth = 64 
    dropout = 0.4
    dim = 7


    model.add(Conv2D(int(depth*1), 5, strides=1, input_shape=(128, 128, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(int(depth*2), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*4), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*8), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(int(depth*16), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    
    #model.add(BatchNormalization())
    
    #model.add(Conv2D(depth*32, 5, strides=5, padding='same', activation=LeakyReLU(alpha=0.2)))
    #model.add(Dropout(0.4))
    
    #model.add(BatchNormalization())
    
    #model.add(Conv2D(depth*64, 5, strides=5, padding='same', activation=LeakyReLU(alpha=0.2)))
    #model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # 32 is the 'units' parameter, output shape will have 32 dimensions, dont know what to put for units. Just put 32,
    print("\n\t 5th Disrciminator's Layer information:\n")
    model.summary()
    return model


def build_discriminator_6():
    model = models.Sequential()

    depth = 64 
    dropout = 0.4
    dim = 7


    model.add(Conv2D(int(depth*1), 5, strides=1, input_shape=(256, 256, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(int(depth*2), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*4), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(Conv2D(int(depth*8), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(int(depth*16), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    
    #model.add(BatchNormalization())
    
    model.add(Conv2D(int(depth*32), 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    
    #model.add(BatchNormalization())
    
    #model.add(Conv2D(depth*64, 5, strides=5, padding='same', activation=LeakyReLU(alpha=0.2)))
    #model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # 32 is the 'units' parameter, output shape will have 32 dimensions, dont know what to put for units. Just put 32,
    print("\n\t 6th Disrciminator's Layer information:\n")
    model.summary()
    return model




def build_combined_1(generator_1, discriminator_1):
    model = models.Sequential()
    generator = generator_1
    discriminator = discriminator_1
    model.add(generator)        
    discriminator.trainable = True
    model.add(discriminator)
    print("\n\n\t 1st Combined Model Summary: \n\n")
    model.summary()
    return model


def build_combined_2(generator_2, discriminator_2):
    model = models.Sequential()
    model.add(generator_2)        
    discriminator_2.trainable = True
    model.add(discriminator_2)
    print("\n\n\t 2nd Combined Model Summary: \n\n")
    model.summary()
    return model


def build_combined_3(generator_3, discriminator_3):
    model = models.Sequential()
    model.add(generator_3)        
    discriminator_3.trainable = True
    model.add(discriminator_3)
    print("\n\n\t 3rd Combined Model Summary: \n\n")
    model.summary()
    return model


def build_combined_4(generator_4, discriminator_4):
    model = models.Sequential()
    model.add(generator_4)        
    discriminator_4.trainable = True
    model.add(discriminator_4)
    print("\n\n\t 4th Combined Model Summary: \n\n")
    model.summary()
    return model

def build_combined_5(generator_5, discriminator_5):
    model = models.Sequential()
    model.add(generator_5)        
    discriminator_5.trainable = True
    model.add(discriminator_5)
    print("\n\n\t 5th Combined Model Summary: \n\n")
    model.summary()
    return model


def build_combined_6(generator_6, discriminator_6):
    model = models.Sequential()
    model.add(generator_6)        
    discriminator_6.trainable = True
    model.add(discriminator_6)
    print("\n\n\t 6th Combined Model Summary: \n\n")
    model.summary()
    return model


def get_epoch_value(epoch_counter):
	temp_epoch = epoch_counter
	return temp_epoch



#Training Here:

batch_size = 150 #increments   https://github.com/tensorflow/tensorflow/issues/18736   CALLED Mini batchsize gradient decent 
total_epochs = 6
loss = []


#Build the stuff here:


fake_images = []
the_variable = 0
g_loss_array = []
d_loss_array = []
disc_array = []
epoch_counter = -1
current_epoch = 1
for epoch in range(total_epochs):
    print("\n\nEpoch : ", current_epoch)
    current_epoch += 1
    #print(epoch_counter)
    print("\n\n")
    epoch_counter += 1
    if epoch_counter == 0:
    	print("First 5 epochs: ")
    	w = 1
    	generator = build_generator_1()
    	discriminator = build_discriminator_1()
    	discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    	combined_model = build_combined_1(generator, discriminator)                          
    	combined_model.compile(optimizer='adam', loss='binary_crossentropy') 
    elif epoch_counter == 1:
    	print("Next 5 epochs(5 and 10): \n")
    	w = 2
    	generator = build_generator_2()
    	discriminator = build_discriminator_2()
    	discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    	combined_model = build_combined_2(generator, discriminator)                          
    	combined_model.compile(optimizer='adam', loss='binary_crossentropy')
    elif epoch_counter == 2:
    	print("Next 5 epochs(10 and 15): \n")
    	w = 3
    	generator = build_generator_3()
    	discriminator = build_discriminator_3()
    	discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    	combined_model = build_combined_3(generator, discriminator)                          
    	combined_model.compile(optimizer='adam', loss='binary_crossentropy')
    elif epoch_counter == 3:
    	print("Next 5 epochs(15 and 20): \n")
    	w = 4
    	generator = build_generator_4()
    	discriminator = build_discriminator_4()
    	discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    	combined_model = build_combined_4(generator, discriminator)                          
    	combined_model.compile(optimizer='adam', loss='binary_crossentropy')
    elif epoch_counter == 4:
    	print("Next 5 epochs(20 and 25): \n")
    	w = 5
    	generator = build_generator_5()
    	discriminator = build_discriminator_5()
    	discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    	combined_model = build_combined_5(generator, discriminator)                          
    	combined_model.compile(optimizer='adam', loss='binary_crossentropy')
    elif epoch_counter == 5:
    	print("Next 5 epochs(25 and 30): \n")
    	w = 6
    	generator = build_generator_6()
    	discriminator = build_discriminator_6()
    	discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    	combined_model = build_combined_6(generator, discriminator)                          
    	combined_model.compile(optimizer='adam', loss='binary_crossentropy')
    else:
        print("\n\n\n\nEpoch counter value :  ", epoch_counter)
        print("W value : ", w)
        print("\n")
    	#w == 129
    for i in range(0, 1000, batch_size):
        #print(i) # Prints at each batch size until the i reaches the batch size. Then restarts from 0 to the batch size again, this occurs for however many epochs
        if w == 1:
        	xview_numpy_array_1 = down_chip.get_8x8(batch_size)
        	noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
        	fake = generator.predict(noise)
        	fake_images.append(generator.predict(noise)[the_variable])
        	os.chdir(r"/media/root/New Volume/fake_images/8x8_fake_images") 
        	plt.imsave('generated_8x8_%d.png' % the_variable, arr= fake_images[the_variable])
        	real1 = np.array(xview_numpy_array_1)
        	real = real1[i:i+batch_size].reshape(-1, 8, 8, 3)              #Uncomment this
        	#shuffle_idx = np.arrange(b, the_variable, batch_size)
        	shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        	np.random.shuffle(shuffle_idx)
        	print("8x8 Image Generated: ", the_variable)
        	x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
        	y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        	discriminator.trainable = False
        	d_loss = discriminator.train_on_batch(x, y)
        	print("Adding the Discriminator Loss for 8x8 ")
        	disc_array.append(d_loss)
        	print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
        	g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))
        	print("Adding the Generator loss...")
        	g_loss_array.append(g_loss)
        	print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
        	loss.append([d_loss, g_loss])
        	the_variable += 1
        	print("\n\n")
        elif w == 2:
        	xview_numpy_array_2 = down_chip.get_16x16(batch_size)
        	noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
        	fake = generator.predict(noise)
        	fake_images.append(generator.predict(noise)[the_variable])
        	os.chdir(r"/media/root/New Volume/fake_images/16x16_fake_images") 
        	plt.imsave('generated_16x16_%d.png' % the_variable, arr= fake_images[the_variable])
        	real1 = np.array(xview_numpy_array_2)
        	real = real1[i:i+batch_size].reshape(-1, 16, 16, 3)              #Uncomment this
        	#shuffle_idx = np.arrange(b, the_variable, batch_size)
        	shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        	np.random.shuffle(shuffle_idx)
        	print("16x16 Image Generated: ", the_variable)
        	x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
        	y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        	discriminator.trainable = False
        	d_loss = discriminator.train_on_batch(x, y)
        	print("Adding the Discriminator Loss for 16x16 ")
        	disc_array.append(d_loss)
        	print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
        	g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))
        	print("Adding the Generator loss...")
        	g_loss_array.append(g_loss)
        	print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
        	loss.append([d_loss, g_loss])
        	the_variable += 1
        	print("\n\n")
        elif w == 3:
        	xview_numpy_array_3 = down_chip.get_32x32(batch_size)
        	noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
        	fake = generator.predict(noise)
        	fake_images.append(generator.predict(noise)[the_variable])
        	os.chdir(r"/media/root/New Volume/fake_images/32x32_fake_images") 
        	plt.imsave('generated_32x32_%d.png' % the_variable, arr= fake_images[the_variable])
        	real1 = np.array(xview_numpy_array_3)
        	real = real1[i:i+batch_size].reshape(-1, 32, 32, 3)              #Uncomment this
        	#shuffle_idx = np.arrange(b, the_variable, batch_size)
        	shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        	np.random.shuffle(shuffle_idx)
        	print("32x32 Image Generated: ", the_variable)
        	x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
        	y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        	discriminator.trainable = False
        	d_loss = discriminator.train_on_batch(x, y)
        	print("Adding the Discriminator Loss for 32x32 ")
        	disc_array.append(d_loss)
        	print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
        	g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))
        	print("Adding the Generator loss...")
        	g_loss_array.append(g_loss)
        	print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
        	loss.append([d_loss, g_loss])
        	the_variable += 1
        	print("\n\n")
        elif w == 4:
        	xview_numpy_array_4 = down_chip.get_64x64(batch_size)
        	noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
        	fake = generator.predict(noise)
        	fake_images.append(generator.predict(noise)[the_variable])
        	os.chdir(r"/media/root/New Volume/fake_images/64x64_fake_images") 
        	plt.imsave('generated_64x64_%d.png' % the_variable, arr= fake_images[the_variable])
        	real1 = np.array(xview_numpy_array_4)
        	real = real1[i:i+batch_size].reshape(-1, 64, 64, 3)              #Uncomment this
        	#shuffle_idx = np.arrange(b, the_variable, batch_size)
        	shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        	np.random.shuffle(shuffle_idx)
        	print("64x64 Image Generated: ", the_variable)
        	x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
        	y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        	discriminator.trainable = False
        	d_loss = discriminator.train_on_batch(x, y)
        	print("Adding the Discriminator Loss for 64x64 ")
        	disc_array.append(d_loss)
        	print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
        	g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))
        	print("Adding the Generator loss...")
        	g_loss_array.append(g_loss)
        	print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
        	loss.append([d_loss, g_loss])
        	the_variable += 1
        	print("\n\n")
        elif w == 5:
        	xview_numpy_array_5 = down_chip.get_128x128(batch_size)
        	noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
        	fake = generator.predict(noise)
        	fake_images.append(generator.predict(noise)[the_variable])
        	os.chdir(r"/media/root/New Volume/fake_images/128x128_fake_images") 
        	plt.imsave('generated_128x128_%d.png' % the_variable, arr= fake_images[the_variable])
        	real1 = np.array(xview_numpy_array_5)
        	real = real1[i:i+batch_size].reshape(-1, 128, 128, 3)              #Uncomment this
        	#shuffle_idx = np.arrange(b, the_variable, batch_size)
        	shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        	np.random.shuffle(shuffle_idx)
        	print("128x128 Image Generated: ", the_variable)
        	x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
        	y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        	discriminator.trainable = False
        	d_loss = discriminator.train_on_batch(x, y)
        	print("Adding the Discriminator Loss for 128x128 ")
        	disc_array.append(d_loss)
        	print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
        	g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))
        	print("Adding the Generator loss...")
        	g_loss_array.append(g_loss)
        	print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
        	loss.append([d_loss, g_loss])
        	the_variable += 1
        	print("\n\n")
        elif w == 6:
        	xview_numpy_array = subfolder_chip.numpyify(batch_size)
        	noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))# Random noise vector
        	fake = generator.predict(noise)
        	os.chdir(r"/media/root/New Volume/fake_images/256x256_fake_images")
        	fake_images.append(generator.predict(noise)[the_variable])# Sending noise vector to genereator
        	plt.imsave('generated_256x256_%d.png' % the_variable, arr= fake_images[the_variable])
        	os.chdir(r"/media/root/New Volume")                         #Uncomment this
        	real1 = np.array(xview_numpy_array)
        	real = real1[i:i+batch_size].reshape(-1, 256, 256, 3)              #Uncomment this
        	shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
        	np.random.shuffle(shuffle_idx)                                                 #Uncomment this
        	print('256x256 Image Generated: %d ' % the_variable)
        	x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
        	y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
        	discriminator.trainable = False#True 
        	d_loss = discriminator.train_on_batch(x, y)
        	print("Adding Discriminar Loss")
        	disc_array.append(d_loss)
        	print("Discriminator loss for this image: %d, Loss: %d ", the_variable, disc_array[the_variable])
        	#discriminator.trainable = False#True
        	g_loss = combined_model.train_on_batch(noise, np.zeros(batch_size))
        	print("Adding Generator Loss")
        	g_loss_array.append(g_loss)
        	print("Generator Loss for this image: %d, Loss: %d ", the_variable, g_loss)
        	print("\n\n")
        	loss.append([d_loss, g_loss])
        	the_variable += 1
        	#epoch_counter += 1
        else:
        	print("Something went wrong homie")
        	end()


loss = np.array(loss)
plt.figure()
plt.plot(loss[:, 0], label='Discriminator Loss')
plt.plot(loss[:, 1], label='Generator Loss')
plt.legend()
os.chdir(r"/media/root/New Volume/fake_images")
plt.savefig("Loss_Graph_DownsamplingGAN.png")



end = time.time()
print("\n\nTime it took to run the code(in seconds): ", end - start)

'''elif epoch_counter == 6:
        print("Next 5 epochs(30 and beyond): \n")
        w = 7
        generator = build_generator_7()
        discriminator = build_discriminator_6()
        discriminator.compile(optimizer='adam', loss='binary_crossentropy')
        combined_model = build_combined_6(generator, discriminator)                          
        combined_model.compile(optimizer='adam', loss='binary_crossentropy')'''