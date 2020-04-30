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
    model.add(AveragePooling2D(pool_size=(1,1), padding='valid'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())


    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    
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
    #model.add(BatchNormalization())

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
    model.add(LeakyReLU(0.6))
    #model.add(BatchNormalization())
    
    # Reshape to a 8x8x256 tensor
    model.add(Reshape((dim, dim, depth)))
    
    # Upsample to 16x16x128 convolutional block
    model.add(UpSampling2D())    
    model.add(Conv2D(int(depth/2), (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.5))
    #model.add(BatchNormalization())

    # Upsample to 32x32x64 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.5))
    model.add(BatchNormalization())

    # Upsample to 64x64x32 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/8), (5, 5), padding='same'))
    #model.add(MaxPooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    #model.add(BatchNormalization())
    

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
    model.add(MaxPooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    # Upsample to 32x32x64 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())

    # Upsample to 64x64x32 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/8), (5, 5), padding='same'))
    #model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    #model.add(BatchNormalization())
    
    # Upsample to 128x128x16 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/16), (5, 5), padding='same'))
    #model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    #model.add(BatchNormalization())

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
    model.add(MaxPooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    model.add(BatchNormalization())
    
    # Upsample to 32x32x64 convolutional block
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/4), (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(1,1), padding='same'))
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
    
    
    
    
    ####################################################################
    model.add(UpSampling2D())
    model.add(Conv2D(int(depth/32), (5, 5), padding='same'))
    #model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    model.add(LeakyReLU(0.4))
    #model.add(BatchNormalization())
######################################################################

    
    
    model.add(Conv2D(3, (5, 5), padding='same', activation='sigmoid'))
    
    print("\n\n\n\t 6th Generator's Layer information:\n")
    model.summary()    
    return model



def build_discriminator_1():
    model = models.Sequential()
    depth = 64
    model.add(Conv2D(depth*1, 5, strides=1, input_shape=(8, 8, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid')) 
    print("\n\t 1st Disrciminator's Layer information:\n")
    model.summary()
    return model


def build_discriminator_2():
    model = models.Sequential()
    depth = 64
    model.add(Conv2D(depth*1, 1, strides=1, input_shape=(16, 16, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(AveragePooling2D(pool_size=(1,1), padding='same'))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*2, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid')) 
    print("\n\t 2nd Disrciminator's Layer information:\n")
    model.summary()
    return model



def build_discriminator_3():
    model = models.Sequential()
    depth = 64
    model.add(Conv2D(depth*1, 5, strides=1, input_shape=(32, 32, 3), padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.6))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*2, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*4, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  
    print("\n\t 3rd Disrciminator's Layer information:\n")
    model.summary()
    return model


def build_discriminator_4():
    model = models.Sequential()
    depth = 64
    model.add(Conv2D(depth*1, 1, strides=1, input_shape=(64, 64, 3), padding='same', activation=LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.7))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*2, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Conv2D(depth*4, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*8, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.3)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid')) 
    print("\n\t 4th Disrciminator's Layer information:\n")
    model.summary()
    return model

  
def build_discriminator_5():
    model = models.Sequential()
    depth = 64
    
    model.add(Conv2D(depth*1, 1, strides=1, input_shape=(128, 128, 3), padding='same', activation=LeakyReLU(alpha=0.4)))
    model.add(Dropout(0.6))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*2, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.5)))
    model.add(Dropout(0.3))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*4, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.5)))
    model.add(Dropout(0.3))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*8, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.5)))
    model.add(Dropout(0.3))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*16, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.5)))
    model.add(Dropout(0.3))
    
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid')) 
    print("\n\t 5th Disrciminator's Layer information:\n")
    model.summary()
    return model


def build_discriminator_6():
    model = models.Sequential()
    depth = 64
    model.add(Conv2D(depth*1, 1, strides=1, input_shape=(256, 256, 3), padding='same', activation=LeakyReLU(alpha=0.4)))
    model.add(Dropout(0.6))
    model.add(BatchNormalization())
    model.add(Conv2D(depth*2, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Conv2D(depth*4, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*8, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Conv2D(depth*16, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    
    model.add(Conv2D(depth*32, 1, strides=2, padding='same', activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  
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

batch_size = 100 #increments   https://github.com/tensorflow/tensorflow/issues/18736   CALLED Mini batchsize gradient decent 
total_epochs = 60
loss = []
p = 0


#Build the stuff here:


fake_images = []
fake_images2 = []
fake_images3 = []
fake_images4 = []
fake_images5 = []
fake_images6 = []
the_variable = 0
the_variable2 = 0
the_variable3 = 0
the_variable4 = 0
the_variable5 = 0
the_variable6 = 0
g_loss_array = []
d_loss_array = []
disc_array = []
epoch_counter = -1
current_epoch = 1
for epoch in range(total_epochs):
    print("\n\nEpoch : ", current_epoch)
    current_epoch += 1
    print("\n\n")
    epoch_counter += 1
    graph_value = epoch_counter%5
    if epoch_counter == 0:
        print("First 5 epochs: ")
        w = 1
        generator1 = build_generator_1()    
        discriminator1 = build_discriminator_1()
        discriminator1.compile(optimizer='adam', loss='binary_crossentropy')
        combined_model1 = build_combined_1(generator1, discriminator1)#             0-250       1
        combined_model1.compile(optimizer='adam', loss='binary_crossentropy')
        generator1.save_weights('gen1_weights.h5')
        discriminator1.save_weights('disc1_weights.h5')
    elif epoch_counter == 5:
        print("Next 5 epochs(5 and 10): \n")
        w = 2
        generator2 = build_generator_2()
        os.chdir(r"/home/ubuntu")
        generator2.load_weights('gen1_weights.h5', by_name=True)
        discriminator2 = build_discriminator_2()
        discriminator2.load_weights('disc1_weights.h5', by_name=True)    
        discriminator2.compile(optimizer='adam', loss='binary_crossentropy')
        combined_model2 = build_combined_2(generator2, discriminator2) #              250-500        2                         
        combined_model2.compile(optimizer='adam', loss='binary_crossentropy')
        generator2.save_weights('gen2_weights.h5')
        discriminator2.save_weights('disc2_weights.h5')    
    elif epoch_counter == 10:
        print("Next 5 epochs(10 and 15): \n")
        w = 3
        generator3 = build_generator_3()
        os.chdir(r"/home/ubuntu")
        generator3.load_weights('gen2_weights.h5', by_name=True)    
        discriminator3 = build_discriminator_3()
        discriminator3.load_weights('disc2_weights.h5', by_name=True)    
        discriminator3.compile(optimizer='adam', loss='binary_crossentropy')
        combined_model3 = build_combined_3(generator3, discriminator3) #           500-750         3                
        combined_model3.compile(optimizer='adam', loss='binary_crossentropy')
        generator3.save_weights('gen3_weights.h5')
        discriminator3.save_weights('disc3_weights.h5')
    elif epoch_counter == 15:
        print("Next 5 epochs(15 and 20): \n")
        w = 4
        generator4 = build_generator_4()
        os.chdir(r"/home/ubuntu")
        generator4.load_weights('gen3_weights.h5', by_name=True)    
        discriminator4 = build_discriminator_4()
        discriminator4.load_weights('disc3_weights.h5', by_name=True)
        discriminator4.compile(optimizer='adam', loss='binary_crossentropy')
        combined_model4 = build_combined_4(generator4, discriminator4)#                750-1000    4                       
        combined_model4.compile(optimizer='adam', loss='binary_crossentropy')
        generator4.save_weights('gen4_weights.h5')
        discriminator4.save_weights('disc4_weights.h5')
    elif epoch_counter == 20:
        print("Next 5 epochs(20 and 25): \n")
        w = 5
        generator5 = build_generator_5()
        os.chdir(r"/home/ubuntu")
        generator5.load_weights('gen4_weights.h5', by_name=True)    
        discriminator5 = build_discriminator_5()
        discriminator5.load_weights('disc4_weights.h5', by_name=True)
        discriminator5.compile(optimizer='adam', loss='binary_crossentropy')
        combined_model5 = build_combined_5(generator5, discriminator5)#               1000-1500   5                         
        combined_model5.compile(optimizer='adam', loss='binary_crossentropy')
        generator5.save_weights('gen5_weights.h5')
        discriminator5.save_weights('disc5_weights.h5')    
    elif epoch_counter == 30:
        print("Next 5 epochs(25 and 30): \n")
        w = 6
        generator6 = build_generator_6()
        os.chdir(r"/home/ubuntu")
        generator6.load_weights('gen5_weights.h5', by_name=True)    
        discriminator6 = build_discriminator_6()
        discriminator6.load_weights('disc5_weights.h5', by_name=True)
        discriminator6.compile(optimizer='adam', loss='binary_crossentropy')
        combined_model6 = build_combined_6(generator6, discriminator6)                          
        combined_model6.compile(optimizer='adam', loss='binary_crossentropy')
        #generator6.save_weights('gen6_weights.h5')
        #discriminator6.save_weights('disc6_weights.h5')    
    else:
        print("\n\n\n\nEpoch counter value :  ", epoch_counter)
        print("W value : ", w)
        print("\n")
        #w == 129
    for i in range(0, 5000, batch_size):
        #print(i) # Prints at each batch size until the i reaches the batch size. Then restarts from 0 to the batch size again, this occurs for however many epochs
        if w == 1:
            xview_numpy_array_1 = down_chip_AWS.get_8x8(batch_size)
            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            fake = generator1.predict(noise)
            fake_images.append(generator1.predict(noise)[p])
            os.chdir(r"/home/ubuntu/fake_images/8x8_fake_images") 
            plt.imsave('generated_8x8_%d.png' % the_variable, arr= fake_images[the_variable])
            real1 = np.array(xview_numpy_array_1)
            real = real1[i:i+batch_size].reshape(-1, 8, 8, 3)              #Uncomment this
            #shuffle_idx = np.arrange(b, the_variable, batch_size)
            shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
            np.random.shuffle(shuffle_idx)
            print("8x8 Image Generated: ", the_variable)
            x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
            y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
            discriminator1.trainable = False
            d_loss = discriminator1.train_on_batch(x, y)
            print("Adding the Discriminator Loss for 8x8 ")
            disc_array.append(d_loss)
            print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
            g_loss = combined_model1.train_on_batch(noise, np.zeros(batch_size))
            print("Adding the Generator loss...")
            g_loss_array.append(g_loss)
            print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
            loss.append([d_loss, g_loss])
            the_variable += 1
            print("\n\nEpoch : ", current_epoch)
            print("\n\n")
        elif w == 2:
            xview_numpy_array_2 = down_chip_AWS.get_16x16(batch_size)
            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            fake = generator2.predict(noise)
            fake_images2.append(generator2.predict(noise)[p])
            os.chdir(r"/home/ubuntu/fake_images/16x16_fake_images") 
            plt.imsave('generated_16x16_%d.png' % the_variable, arr= fake_images2[the_variable2])
            real1 = np.array(xview_numpy_array_2)
            real = real1[i:i+batch_size].reshape(-1, 16, 16, 3)              #Uncomment this
            #shuffle_idx = np.arrange(b, the_variable, batch_size)
            shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
            np.random.shuffle(shuffle_idx)
            print("16x16 Image Generated: ", the_variable)
            x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
            y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
            discriminator2.trainable = False
            d_loss = discriminator2.train_on_batch(x, y)
            print("Adding the Discriminator Loss for 16x16 ")
            disc_array.append(d_loss)
            print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
            g_loss = combined_model2.train_on_batch(noise, np.zeros(batch_size))
            print("Adding the Generator loss...")
            g_loss_array.append(g_loss)
            print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
            loss.append([d_loss, g_loss])
            the_variable2 += 1
            the_variable += 1
            print("Epoch : ", current_epoch)
            print("\n\n")
        elif w == 3:
            xview_numpy_array_3 = down_chip_AWS.get_32x32(batch_size)
            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            fake = generator3.predict(noise)
            fake_images3.append(generator3.predict(noise)[p])
            os.chdir(r"/home/ubuntu/fake_images/32x32_fake_images") 
            plt.imsave('generated_32x32_%d.png' % the_variable, arr= fake_images3[the_variable3])
            real1 = np.array(xview_numpy_array_3)
            real = real1[i:i+batch_size].reshape(-1, 32, 32, 3)              #Uncomment this
            #shuffle_idx = np.arrange(b, the_variable, batch_size)
            shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
            np.random.shuffle(shuffle_idx)
            print("32x32 Image Generated: ", the_variable)
            x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
            y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
            discriminator3.trainable = False
            d_loss = discriminator3.train_on_batch(x, y)
            print("Adding the Discriminator Loss for 32x32 ")
            disc_array.append(d_loss)
            print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
            g_loss = combined_model3.train_on_batch(noise, np.zeros(batch_size))
            print("Adding the Generator loss...")
            g_loss_array.append(g_loss)
            print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
            loss.append([d_loss, g_loss])
            the_variable3 +=1
            the_variable += 1
            print("Epoch : ", current_epoch)
            print("\n\n")
        elif w == 4:
            xview_numpy_array_4 = down_chip_AWS.get_64x64(batch_size)
            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            fake = generator4.predict(noise)
            fake_images4.append(generator4.predict(noise)[p])
            os.chdir(r"/home/ubuntu/fake_images/64x64_fake_images") 
            plt.imsave('generated_64x64_%d.png' % the_variable, arr= fake_images4[the_variable4])
            real1 = np.array(xview_numpy_array_4)
            real = real1[i:i+batch_size].reshape(-1, 64, 64, 3)              #Uncomment this
            #shuffle_idx = np.arrange(b, the_variable, batch_size)
            shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
            np.random.shuffle(shuffle_idx)
            print("64x64 Image Generated: ", the_variable)
            x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
            y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
            discriminator4.trainable = False
            d_loss = discriminator4.train_on_batch(x, y)
            print("Adding the Discriminator Loss for 64x64 ")
            disc_array.append(d_loss)
            print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
            g_loss = combined_model4.train_on_batch(noise, np.zeros(batch_size))
            print("Adding the Generator loss...")
            g_loss_array.append(g_loss)
            print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
            loss.append([d_loss, g_loss])
            the_variable4 += 1
            the_variable += 1
            print("Epoch : ", current_epoch)
            print("\n\n")
        elif w == 5:
            xview_numpy_array_5 = down_chip_AWS.get_128x128(batch_size)
            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            fake = generator5.predict(noise)
            fake_images5.append(generator5.predict(noise)[p])
            os.chdir(r"/home/ubuntu/fake_images/128x128_fake_images") 
            plt.imsave('generated_128x128_%d.png' % the_variable, arr= fake_images5[the_variable5])
            real1 = np.array(xview_numpy_array_5)
            real = real1[i:i+batch_size].reshape(-1, 128, 128, 3)              #Uncomment this
            #shuffle_idx = np.arrange(b, the_variable, batch_size)
            shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
            np.random.shuffle(shuffle_idx)
            print("128x128 Image Generated: ", the_variable)
            x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
            y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
            discriminator5.trainable = False
            d_loss = discriminator5.train_on_batch(x, y)
            print("Adding the Discriminator Loss for 128x128 ")
            disc_array.append(d_loss)
            print("Discriminator loss for this image and the loss for it: ", the_variable, disc_array[the_variable])
            g_loss = combined_model5.train_on_batch(noise, np.zeros(batch_size))
            print("Adding the Generator loss...")
            g_loss_array.append(g_loss)
            print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
            loss.append([d_loss, g_loss])
            the_variable5 += 1
            the_variable += 1
            print("Epoch : ", current_epoch)
            print("\n\n")
        elif w == 6:
            xview_numpy_array = down_chip_AWS.numpyify(batch_size)
            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))# Random noise vector
            fake = generator6.predict(noise)
            os.chdir(r"/home/ubuntu/fake_images/256x256_fake_images")
            fake_images6.append(generator6.predict(noise)[p])
            plt.imsave('generated_%d.png' % the_variable, arr= fake_images6[the_variable6])
            os.chdir(r"/home/ubuntu/")                         #Uncomment this
            real1 = np.array(xview_numpy_array)
            real = real1[i:i+batch_size].reshape(-1, 256, 256, 3)              #Uncomment this
            shuffle_idx = np.arange(batch_size)#.reshape(1,224,224,3)                             #Uncomment this
            np.random.shuffle(shuffle_idx)                                                 #Uncomment this
            print('256x256 Image Generated: %d ' % the_variable)
            x = np.vstack([fake, real])[shuffle_idx]   #Keep this commented, error here
            y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]  #Keep this commented
            discriminator6.trainable = False 
            d_loss = discriminator6.train_on_batch(x, y)
            print("Adding Discriminar Loss")
            disc_array.append(d_loss)
            print("Discriminator loss for this image: %d, Loss: %d ", the_variable, disc_array[the_variable])
            #discriminator.trainable = False#True
            g_loss = combined_model6.train_on_batch(noise, np.zeros(batch_size))
            print("Adding Generator Loss")
            g_loss_array.append(g_loss)
            #print("Generator Loss for this image: %d, Loss: %d ", the_variable, g_loss)
            print("Generator loss for this image and the loss for it: ", the_variable, g_loss_array[the_variable])
            print("\n\n")
            loss.append([d_loss, g_loss])
            the_variable6 += 1
            the_variable += 1
            print("Epoch : ", current_epoch)
            #epoch_counter += 1
        else:
            print("Something went wrong homie")
            end()            
        loss_1 = np.array(loss)
        plt.figure()
        plt.plot(loss_1[:, 0], label='Discriminator Loss')
        plt.plot(loss_1[:, 1], label='Generator Loss')
        plt.legend()
        os.chdir(r"/home/ubuntu/fake_images")
        plt.savefig("Loss_Graph_DownsamplingGAN.png")
        plt.close('all')



loss = np.array(loss)
plt.figure()
plt.plot(loss[:, 0], label='Discriminator Loss')
plt.plot(loss[:, 1], label='Generator Loss')
plt.legend()
os.chdir(r"/home/ubuntu/fake_images")
plt.savefig("Loss_Graph_DownsamplingGAN.png")



end = time.time()
print("\n\nTime it took to run the code(in seconds): ", end - start)