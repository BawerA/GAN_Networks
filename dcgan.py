#!/usr/bin/env python
import matplotlib.pyplot as plt
import keras
from keras.layers import (Conv2D, Input, Reshape, Dropout, LeakyReLU, UpSampling2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout)
from keras import models
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
import gc
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_set = (mnist.train.images - 0.5) / 0.5 # Returns np.array

def Generate_generator():
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=100))
	#model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=100))
	model.add(tf.keras.layers.ReLU(0.2))   #DCGAN has 0.02 instead of 0.2 which the meduium article states to usue
	model.add(tf.keras.layers.BatchNormalization())
	
	model.add(tf.keras.layers.UpSampling2D())
	model.add(tf.keras.layers.Conv2D(64, padding='same', input_shape=(28, 28, 1)))
	model.add(tf.keras.layers.ReLU(0.2))
	model.add(tf.keras.layers.BatchNormalization())

	model.add(tf.keras.layers.UpSampling2D())
	model.add(tf.keras.layers.Conv2D(1, activation='tanh'))
	return model


def Generate_discriminator():
	model = tf.keras.models.Sequential()
	#model.add()
	return model


def Combined(generator, discriminator):
	model = tf.keras.models.Sequential()
	return model

print("\n\ncreate discriminator...\n\n")
discriminator = Generate_discriminator()


print("\nGenerating Generator, sir\n\n")
generator = Generate_generator()
print("\n\nGeneration of Generator complete....\n\n")


print('\n\nGenerator and discriminator doing the fusion dance\n\n')
gendisc = Combined(generator, discriminator)
print("\nFusion COMPLETE\n")


print('\n\ndone\n\n')