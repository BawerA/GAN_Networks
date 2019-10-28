#!/usr/bin/env python
# coding: utf-8

# https://medium.com/@lisulimowicz/dcgan-79af14a1c247

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[2]:

import matplotlib.pyplot as plt
import keras
from keras.layers import (Conv2D, Input, Reshape, Dropout, LeakyReLU, UpSampling2D,
                          MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout)
from keras import models


# In[3]:


def build_generator():
    # The generator will take in a vector of length 100
    # and create a 28x28x1 image
    model = models.Sequential()
    model.add(Dense(128*7*7, input_dim=100))
    model.add(LeakyReLU(0.02))
    model.add(BatchNormalization())
    
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(LeakyReLU(0.02))
    model.add(BatchNormalization())
    
    model.add(UpSampling2D())
    model.add(Conv2D(1, (5, 5), padding='same', activation='tanh'))
    return model

def build_discriminator():
    # The discriminator will take in a 28x28x1 image and
    # try to determine if it was generated or a real image
    model = models.Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_combined(generator, discriminator):
    # The combined model will not train the discriminator.
    # It will be used to train the generator based off of how
    # well the discriminator can tell the generator and real images apart
    model = models.Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


# In[12]:


from keras.optimizers import SGD
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

generator = build_generator()
combined_model = build_combined(generator, discriminator)

combined_model.compile(optimizer='adam', loss='binary_crossentropy')


# In[ ]:


import tensorflow as tf
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_set = (mnist.train.images - 0.5) / 0.5 # Returns np.array


# In[13]:


import numpy as np

total_epochs = 10
batch_size = 500
loss = []
for epoch in range(total_epochs):
    
    for i in range(0, train_set.shape[0], batch_size):
        # Train the discriminator:
        # 1. Generate fake images
        # 2. Sample some real images
        # 3. Train the discriminator to tell the difference
        random_input = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
        fake = generator.predict(random_input)
        
        real = train_set[i:i+batch_size].reshape(-1, 28, 28, 1)
        
        shuffle_idx = np.arange(2*batch_size)
        np.random.shuffle(shuffle_idx)
        x = np.vstack([fake, real])[shuffle_idx]
        y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x, y)
        
        # Train the generator
        # 1. Generate fake images
        # 2. Train the combined network insisting that the fake images are real
        discriminator.trainable = False
        random_input = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
        g_loss = combined_model.train_on_batch(random_input, np.zeros(batch_size))
        loss.append([d_loss, g_loss])


# In[14]:


get_ipython().run_line_magic('matplotlib', 'tk')


loss = np.array(loss)
plt.figure()
plt.plot(loss[:, 0], label='Discriminator Loss')
plt.plot(loss[:, 1], label='Generator Loss')
plt.legend()
plt.savefig("image1.png")
random_input = np.random.randn(25, 100)
fake = generator.predict(random_input)

fig, ax = plt.subplots(5, 5)
for i in range(len(fake)):
    ax[i%5][i//5].imshow(fake[i, :, :, 0], cmap='gray')
plt.show()
plt.savefig("image2.png")

# In[ ]:




