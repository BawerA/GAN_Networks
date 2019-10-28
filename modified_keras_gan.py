#!/usr/bin/env python
# coding: utf-8

# In[1]:

from keras.models import load_model
import os
import keras
from keras.layers import Dense, Input, Reshape, Flatten, Dropout
from keras import models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[2]:



# In[3]:


def build_generator():
    # The generator will take in a vector of length 100
    # and create a 28x28x1 image
    model = models.Sequential()
    model.add(Dense(256, activation='relu', input_dim=100))
    otherway1 = model.get_weights()
    print('\n\nWeights of first Layer: \n\n ', otherway1)
    model.add(Dense(512, activation='relu'))
    otherway2 = model.get_weights()
    print('\n\nWeights of Second Layer: \n\n', otherway2)
    model.add(Dense(784, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    print('\n\n\n Weights:?\n\n\n')
    otherway3 = model.get_weights()
    print('\n\nWeights of Third Layer: \n\n', otherway3)
    print('\n\n\n')
    model.save_weights('Generator_Weights')
    return model

def build_discriminator():
    # The discriminator will take in a 28x28x1 image and
    # try to determine if it was generated or a real image
    model = models.Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_combined(generator, discriminator):
    # The combined model will not train the discriminator.
    # It will be used to train the generator based off of how
    # well the discriminator can tell the generator and real images apart
    discriminator.trainable = False
    
    input_layer = Input((100,))
    gen_image = generator(input_layer)
    dis_output = discriminator(gen_image)
    
    model = models.Model(input_layer, dis_output)
    return model


# In[4]:


discriminator = build_discriminator()
discriminator.compile(optimizer='rmsprop', loss='binary_crossentropy')

generator = build_generator()

combined_model = build_combined(generator, discriminator)                           #Adadelta increases generator loss, only creates '1's
combined_model.compile(optimizer='rmsprop', loss='binary_crossentropy')              #sdg doesnt work, adamax creates fuzzy images 
                                                                                    #nadam has gen loss all over the place at 30 epoch (100 epochs is intersetingfor adagrad), adagrad
                                                                                    #sgd increased loss a lot and very fuzzy images
# In[ ]:                                                                                     rmsprop had the most accurate images, weird gen loss tho(100 epoch)


# saving the weights of 

combined_weights = combined_model.save_weights('Combined.h5')

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_set = (mnist.test.images - 0.5) / 0.5 # Returns np.array


# In[15]:
print('\n\nyoooooooooooooooo\n\n')
print(mnist[0])
print('\n\nyooooooooooo\n\n')
##for layer in keras.layers:
  ##  weights = layer.get_weights() # list of numpy arrays
#print(weights)


total_epochs = 200
batch_size = 100  #could do 500 but too large can decrease accuracy
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
        
        
    


# In[16]:


get_ipython().run_line_magic('matplotlib', 'tk')


loss = np.array(loss)
plt.figure()
plt.plot(loss[:, 0], label='Discriminator Loss')
plt.plot(loss[:, 1], label='Generator Loss')
plt.legend()
plt.savefig("bigbatch_rmsprop.png")
random_input = np.random.randn(25, 100)
fake = generator.predict(random_input)

fig, ax = plt.subplots(5, 5)
for i in range(len(fake)):
    ax[i%5][i//5].imshow(fake[i, :, :, 0], cmap='gray')
plt.show()
plt.savefig("bigbatch_rmsprop2.png")

###############
#model.save('model__.h5')

