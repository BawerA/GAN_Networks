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
import glob
import mahotas as mh
#from chainer.dataset import dataset_mixin
from tqdm import tqdm
# In[2]:



# In[3]:

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
    print("\n\nthink it works\n\n") 
    return coords, chips, classes




#class LabeledImageDataset(dataset_mixin.DatasetMixin):
 #   def __init__(self, dataset, root, label_root, dtype=np.float32,
  #               label_dtype=np.int32, mean=0, crop_size=256, test=False,
   #              distort=False):
    #    #_check_pillow_availability()
     #   if isinstance(dataset, six.string_types):
      #      dataset_path = dataset
       #     with open(dataset_path) as f:
        #        pairs = []
         #       for i, line in enumerate(f):
          #          line = line.rstrip('\n')
           #         image_filename = line
            #       label_filename = line
             #       pairs.append((image_filename, label_filename))
        #self._pairs = pairs
        #self._root = root
        #self._label_root = label_root
        #self._dtype = dtype
        #self._label_dtype = label_dtype
        #self._mean = mean[np.newaxis, np.newaxis, :]
        #self._crop_size = crop_size
        #self._test = test
#self._distort = distort



thechips = []
theclasses = []
thecoords = []
thecoords, thechips, theclasses = get_labels()


print(thecoords[0])
print(thechips[0])
print(theclasses[0])

#yo = chip_image('2355.tif', thecoords[0], theclasses[0])
chip_name = '104.tif'
arr = wv.get_image(chip_name)

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(arr)
print('\nyo\n')




c_img, c_box, c_cls = wv.chip_image(img = arr, coords=thecoords, classes=theclasses, shape=(1000,1000))
print("Num Chips: %d" % c_img.shape[0])

print("\nnumber of chips : ", c_img.shape[0])

w = c_img.shape[0]

#We can plot some of the chips
fig,ax = plt.subplots(3)
fig.set_figheight(5)
fig.set_figwidth(5)

for k in range(9):
    plt.subplot(3,3,k+1)
    plt.axis('off')
    plt.imshow(c_img[np.random.choice(range(c_img.shape[0]))])
    #plt.imsave(fname= 'png', arr= c_img[k])

##############################################################################
#chipped = Image("chipped_images")

##########     http://tutorial.simplecv.org/en/latest/examples/basics.html  ^^^^^^^^^^^
per = 1
X_data = []
files2 = glob.glob ("/root/Desktop/seniordesign/chippedimages/*.tif")
files = glob.glob ("/root/Desktop/seniordesign/testing_images/*.tif")
for myFile in files:
    t = 0
    print('\nChipping image: ', myFile)
    image = cv2.imread (myFile)
    X_data.append (image) #                                                         https://stackoverflow.com/questions/37747021/create-numpy-array-of-images
    chipped_img, chipped_box, chipped_classes = wv.chip_image(img = image, coords = thecoords, classes=theclasses, shape=(1000,1000))
    numberOfChips = chipped_img.shape[0] 
    while t < numberOfChips:
        print(t + 1)
        #chipped_img.save("/root/Desktop/seniordesign/chippedimages")
        os.chdir(r"/root/Desktop/seniordesign/chippedimages") 
        mh.imsave('%d.tif' % per, chipped_img[t])
        os.chdir(r"/root/Desktop/seniordesign")
        t += 1
        per += 1

#print('X_data shape:', np.array(X_data).shape)


################################################################################

npchipped = np.array([np.array(Image.open(myFile)) for myFile in files2]) ### This puts all of the images in to one nparray, ( i think the block of code above does the same thing)
print('\nyo\n') #                                                                   https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array
##############################################################################

#plt.show()





#################################################################################################################################################################################

#################################################################################################################################################################################

#################################################################################################################################################################################

#################################################################################################################################################################################

#################################################################################################################################################################################

#################################################################################################################################################################################

#################################################################################################################################################################################

def build_generator():
    # The generator will take in a vector of length 100
    # and create a 28x28x1 image
    model = models.Sequential()
    model.add(Dense(256, activation='relu', input_dim=1000))
    #otherway1 = model.get_weights()
    #print('\n\nWeights of first Layer: \n\n ', otherway1)
    model.add(Dense(512, activation='relu'))
   # otherway2 = model.get_weights()
    #print('\n\nWeights of Second Layer: \n\n', otherway2)
    model.add(Dense(784, activation='tanh'))
   # model.add(Reshape((1000, 1000, 1)))
    #print('\n\n\n Weights:?\n\n\n')
    #otherway3 = model.get_weights()
   # print('\n\nWeights of Third Layer: \n\n', otherway3)
    #print('\n\n\n')
    #model.save_weights('Generator_Weights')
    return model

def build_discriminator():
    # The discriminator will take in a 28x28x1 image and
    # try to determine if it was generated or a real image
    model = models.Sequential()
    model.add(Flatten(input_shape=(1000, 1000, 1)))
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
    
    input_layer = Input((1000,))
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

#mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#rain_set = (mnist.test.images - 0.5) / 0.5 # Returns np.array
train_set = npchipped

# In[15]:
print('\n\nyoooooooooooooooo\n\n')
#print(mnist[0])
print('\n\nyooooooooooo\n\n')
##for layer in keras.layers:
  ##  weights = layer.get_weights() # list of numpy arrays
#print(weights)


total_epochs = 10
batch_size = 50 #could do 500 but too large can decrease accuracy
loss = []
for epoch in range(total_epochs):
    
    for i in range(0, train_set.shape[0], batch_size):
        # Train the discriminator:
        # 1. Generate fake images
        # 2. Sample some real images
        # 3. Train the discriminator to tell the difference
        random_input = np.random.uniform(-1.0, 1.0, size=(batch_size, 1000))
        fake = generator.predict(random_input)
        
        real = train_set[i:i+batch_size].reshape(-1, 1000, 1000, 1)
        
        shuffle_idx = np.arange(2*batch_size)
        np.random.shuffle(shuffle_idx)
        x = np.vstack([fake, real])[shuffle_idx]   # Currently getting an error for this line
        y = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])[shuffle_idx]
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(x, y)
        
        # Train the generator
        # 1. Generate fake images
        # 2. Train the combined network insisting that the fake images are real
        discriminator.trainable = False
        random_input = np.random.uniform(-1.0, 1.0, size=(batch_size, 1000))
        g_loss = combined_model.train_on_batch(random_input, np.zeros(batch_size))
        loss.append([d_loss, g_loss])
            
        
    

#tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create graph
#a = tf.constant(2, name="a")
#b = tf.constant(3, name="b")
#c = tf.add(a, b, name="addition")
# creating the writer out of the session#
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# launch the graph in a session
#with tf.Session() as sess:
    # or creating the writer inside the session
  #  writer = tf.summary.FileWriter('./graphs', sess.graph)
 #   print(sess.run(c))
# In[16]:


get_ipython().run_line_magic('matplotlib', 'tk')  #  changed it from inline to tk  (event loop)


loss = np.array(loss)
print(loss)  #  This line is meant for debugging, was getting an error for the loss array before
#plt.figure()
plt.plot(loss[:, 0], label='Discriminator Loss')
plt.plot(loss[:, 1], label='Generator Loss')
plt.legend()
plt.savefig("bigbatch_rmsprop.png")
random_input = np.random.randn(1000, 1000)
fake = generator.predict(random_input)

fig, ax = plt.subplots(5, 5)
for i in range(len(fake)):
    ax[i%5][i//5].imshow(fake[i, :, :, 0], cmap='gray')
#plt.show()
plt.savefig("bigbatch_rmsprop2.png")

###############
#model.save('model__.h5')

