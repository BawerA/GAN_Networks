from PIL import Image
import tensorflow as tf
import io
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import skimage.filters as filters
from PIL import Image
import tensorflow as tf
import io
import glob
from tqdm import tqdm
import numpy as np
import logging
import argparse
import os
import json
import wv_util as wv
#import tfr_util as tfr
#import aug_util as aug
import csv
import matplotlib.pyplot as plt
import random
import six
import cv2
import glob
import numpy as np
import mahotas as mh

#from chainer.dataset import dataset_mixin
from tqdm import tqdm

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
    print("\n\nHey nice job, i think it works\n\n") 
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
#files2 = glob.glob ("/root/Desktop/seniordesign/chippedimages/*.tif")
files = glob.glob ("/root/Desktop/seniordesign/testing_images/*.tif")
for myFile in files:
    t = 0
    print('Chipping image: ', myFile)
    image = cv2.imread (myFile)
    X_data.append (image) # 														https://stackoverflow.com/questions/37747021/create-numpy-array-of-images
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

print('X_data shape:', np.array(X_data).shape)


################################################################################

nphomie = np.array([np.array(Image.open(myFile)) for myFile in files]) ### This puts all of the images in to one nparray, ( i think the block of code above does the same thing)
print('\nyo\n') # 																	https://stackoverflow.com/questions/39195113/how-to-load-multiple-images-in-a-numpy-array
##############################################################################

plt.show()

