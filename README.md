# GAN_Networks
This is Team Machines GitHub for GAN networks.


dcgan.py - -  Our code so far in trying to create our own DCGAN based off of the architecture from the medium article (https://medium.com/@liyin2015/dcgan-79af14a1c247)



modified_keras_gan.py - - This is our heavily modified code based off of the sample code given to us. We did a lot of playing around to see how using different optimizers, activations, batch sizes, and epoch values would affect the generator/discriminator loss and the accuracy of the images created. 




https://github.com/DIUx-xView/data_utilities
xview2.py  - November 11th, 2019 -  Where the magic happened :)   - - Trying to import images from the xView dataset and then chipping those images in to smaller images. In the code, we chipped them to 1000x1000 but this can easily be changed. We will most likely go with 224x224 to start and then slowly start to input and generate larger images. Probably the size of 600x600 chip or larger. We are implementing this code in our "Keras-GAN.py" program for now instead of the "Keras_DCGAN.py". This is because the latter program takes a much longer time to run and at this moment, we are mostly going through trial and error with what we have right now. After we can plot generator and discriminator loss and generate a sample of fake images, we will move on to the DCGAN program and alter that architecture. We are currently assuming that at first, we will generate low quality images and the goal after that is to deepen our architecture to improve the quality. If our customer or we are not satisfied with the results, we'll change the architecture completely and go with a different approach. This means we will use a different type of GAN. 
  Later on, we will upload our prototype code that generates images from the xView dataset. That will be uploaded as soon as that happens. From there, we will upload all of our improved GANs as well. 

chip_xview.py - - This is the program that was modified from "Keras-GAN.py".
wv_util.py - -    This is the program that is imported in to chip_xview.py


November 21st, 2019 - chip_script.py - : 
