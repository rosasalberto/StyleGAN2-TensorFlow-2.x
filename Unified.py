#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as r

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
from matplotlib.pyplot import imshow
import dlib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error,logcosh
from tensorflow.keras.optimizers import Adam

from PIL import Image
from matplotlib.pyplot import imshow
from tqdm import tqdm
from time import time

from utils.utils_stylegan2 import convert_images_to_uint8
from stylegan2_generator import StyleGan2Generator

from sklearn.model_selection import train_test_split
from scipy.spatial import distance
import matplotlib.pylab as plt
import json

import os
import subprocess
import re
import sys
import imageio


# In[78]:


filename = sys.argv[1]
# filename = "jude" #for jupyter notebook testing
print

# ### First prep and load data and models

# In[48]:


detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")
vgg16=VGG16(include_top=False,input_shape=(224,224,3),weights='imagenet')
vggmod=Model(vgg16.input, vgg16.layers[6].output)
fnt = tf.keras.models.load_model("facenet_keras_new.h5")
truncation_psi = 0.5
w_average = np.load('weights/{}_dlatent_avg.npy'.format("ffhq"))
generator = StyleGan2Generator(weights="ffhq", impl="ref", gpu=False)


# In[3]:


class Facenet(Model):
    def __init__(self,model):
        super(Facenet,self).__init__(name = 'faceoff')
        self.gen = model
    def call(self,inputs):
        x = inputs
        x = (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)
        x = self.gen(x)
        return x

class VGG(Model):
    def __init__(self,model):
        super(VGG,self).__init__(name = 'my_model')
        self.gen = model
    def call(self,inputs):
        x = self.gen(inputs)
        return x


# In[4]:


vgg = VGG(vggmod)
vgg.trainable = False
facenet = Facenet(fnt)
facenet.trainable = False


# In[20]:


data = pd.read_json("data.json")
directions = pd.read_json('learned_directions_w.json')
descriptors = np.load("seeds.npy")


# ### Then initialize some functions

# In[43]:


def wpize(n):
    rnd = np.random.RandomState(n)
    z = rnd.randn(1, 512).reshape(1,512)
    w = generator.mapping_network(z.astype('float32'))
    wp = np.array(w[0][1]).reshape(1,512)
    return wp

def generateFromWP(wp):
    wp = tf.convert_to_tensor(np.repeat(wp.reshape((1,512)),18,0).reshape((1,18,512)))
    dlatents = w_average + (wp - w_average) * truncation_psi
    out = generator.synthesis_network(dlatents)
    int8 = convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)
    img = Image.fromarray(int8.numpy()[0])
    return img

def displaySideBySide(im1,im2):
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(np.rot90(im1,4))
    f.add_subplot(1,2, 2)
    plt.imshow(np.rot90(im2,4))
    plt.show(block=True)

def findClosestKnownImage(features):
    features = features.numpy()
    min_seed_at = 0
    min_seed = 1000
    for i in range(1,30001):
        dist = distance.euclidean(descriptors[i].reshape(1,512),features)
        min_seed_at = i if dist < min_seed else min_seed_at
        min_seed = min(dist,min_seed)
        if dist <= min_seed:
            print("Found ",i," with distance ",min_seed)
    img = generateFromWP(wpize(min_seed_at))
    return min_seed_at,img


# In[44]:


### Function to align image and output an image similar to FFHQ
def align(img, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
        imgarray = np.array(img)
        dets = detector(imgarray, 1)
        land_list = []
        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in shape_predictor(imgarray, detection).parts()]
                land_list.append(face_landmarks)
            except Exception as e:
                print("Exception in get_landmarks()!",e)
        lm = np.array(land_list[0])
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= x_scale
        y = np.flipud(x) * [-y_scale, y_scale]
        c = eye_avg + eye_to_mouth * em_scale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = np.uint8(np.clip(np.rint(img), 0, 255))
            if alpha:
                mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
                mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
                img = np.concatenate((img, mask), axis=2)
                img = Image.fromarray(img, 'RGBA')
            else:
                img = Image.fromarray(img, 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), Image.ANTIALIAS)

        return img


# ### Setup and execute gradient descent loop

# In[56]:


raw = Image.open(filename+".jpg")


# In[57]:


image = align(raw)
desc = facenet(np.array(image.crop((200,200,864,924)).resize((160,160))).astype('float32').reshape(1,160,160,3))


# In[58]:



seed,img = findClosestKnownImage(desc)
enco = wpize(seed)
label = vgg(preprocess_input(np.array(image.resize((224,224))).reshape(1,224,224,3)))
vggweights = vggmod.get_weights()[-1]

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.029,decay_steps=50,decay_rate=0.9)
optimizer = Adam(learning_rate=0.100,amsgrad=False) # Create Adam optimizer to iterate


# In[83]:


ssot = tf.constant(label)
descriptor = tf.constant(desc)
weights = tf.constant(vggweights)
X = tf.Variable(enco.reshape(1,1,512))
character_losses = []
perceptual_losses = []
discovered_latents = []
total_losses = []
for i in range(200):
    with tf.GradientTape() as tape:
        tape.watch(X)
        q = tf.constant(w_average) + (tf.repeat(X,18,1) - tf.constant(w_average)) * tf.constant(truncation_psi)
        # q = tf.repeat(X,18,1)
        o = generator.synthesis_network(q)
        z = tf.transpose(o, [0, 2, 3, 1])
        z = z * 127.5 + (0.5 + 1.0 * 127.5)
        r = tf.image.resize(z,[224,224])
        p = preprocess_input(r)
        y = vgg(p)
        t = tf.image.central_crop(r,0.714)
        d = facenet(t)
        percept_loss = tf.math.reduce_mean(tf.math.abs(weights*y-weights*ssot))
        percept_loss = percept_loss + tf.math.reduce_mean(tf.keras.losses.logcosh(weights*y,weights*ssot))
        char_loss = tf.norm(d-descriptor,ord='euclidean')*100.0 
        total_loss = 0.05*(1 - (1/tf.math.log(char_loss)))*char_loss + 0.99*percept_loss #fancy formula for character loss to deemphasize lower values
        if(i%10 == 0):
            print(str(i),"th iteration",percept_loss.numpy().round(2),char_loss.numpy().round(2),total_loss.numpy().round(2))
        if char_loss < percept_loss and percept_loss <= 30:
            break
    gradients = tape.gradient(total_loss, X)
    grads_and_vars = zip([gradients],[X])
    optimizer.apply_gradients(grads_and_vars)
    discovered_latents.append(X.numpy())
    perceptual_losses.append(percept_loss.numpy())
    character_losses.append(char_loss.numpy())
    total_losses.append(total_loss.numpy())


# In[80]:


best_latent = discovered_latents[-1]
np.save("generated.npy",best_latent.reshape(1,512))
best_image = generateFromWP(best_latent.reshape(1,512))


# In[81]:


displaySideBySide(generateFromWP(discovered_latents[190].reshape(1,512)),generateFromWP(discovered_latents[-1].reshape(1,512)))


# In[82]:


imgarr = []
nparr = []
for i in tqdm(range(-50,50)):
    img = generateFromWP(best_latent.reshape(1,512) + (i/100.0) * np.array([directions['age'].loc[0]]))
    imgarr.append(img)
    nparr.append(np.array(img))


# In[ ]:


imageio.mimsave(filename+'.gif', nparr,fps=10) #Generate a 10s GIF


# In[ ]:




