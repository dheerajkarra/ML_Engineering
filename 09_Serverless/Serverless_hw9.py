#@ IMPORTING LIBRARIES AND DEPENDENCIES:
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img

import tensorflow

tensorflow.__version__
# '2.9.1'

import tensorflow as tf
from tensorflow import keras
import os

#@ Data Preparation:
    
# os.chdir('F:\\Projects\\ML_Engineering\\09_Serverless\\')
os.chdir('F:\\Projects\\ML_Engineering_data\\')

###################################################################
# Question 1

model = keras.models.load_model('dino_dragon_10_0.899.h5')
print(model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('dino-dragon-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path='dino-dragon-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
output_index
13
from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img
img = download_image('https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg')
img = prepare_image(img, target_size=(150, 150))
img


