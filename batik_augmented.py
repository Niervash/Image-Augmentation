# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:50:02 2024

@author: Eve
"""

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
print(tf.__version__)


datagen = ImageDataGenerator(
    rotation_range = 45,  #random rotation beetween 0 and 45
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect')  # u can also try nearest,constant,reflect,wrap (for constant u musd add 1 more parameter "cval=125" > 125 mean what color 125 is gray)

i = 0
for batch in datagen.flow_from_directory(
        directory='batik_skripsi/',
        target_size=(150,150),
        batch_size=15,
        color_mode='rgb',
        save_to_dir='data_augmented',
        save_prefix='aug',
        save_format='png'):
    i += 1
    if i > 150:
        break