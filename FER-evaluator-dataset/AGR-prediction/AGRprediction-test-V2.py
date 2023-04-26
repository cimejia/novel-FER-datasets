# Predicting age, gender and race on FER2013, NHFI, and AffectNet datasets
import numpy as np
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

IM_WIDTH = IM_HEIGHT = 198
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = dict((r, i) for i, r in ID_RACE_MAP.items())

from keras.models import Model

path = '/mnt/datosnas/Users/cmejia/EMOTION-DETECTION/NHFI-NUEVO/' # Replace with path to each dataset (FER2013, NHFI, AffectNet)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen_test = ImageDataGenerator(rescale = 1/255.0)
test_generator = datagen_test.flow_from_directory(path,
                                                    target_size = (IM_HEIGHT,IM_WIDTH),
                                                    color_mode = 'rgb',
                                                    batch_size = 1,
                                                    classes = ['6'], # Replace with each emotion category (0 to 6)
                                                    shuffle = False)

from tensorflow.python.keras import models
best_model = models.load_model('best_model_agr.h5')
age_pred, race_pred, gender_pred = best_model.predict(test_generator, verbose=1, steps=len(test_generator.filenames))

max_age = 116 # Maximum value of the age in the training dataset (UTKFace)
race_pred, gender_pred = race_pred.argmax(axis=-1), gender_pred.argmax(axis=-1)

import csv
f = open('SURPRISE-results.csv', 'w') # Create a csv predictions file for each emotion category

i = 0
for file in test_generator.filenames:
    cat, name = file.split('/')
    print(name)
    age_pred[i] = age_pred[i] * max_age
    age = int(age_pred[i][0])
    if age<15:
        cat_age = 'child'
    elif age>=15 and age<30:
        cat_age = 'young'
    elif age>=30 and age<65:
        cat_age = 'adult'
    else:
        cat_age = 'old'
    race = ID_RACE_MAP[race_pred[i]]
    gender = ID_GENDER_MAP[gender_pred[i]]
    row = name + ',' + cat_age + ',' + race + ',' + gender + '\n'
    f.write(row)
    i = i+1
