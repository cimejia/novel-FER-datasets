import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#from tensorflow.keras.utils import plot_model

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="0"
path = '/mnt/datosnas/Users/cmejia/EMOTION-DETECTION/MIXED-STABLE-DIFFUSION/TEST/'

img_size = 224
channels = 3
batch_size = 1

#def rgb_to_gray(img):
#    img = tf.image.rgb_to_grayscale(img)
#    return img

datagen = ImageDataGenerator(
        #horizontal_flip=True,
        #rescale=1./255,
        #preprocessing_function=rgb_to_gray,
        #validation_split=0.2
        )


test_generator = datagen.flow_from_directory(path,
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    #subset='training'
                                                    )
'''
validation_generator = datagen.flow_from_directory(path,
                                                    target_size=(img_size,img_size),
                                                    color_mode="grayscale",
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='validation')

# Initialising the CNN

model = Sequential()

model.add(Conv2D(64,(3,3), padding='same', input_shape=(img_size,img_size,channels)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))
model.summary()

opt = Adam(learning_rate=0.0003)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50
steps_per_epoch = train_generator.n//train_generator.batch_size
validation_steps = validation_generator.n//validation_generator.batch_size

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, mode='auto')
checkpoint = ModelCheckpoint("best_model_nhfi.h5", monitor='val_accuracy', save_best_only=True, save_weights_only=False, mode='max', verbose=1)
callbacks = [checkpoint, reduce_lr]

history = model.fit(
    x = train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    callbacks = callbacks
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()),1])
plt.ylim([0,1])
plt.title('Training and Validation Accuracy')
#plt.savefig('Accuracy.png')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0,5])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("LearningCurves.eps")
#plt.show()

input()
'''
from tensorflow.python.keras import models

best_model = tf.keras.models.load_model('best_model_nhfi.h5')

eval_test = best_model.evaluate(test_generator, verbose=1, steps=len(test_generator.filenames))
print("Test Acc: ", eval_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = best_model.predict(test_generator, verbose=1, steps=len(test_generator.filenames))
pred = np.argmax(pred, axis=-1)

cm = confusion_matrix(test_generator.classes, pred)
print(cm)
cm2 = np.around(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis], decimals=2)
print(cm2)

from plotConfusionMatrix import plot_confusion_matrix
emotions = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
cm3 = plot_confusion_matrix(cm, emotions, normalize=True)
