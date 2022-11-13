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

#@ Data Preparation:
    
os.chdir('F:\\Projects\\ML_Engineering\\08_Deep_Learning\\')

train_dir = 'Data\\train\\'
valid_dir = 'Data\\test\\'

path = train_dir + "dino\\"
name = '00b7f1d3-9265-4971-9c51-4686ce97eadd.jpg'
fullname = f'{path}/{name}'
load_img(fullname)

img = load_img(fullname, target_size=(299, 299))

x = np.array(img)
x.shape
# (299, 299, 3)

###################################################################
# Question 1 - binary crossentropy

# Model Creation
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',\
             optimizer=optimizers.SGD(learning_rate=0.002, momentum=0.8),\
             metrics=['acc'])    
model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 148, 148, 32)      896       
                                                                 
#  max_pooling2d (MaxPooling2D  (None, 74, 74, 32)       0         
#  )                                                               
                                                                 
#  flatten (Flatten)           (None, 175232)            0         
                                                                 
#  dense (Dense)               (None, 64)                11214912  
                                                                 
#  dense_1 (Dense)             (None, 1)                 65        
                                                                 
# =================================================================
# Total params: 11,215,873
# Trainable params: 11,215,873
# Non-trainable params: 0
# _________________________________________________________________

###################################################################
# Question 2
# 11,215,873

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    shuffle=True)

# Found 1594 images belonging to 2 classes.
validation_generator = val_datagen.flow_from_directory(valid_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        shuffle=True)

# Found 394 images belonging to 2 classes.
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# data batch shape: (20, 150, 150, 3)
# labels batch shape: (20,)

# Model fitting and Accuracy/Loss Evaluation
history = model.fit(
    train_generator,
    # steps_per_epoch=50,
    epochs=10,
    validation_data=validation_generator
    # validation_steps=50
    )

# Epoch 1/10
#  80/100 [=======================>......] - ETA: 20s - loss: 0.6022 - acc: 0.6600  WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 1000 batches). You may need to use the repeat() function when building your dataset.
# WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 50 batches). You may need to use the repeat() function when building your dataset.
# 100/100 [==============================] - 106s 992ms/step - loss: 0.6022 - acc: 0.6600 - val_loss: 0.4753 - val_acc: 0.8376

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


###################################################################
# Question 3

acc_median = np.median(acc)
acc_median
# 0.8757545351982117
# 0.831242173910141

###################################################################
# Question 4

loss_std = np.std(loss)
loss_std
# 0.06815432012618164
# 0.16168350790447594

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

from tensorflow.keras.preprocessing import image

path = train_dir + "dino\\"
name = '00b7f1d3-9265-4971-9c51-4686ce97eadd.jpg'
fullname = f'{path}/{name}'
load_img(fullname)


fnames = [os.path.join(train_dir+"dino\\", fname) for
    fname in os.listdir(train_dir+"dino\\")]

img_path = fnames[0]

img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150), 
                                                    batch_size=32, 
                                                    class_mode='binary')
# Found 1594 images belonging to 2 classes.

validation_generator = test_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')
# Found 394 images belonging to 2 classes.

# Model fitting (augmented) and Accuracy/Loss Evaluation
history = model.fit(
    train_generator,
    # steps_per_epoch=50,
    epochs=10,
    validation_data=validation_generator
    # validation_steps=50
    )

acc_aug = history.history['acc']
val_acc_aug = history.history['val_acc']
loss_aug = history.history['loss']
val_loss_aug = history.history['val_loss']

epochs_aug = range(1, len(acc) + 1)

plt.plot(epochs_aug, acc_aug, 'bo', label='Training acc')
plt.plot(epochs_aug, val_acc_aug, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs_aug, loss_aug, 'bo', label='Training loss')
plt.plot(epochs_aug, val_loss_aug, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


###################################################################
# Question 5
loss_mean_aug = np.mean(val_loss_aug)
loss_mean_aug
# 0.5147744506597519

# 0.44 - for validation loss without augumentations
# 0.37

###################################################################
# Question 6
val_acc_aug[5:10]
# [0.8020304441452026,
#  0.779187798500061,
#  0.7944162487983704,
#  0.7715736031532288,
#  0.7411167621612549]

acc_mean_aug = np.mean(val_acc_aug[5:10])
acc_mean_aug
# 0.7776649713516235

# 0.84

