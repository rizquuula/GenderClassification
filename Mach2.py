import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

# Images Dimensions
img_width, img_height = 128, 128

train_data_dir = 'Train'
validation_data_dir = 'Validation'
nb_train_samples = 73*2
nb_validation_samples = 17*2
epochs = 50
batch_size = 16

# TensorBoard Callbacks
callbacks = TensorBoard(log_dir='./Graph')

# Training Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Rescale Testing Data
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Train Data Generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Testing Data Generator
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
    
# Feature Extraction Layer
inputs = Input(shape=(img_width, img_height, 3))
conv_layer = Conv2D(16, (5, 5), strides=(2,2), activation='relu')(inputs) 
conv_layer = ZeroPadding2D(padding=(1,1))(conv_layer) 
conv_layer = Conv2D(32, (5, 5), strides=(1,1), activation='relu')(conv_layer) 
conv_layer = MaxPooling2D((2, 2))(conv_layer) 
conv_layer = Conv2D(64, (3, 3), strides=(1,1), activation='relu')(conv_layer) 
conv_layer = Conv2D(64, (3, 3), strides=(1,1), activation='relu')(conv_layer)

# Flatten Layer
flatten = Flatten()(conv_layer) 

# Fully Connected Layer
fc_layer = Dense(64, activation='relu')(flatten)
fc_layer = Dense(32, activation='relu')(fc_layer)
outputs = Dense(1, activation='sigmoid')(fc_layer)

model = Model(inputs=inputs, outputs=outputs)

# Adam Optimizer and Cross Entropy Loss
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# Print Model Summary
print(model.summary())

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size, 
    callbacks=[callbacks])

model.save_weights('kornet.h5')
model.save('kornet.model')