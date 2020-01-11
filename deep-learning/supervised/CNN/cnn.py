from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import tensorflow as tf

# init model
with tf.device('/gpu:0'):
    model = Sequential()
# convolution 1
model.add(Conv2D(filters = 32,
                 kernel_size = 3,
                 strides = 1,
                 padding = "same",
                 activation = "relu",
                 input_shape=(128, 128, 3)))
# max pooling 1
model.add(MaxPool2D(pool_size=(2,2)))

# convolution 2
model.add(Conv2D(filters = 32,
                 kernel_size = 3,
                 strides = 1,
                 padding = "same",
                 activation = "relu"))
# max pooling 1
model.add(MaxPool2D(pool_size=(2,2)))

# convolution 2
model.add(Conv2D(filters = 64,
                 kernel_size = 3,
                 strides = 1,
                 padding = "same",
                 activation = "relu"))
# max pooling 1
model.add(MaxPool2D(pool_size=(2,2)))

# flatenning
model.add(Flatten())

# create a classic ANN to classify convoluted, pooled and flattened img
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate = 0.5))
# as we have only 2 classes no need for softmax
model.add(Dense(1, activation="sigmoid"))

# compile the model
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=['accuracy'])

# fit model on images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=90,
        validation_data=test_set,
        validation_steps=2000/32)

#make predictions on single image
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img(
        'dataset/single_prediction/cat_or_dog_1.jpg', target_size=(128, 128))
test_image = image.img _to_array(test_image) # to make it 64x64x3
test_image = np.expand_dims(test_image, axis=0) # add dimention of batch as a first dim

result = model.predict(test_image)
training_set.class_indices # get mapping between number and classes

