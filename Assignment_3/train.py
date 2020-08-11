# Tensorflow
#from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers

import math
import pandas as pd
import os
import custom_metric

dirpath = os.getcwd()
# load the csv file and add the absolute path to the files
df = pd.read_csv('PlantData_ranked.csv')
df['label'] = df['label'].astype(str)
df['id'] = dirpath+'/'+df['id']

# this folder should contain the images
data_dir = dirpath+'/'+'dataset'

# get the first 20 classes and split the dataset in training, validation and testing
firstclasses = df['label'].value_counts()[:20].index.tolist()
df_n = df[df['label'].isin(firstclasses)]
train=df_n.sample(frac=0.8,random_state=200) #random state is a seed value
test=df_n.drop(train.index)

datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input, validation_split=0.2)
# Train generator
train_generator = datagen.flow_from_dataframe(
    dataframe=train,
    directory=data_dir,
    x_col="id",
    y_col="label",
    has_ext=False,
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224))

# validation generator
val_generator = datagen.flow_from_dataframe(
            dataframe=train,
            directory=data_dir,
            x_col="id",
            y_col="label",
            has_ext=False,
            subset="validation",
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=(224,224))

datagen_test = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
# testing generator
test_generator = datagen_test.flow_from_dataframe(
    dataframe=test,
    directory=data_dir,
    x_col="id",
    y_col="label",
    has_ext=False,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(224,224))

training_steps = math.ceil(train_generator.n / 32)
validation_steps = math.ceil(val_generator.n / 32)
test_steps = math.ceil(test_generator.n / 32)

# get the labels and save them in a csv to be used in the custom metrics
labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
test_label_df = pd.DataFrame.from_dict(labels, orient='index',columns=['keras_index'])
test_label_df.to_csv(dirpath+'/keras_dict.csv')


# load model and freeze the weights in the convolutional layers
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
for layer in base_model.layers:
    layer.trainable = False

# add the last fully connected layers
x = Flatten()(base_model.output)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
prediction = Dense(20, activation='softmax')(x)

save_path = dirpath+'/saved_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(save_path, monitor='val_acc', verbose=1, mode='max')
callbacks_list = [checkpoint]


rms_prop = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(optimizer=rms_prop, loss='categorical_crossentropy',metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=training_steps, epochs=20,
                validation_data=val_generator,validation_steps=validation_steps,callbacks=callbacks_list, verbose=1)

# evaluate the test dataset
test_generator.reset()
pred=model.predict_generator(test_generator,steps=test_steps)

custom_metric.first_metric()
