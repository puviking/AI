
# coding: utf-8

# In[5]:


import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
#from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tensorflow as tf    #Import Tensorflow
import glob                #This will extract all files from the folder
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import h5py
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
import pyaudio  
import wave  


# In[13]:

img_width, img_height = 200, 200


# In[16]:


train_data_dir = '/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/Training'
validation_data_dir = '/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/Testing'
nb_train_samples = 3000            #Get this in step 2. These are examples per subfolder inside train data folder
nb_validation_samples = 500       #Get this in step 2. These are examples per subfolder inside validation data folder
epochs = 3                     #Define number of epochs
batch_size = 10                  #Define batch size. This should be less than the total number of examples in validation and training 


# In[17]:


if K.image_data_format() == 'channels_first':        #We usually use channel first approach
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[46]:


model=Sequential()

model.add(Conv2D(128, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[47]:


train_datagen = ImageDataGenerator(rescale=1. / 255)  #We can augment the number of images with ImageDataGenerator #Google to know more
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')             

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/PredictTamilLetters.h5')


# In[18]:


print("Predicting...")
test_model = load_model('/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/PredictTamilLetters.h5')
#img = load_img('/Users/Administrator/Documents/Machine Learning/CNN/MNIST/Images/mnist_png/TestingSet1/2/237.png',False,target_size=(img_width,img_height))
img1 = cv2.imread('/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/O.png')
img1 = cv2.resize(img1, (200,200))
x = img_to_array(img1)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)
if preds == [4]:
    print("The Tamizh letter identified is O")
    chunk = 1024  
    f = wave.open(r"/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/O.wav","rb")  
    p = pyaudio.PyAudio()  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
    data = f.readframes(chunk)  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
    stream.stop_stream()  
    stream.close()  
    p.terminate()
elif preds == [3]:
    print("The Tamizh letter identified is A")
    chunk = 1024  
    f = wave.open(r"/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/EA.wav","rb")  
    p = pyaudio.PyAudio()  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
    data = f.readframes(chunk)  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
    stream.stop_stream()  
    stream.close()  
    p.terminate()
elif preds == [2]:
    print("The Tamizh letter identified is E")
    chunk = 1024  
    f = wave.open(r"/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/E.wav","rb")  
    p = pyaudio.PyAudio()  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
    data = f.readframes(chunk)  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
    stream.stop_stream()  
    stream.close()  
    p.terminate()
elif preds == [1]:
    print("The Tamizh letter identified is AAA")
    chunk = 1024  
    f = wave.open(r"/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/AA.wav","rb")  
    p = pyaudio.PyAudio()  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
    data = f.readframes(chunk)  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
    stream.stop_stream()  
    stream.close()  
    p.terminate()
else:
    chunk = 1024  
    print("The Tamizh letter identified is AA")
    f = wave.open(r"/Users/Administrator/Documents/Machine Learning/CNN/Tamil Dataset/A.wav","rb")  
    p = pyaudio.PyAudio()  
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
    data = f.readframes(chunk)  
    while data:  
        stream.write(data)  
        data = f.readframes(chunk)  
    stream.stop_stream()  
    stream.close()  
    p.terminate()
