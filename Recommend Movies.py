
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ds = pd.read_csv("/Users/Administrator/Documents/Machine Learning/RNN/Movie Recommendation/ratings.csv")


# In[3]:


ds.userId = ds.userId.astype('category').cat.codes.values
ds.movieId = ds.movieId.astype('category').cat.codes.values


# In[4]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(ds, test_size=0.2)


# In[56]:


import keras
from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from keras.layers import Dense, Activation
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


# In[9]:


n_users, n_movies = len(ds.userId.unique()), len(ds.movieId.unique())
n_latent_factors_user = 5
n_latent_factors_movie = 8


# In[20]:


movie_input = keras.layers.Input(shape=[1],name='Movie')
movie_embedding=keras.layers.Embedding(n_movies + 1, n_latent_factors_movie, name='Movie-Embedding')(movie_input)
movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)
model_movie = Dense(200)(movie_vec)
model_movie = Activation('relu')(model_movie)
model_movie = keras.layers.Dropout(0.2)(model_movie)


# In[19]:


user_input = keras.layers.Input(shape=[1],name='User')
User_Embedding = keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input)
user_vec = keras.layers.Flatten(name='FlattenUsers')(User_Embedding)
model_user = Dense(200)(user_vec)
model_user = Activation('relu')(model_user)
model_user = keras.layers.Dropout(0.2)(model_user)


# In[27]:


model_user_movie = keras.layers.concatenate([model_movie, model_user])


# In[29]:


dense = keras.layers.Dense(200,name='FullyConnected')(model_user_movie)
dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
dense_2 = keras.layers.Dense(100,name='FullyConnected-1')(model_user_movie)
dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
dense_3 = keras.layers.Dense(50,name='FullyConnected-2')(dense_2)
dropout_3 = keras.layers.Dropout(0.2,name='Dropout')(dense_3)
dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)
result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)


# In[53]:


adam = Adam(lr=0.005)
model = keras.Model([user_input, movie_input], result)
model.compile(optimizer=adam,loss= 'mean_absolute_error', metrics=['accuracy'])


#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])


# In[58]:


model.summary()


# In[60]:


filename = '/Users/Administrator/Documents/Machine Learning/RNN/Movie Recommendation/RecommendMovies.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
model.fit([train.userId, train.movieId], train.rating,
          batch_size=10,
          epochs=3,
          verbose=1,
          validation_data=([test.userId, test.movieId], test.rating))
score = model.evaluate([test.userId, test.movieId], test.rating, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save('/Users/Administrator/Documents/Machine Learning/RNN/Movie Recommendation/RecommendMovies.h5')


# In[61]:


test_model = load_model('/Users/Administrator/Documents/Machine Learning/RNN/Movie Recommendation/RecommendMovies.h5')


# In[62]:


train.userId

