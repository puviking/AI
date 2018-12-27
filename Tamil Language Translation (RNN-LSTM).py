
# coding: utf-8

# In[162]:


import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from pickle import load
from numpy.random import rand
from numpy.random import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from numpy import argmax
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


# # Clean Text

# In[7]:


filename = '/Users/Administrator/Documents/Machine Learning/RNN/tamil-eng/tamil.txt'
file = open(filename, mode='rt', encoding='utf-8')
text = file.read()
file.close()


# In[8]:


lines = text.strip().split('\n')
pairs = [line.split('\t') for line in  lines]


# In[67]:


cleaned = list()
re_print = re.compile('[^%s]' % re.escape(string.printable))
table = str.maketrans('', '', string.punctuation)
for i in pairs:
    clean_pair = list()
    for line in i:
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        line = line.split()
        line = [word.lower() for word in line]
        line = [word.translate(table) for word in line]
        line = [re_print.sub('', w) for w in line]
        line = [word for word in line if word.isalpha()]
        clean_pair.append(' '.join(line))
    cleaned.append(clean_pair)
cleaned = array(cleaned)


# In[68]:


dump(cleaned, open('/Users/Administrator/Documents/Machine Learning/RNN/tamil-eng/english-tamil.pkl', 'wb'))


# # Split Text

# In[110]:


raw_dataset = load(open('/Users/Administrator/Documents/Machine Learning/RNN/tamil-eng/english-tamil.pkl', 'rb'))
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
shuffle(dataset)
train, test = dataset[:9000], dataset[9000:]


# In[70]:


dump(dataset, open('/Users/Administrator/Documents/Machine Learning/RNN/tamil-eng/english-tamil-both.pkl', 'wb'))
dump(train, open('/Users/Administrator/Documents/Machine Learning/RNN/tamil-eng/english-tamil-train.pkl', 'wb'))
dump(test, open('/Users/Administrator/Documents/Machine Learning/RNN/tamil-eng/english-tamil-test.pkl', 'wb'))


# # Train Neural Translation Model

# In[106]:


engtokenizer = Tokenizer()
engtokenizer.fit_on_texts(dataset[:,0])
eng_vocab_size = len(engtokenizer.word_index) + 1
eng_length=max(len(line.split()) for line in dataset[:, 0])
print('English Vocab size:', eng_vocab_size)
print('English Max Length:', eng_length)

tamiltokenizer = Tokenizer()
tamiltokenizer.fit_on_texts(dataset[:,1])
tamil_vocab_size = len(tamiltokenizer.word_index) + 1
tamil_length=max(len(line.split()) for line in dataset[:, 0])
print('Tamil Vocab size:', tamil_vocab_size)
print('Tamil Max Length:', tamil_length)


# In[158]:


Y = engtokenizer.texts_to_sequences(train[:, 0])
trainy = pad_sequences(Y, maxlen=eng_length, padding='post')


X = tamiltokenizer.texts_to_sequences(train[:, 1])
trainX = pad_sequences(X, maxlen=tamil_length, padding='post')

ylist = list()
for i in trainy:
    encoded = to_categorical(i, num_classes=eng_vocab_size)
    ylist.append(encoded)
y1 = array(ylist)
trainY = y1.reshape(trainy.shape[0], trainy.shape[1], eng_vocab_size)


# In[159]:


Y = engtokenizer.texts_to_sequences(test[:, 0])
testy = pad_sequences(Y, maxlen=eng_length, padding='post')


X = tamiltokenizer.texts_to_sequences(test[:, 1])
testX = pad_sequences(X, maxlen=tamil_length, padding='post')

ylist = list()
for i in testy:
    encoded = to_categorical(i, num_classes=eng_vocab_size)
    ylist.append(encoded)
y1 = array(ylist)
testY = y1.reshape(testy.shape[0], testy.shape[1], eng_vocab_size)


# In[164]:


model = Sequential()
model.add(Embedding(tamil_vocab_size, 256, input_length=tamil_length, mask_zero=True))
model.add(LSTM(256))
model.add(RepeatVector(eng_length))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(eng_vocab_size, activation='softmax')))
          
model.compile(optimizer='adam', loss='categorical_crossentropy')
          
print(model.summary())
          
filename = '/Users/Administrator/Documents/Machine Learning/RNN/fra-eng/TamilEngTranslate.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)


# In[169]:


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[170]:


def predict_sequence(model, tokenizer, source):
    prediction = model.predict(source, verbose=0)[0]
    integers = [argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


# In[171]:


def evaluate_model(model, tokenizer, sources, raw_dataset):
    actual, predicted = list(), list()
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_sequence(model, tokenizer, source)
        raw_target, raw_src = raw_dataset[i]
        if i < 100:
            print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
        actual.append(raw_target.split())
        predicted.append(translation.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# In[172]:


print('train')
evaluate_model(model, engtokenizer, trainX, train)


# In[174]:


print('test')
evaluate_model(model, engtokenizer, testX, test)

