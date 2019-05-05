import pandas as pd
import numpy as np
import csv
import re
import h5py
import nltk
from numpy import *
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics,svm, tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt



happy_emoji = [':-)', ':)', ':-]', ':]',
               ':-3', ':3', ':->', ':>',
               '8-)', '8)', ':-}', ':}',
               ':o)', ':c)', ':^)', '=]',
               '=)', ':-D', ':D', '8-D',
               '8D', 'x-D', 'xD', 'X-D',
               'XD', '=D', '=3', 'B^D', ':-))']

sad_emoji = [':-(', ':(', ':-c', ':c',
             ':-<', ':<', ':-[', ':[',
             ':-||', '>:[', ':{', ':@', '>:(']

negative_words = ["doesn't", "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't", "can't", "don't"]
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()


def preprocess_data(fileName, windowSize = 6, Testing = False):
    with open(fileName, newline='') as inputFile:
        fparser = csv.DictReader(inputFile)
        tokens = []
        asps_tokens = []
        reviews = []
        asp_term = []
        labels = []
        aspects = []
        ids = []
        for row in fparser:
            properAspect = []
            for asp_temp in tokenizer.tokenize(row[' aspect_term']):
                properAspect.append(lemmatizer.lemmatize(asp_temp))

            aspects.append(properAspect)

            if not Testing:
                labels.append(row[' class'])
            else:
                ids.append(row['example_id'])

            tempStr = row[' text'].replace('[comma]', ',').lower()
            tempStr = tempStr.replace('_', ' ').lower()
            tempAspect = row[' aspect_term'].replace('[comma]', ',').lower()
            tempAspect = tempAspect.replace('_', ' ').lower()

            for emoji in happy_emoji:
                if emoji in tempStr:
                    tempStr = tempStr.replace(emoji, 'happy')

            for emoji in sad_emoji:
                if emoji in tempStr:
                    tempStr = tempStr.replace(emoji, 'sad')

            for word in negative_words:
                if word in tempStr:
                    tempStr = tempStr.replace(word, 'not')

            for token in tokenizer.tokenize(tempStr):
                if token not in stop_words or token == 'not':
                    tokens.append(lemmatizer.lemmatize(token))

            for token in tokenizer.tokenize(tempAspect):
                if token not in stop_words or token == 'not':
                    asps_tokens.append(lemmatizer.lemmatize(token))

            reviews.append(tokens)
            asp_term.append(asps_tokens)
            tokens = []
            asps_tokens = []

        processedList = []
        for tk in reviews:
            processedList.append(' '.join(tk))

        newAsps = []
        for tk in asp_term:
            newAsps.append(' '.join(tk))

        windowS = []

        for idx, asp_term in enumerate(newAsps):
            sentence = processedList[idx]
            lh, _, rh = sentence.partition(asp_term)
            window = ' '.join(lh.split()[-windowSize:] + [asp_term] + rh.split()[:windowSize])
            windowS.append(window)

        if not Testing:
            return windowS, labels
        else:
            return windowS, ids



print("Preprocessing the data")

trainData, labels = preprocess_data('Data_2_train.csv', windowSize=6, Testing=False)
testData, ids = preprocess_data('Data-2_test.csv', windowSize=6, Testing=True)

print("Vectorizing the data")
tfi = TfidfVectorizer(ngram_range=(1, 1))
tfidf_train = tfi.fit_transform(trainData)
trainDataArray = tfidf_train.toarray()

tfidf_test = tfi.transform(testData)

print("Shuffling the data")
trainDataArray, labels = shuffle(trainDataArray, labels, random_state = 0)

print("Duplicating the data")
# Duplicating negative and neutral examples so we have enough sample for both.
for i in range(0,len(labels)):
    if labels[i] == "-1" or labels[i] == "0":
        append(trainDataArray,(trainDataArray[i]))
        append(labels,(labels[i]))
        append(trainDataArray,(trainDataArray[i]))
        append(labels,(labels[i]))
        append(trainDataArray,(trainDataArray[i]))
        append(labels,(labels[i]))

print("Encoding the data")
# Encoding the data
cla = []
for i in range(len(labels)):
    if labels[i] == '-1': cla.append(0)
    if labels[i] == '0': cla.append(1)
    if labels[i] == '1': cla.append(2)

x = pd.DataFrame(trainDataArray)
y = pd.DataFrame(cla)

# spliting training data into training + validation.
xTrain, xValidate , yTrain, yValidate = train_test_split(x,y,test_size=0.10)
xTest = pd.DataFrame(tfidf_test.toarray())

# one hot encoding the data. To be used in case there more than binary classes.
yTrainHot = to_categorical(yTrain, num_classes=3)
yValidateHot = to_categorical(yValidate, num_classes=3)

print("NN Starting")

# Defining the model
model1 = Sequential([
    Dense(600, input_dim=xTrain.shape[1]),
    Activation('relu'),
    BatchNormalization(),
    Dense(300),
    Activation('relu'),
    BatchNormalization(),
    Dense(150),
    Activation('relu'),
    BatchNormalization(),
    Dense(3),
    Activation('softmax')
])


# Model checkpoints and stoppers. Save the best weights and stops the model when no increase in performance to save time.
nadam = optimizers.Nadam()
model1.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=['accuracy'])
esc = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=0, mode='auto')
cp = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)


# fitting the model.
m1 = model1.fit(xTrain, yTrainHot, epochs=500, callbacks=[esc, cp], validation_data=(xValidate, yValidateHot))


print(" Loading Weights")
# loading the best weights
model1.load_weights("weights.hdf5")

# predicting classes for testing cases.
yP1 = model1.predict(xTest)
yP2 = np.argmax(yP1,axis=1)

# decoding the data
yPredicted = []
for i in range(len(yP2)):
    if yP2[i] == 0: yPredicted.append('-1')
    if yP2[i] == 1: yPredicted.append('0')
    if yP2[i] == 2: yPredicted.append('1')


print('Saving results')

outfile = open(r'result2.txt','w+')
for i in range(len(ids)):
    print(ids[i], ';;', yPredicted[i], sep='', file=outfile)
outfile.close()
