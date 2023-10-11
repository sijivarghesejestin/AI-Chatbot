import random                                                                    #for choosing random responses
import json
import pickle                                                                    #for serialisation
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import tensorflow.keras

                                                                                 #for natural language processing

import tensorflow.keras.optimizers
#import tensorflow as tf
#import tensorflow.python.keras as keras
#import keras.models

#from keras.models import load_model
from nltk.stem import WordNetLemmatizer                                          # recognise word with same stem(eg:work,worked,working etc)
from tensorflow.keras.models import Sequential                                   #keras sequential which deals with ordering or sequencing of layers within a model.
from tensorflow.keras.layers import Dense, Activation, Dropout                   #A dense layer is a classic fully connected neural network layer : each input node is connected to each output node. A dropout layer is similar except that when the layer is used, the activations are set to zero for some random nodes. This is a way to prevent overfitting.
from tensorflow.keras.optimizers import SGD                                      #Stochastic Gradient Descent (SGD) is a variant of the Gradient Descent algorithm that is used for optimizing machine learning models.


lemmatizer = WordNetLemmatizer()


intents = json.loads(open('/content/drive/MyDrive/chatbot /intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents']:
   for pattern in intent['patterns']:
       word_list = nltk.word_tokenize(pattern)                                      #splits a sentence into words
      # print(word_list)
       words.extend(word_list)                                                      #append new words to existing list
       documents.append((word_list, intent['tag']))                                 #determines which word belong to which tag class
       #print(documents)
       if intent['tag'] not in classes:
           classes.append(intent['tag'])                                            # if words are not there in tag append to tag

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))                                                          #to remove duplicate word use 'set'

classes = sorted(set(classes))

# process to converts any kind of python objects (list, dict, etc.) into byte streams (0s and 1s) is called pickling or serialization or flattening or marshalling
#dump function takes 3 arguments. The first argument is the object that you want to store. The second argument is the file object you get by opening the desired file in write-binary (wb) mode. And the third argument is the key-value argument

pickle.dump(words, open('words.pkl', 'wb'))                                         #dump() writes the data to a file
pickle.dump(classes, open('classes.pkl', 'wb'))

#neural network dont needs word it need numbers so we have to convert these chars/words to numbers so that it can work on


#below code appends all document data to training to work on neural network
training = []
output_empty = [0] * len(classes)
#print(output_empty)

for document in documents:
    bag = []                                                                        # for each combination we will create a empty bag of words
    word_patterns = document[0]

    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    #print(word_patterns)
    for word in words:
       bag.append(1) if word in word_patterns else bag.append(0)


    output_row = list(output_empty)                                                #copy it to a list
    output_row[classes.index(document[1])] = 1                                     #set index to the output_row to 1
    training.append([bag, output_row])
    print(training)




    random.shuffle(training)                                                      #shuffle the training data
    #replace training instead of data
    training = np.array(training,dtype="object")


    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # code to build neural network

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))


    sgd = SGD(learning_rate= 0.01,weight_decay= 1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save("chatbotmodel.h5", hist)
    print("Done")
