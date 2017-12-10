#!/usr/bin/python

import math
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, movie_reviews
from nltk.stem import SnowballStemmer
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, GRU, SpatialDropout1D, Bidirectional
from keras.utils import np_utils
from keras.preprocessing import sequence

import numpy as np
import csv
from sklearn.datasets import load_digits

from keras import backend as K

from keras.optimizers import SGD
from keras.layers import Dense

from keras.layers import Input
from keras.models import Model

from neural_tensor_layer import NeuralTensorLayer

np.random.seed(0)
class SentimentAnalysis:
  def __init__(self, trainPath, testPath):
    self.trainData, self.y = self.readFile(trainPath)
    self.testData, _ = self.readFile(testPath)
    self.stopWords = set(stopwords.words("english"))
    self.run()

  def readFile(self, filePath):
    data = []
    y = []
    with open(filePath, 'r') as file:
      csvreader = csv.reader(file, delimiter='\t')
      next(csvreader)
      for row in csvreader:
        data.append(row[2])
        if len(row) > 3:
          y.append(row[3])
    return data, y


  def preprocess(self, data):
    preprocessedCorpus = []

    for phrase in data:
      # All to lower case
      phrase = phrase.lower()

      # Split to tokens
      tokenizer = RegexpTokenizer(r'\w+')
      tokens = tokenizer.tokenize(phrase)

      # Stopword filtering
      nonstopTokens = [token for token in tokens if not token in self.stopWords]

      # Stemming
      stemmer = SnowballStemmer("english")
      for index, item in enumerate(nonstopTokens):
        stemmedWord = stemmer.stem(item)
        nonstopTokens[index] = stemmedWord

      # Add to corpus
      preprocessedCorpus.append(" ".join(nonstopTokens))

    return preprocessedCorpus

  # def get_data():
  #   #digits = load_digits()
  #   trainData = pd.read_csv('train.tsv', sep='\t', header=0)
  #   testData = pd.read_csv('test.tsv', sep='\t', header=0)
  #   trainPhrases = trainData['Phrase'].values
  #   testPhrases = testData['Phrase'].values
  #   X_train = trainData['Phrase'].values
  #   y_train = trainData['Sentiment'].values
  #   X_test = testData['Phrase'].values
  #   return X_train, y_train, X_test

  def extractFeatures(self, corpus):
    wordIds = []
    for phrase in corpus:
      wordIds.append([self.word2id[word] for word in phrase.split(" ")])
    return wordIds

  def run(self):
    # Preprocessing
    train_corpus = self.preprocess(self.trainData)
    test_corpus = self.preprocess(self.testData)

    # Generate dictionary
    wordFreq = FreqDist([word for phrase in train_corpus + test_corpus for word in phrase.split(" ")])
    self.vocabulary = list(wordFreq.keys())
    self.word2id = {word: i for i, word in enumerate(self.vocabulary)}

    # Extracting features
    self.X = self.extractFeatures(train_corpus)
    self.testData = self.extractFeatures(test_corpus)

    # Determine max sequence length
    # lenStats = sorted([len(phrase) for phrase in self.X + self.testData])
    # maxLength = lenStats[int(len(lenStats) * 0.8)]
    maxLength = 64
    # Pad sequences
    self.X = sequence.pad_sequences(np.array(self.X), maxlen=maxLength)
    self.testData = sequence.pad_sequences(np.array(self.testData), maxlen=maxLength)

    # Split validation set
    self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.y, test_size=0.1, random_state=0)

    input1 = Input(shape=(64,), dtype='float32')
    input2 = Input(shape=(64,), dtype='float32')
    btp = NeuralTensorLayer(output_dim=32, input_dim=64)([input1, input2])

    p = Dense(output_dim=1)(btp)
    model = Model(input=[input1, input2], output=[p])

    sgd = SGD(lr=0.0000000001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    npxtrain = np.asarray(self.trainX, dtype=np.int32)
    npytrain = np.asarray(self.trainY, dtype=np.int32)
    npxtest = np.asarray(self.testX, dtype=np.int32)
    npytest = np.asarray(self.testY, dtype=np.int32)
    X_train = npxtrain.astype(np.float32)
    Y_train = npytrain.astype(np.float32)
    X_test = npxtest.astype(np.float32)
    Y_test = npytest.astype(np.float32)

    model.fit([X_train, X_train], Y_train, nb_epoch=50, batch_size=128)
    score = model.evaluate([X_test, X_test], Y_test, batch_size=128)
    print score

    pred = K.get_value(model.layers[2].W)
    # count = 0
    # totalcount = 0
    # for i in range(0, len(pred)):
    #   val = pred[i]
    #   if Y_test[i] == min(abs(val-(-1.0)), abs(val-(-2.0)), abs(val-(0)), abs(val-1), abs(val-2) ):
    #     count+=1
    #   totalcount+=1
    #
    # print count
    # print totalcount
    print pred




if __name__ == "__main__":
  pre = SentimentAnalysis("./train.tsv", "./test.tsv")