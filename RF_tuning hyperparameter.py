from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, movie_reviews
from nltk.stem import SnowballStemmer
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, GRU, SpatialDropout1D, Bidirectional
from keras.utils import np_utils
from keras.preprocessing import sequence

import numpy as np
import matplotlib.pyplot as plt
import csv
import operator

np.random.seed(1234)


class SentimentAnalysis:
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

            # Remove numbers
            finalTokens = [token for token in nonstopTokens if not token.isnumeric()]

            # Add to corpus
            preprocessedCorpus.append(" ".join(nonstopTokens))

        return preprocessedCorpus

    def extractFeatures(self, corpus):
        wordIds = []
        CountVectorizer(binary=binary,
                        tokenizer=lambda x: x.split(),
                        min_df=min_df,
                        ngram_range=(1, 1),
                        stop_words=stopwords),
        ClassifierOvOAsFeatures()
        for phrase in corpus:
            wordIds.append([self.word2id[word] for word in phrase.split(" ")])
        return wordIds

    def classify(self):
        leafNodeSizeRange = range(1,100)
        scoreCrossVal = list()
        for minLeafNodeSize in leafNodeSizeRange:
            self.classifier = RandomForestClassifier(n_estimators=200, criterion='gini',
                                                     min_samples_leaf=minLeafNodeSize, n_jobs=-1)
            scores = cross_val_score(self.classifier, self.X, self.y, cv=10)
            scoreCrossVal.append(scores.mean())

        print(scores.mean())
        index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
        print("Max cross validation score: " + str(val))
        optimLeafNodeSize = leafNodeSizeRange[index]
        print("Optimal min leaf node size: " + str(optimLeafNodeSize))

        plt.figure()
        plt.plot(leafNodeSizeRange, scoreCrossVal)
        plt.xlabel('Minimum samples in leaf node')
        plt.ylabel('Cross validation score')
        plt.title('Random Forest')
        plt.show()
        
		
		maxDepthRange = range(30, 100, 5)
        scoreCrossVal = list()
        for maxTreeDepth in maxDepthRange:
            self.classifier = RandomForestClassifier(n_estimators=200, criterion='gini',
                                         max_depth=maxTreeDepth,n_jobs=-1)
        
            scores = cross_val_score(self.classifier, self.X, self.y, cv=10)
            scoreCrossVal.append(scores.mean())

        index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
        print("Max cross validation score: " + str(val))
        optimTreeDepth = maxDepthRange[index]
        print("Optimal max tree depth: " + str(optimTreeDepth))
        
        plt.figure()
        plt.plot(maxDepthRange, scoreCrossVal)
        plt.xlabel('Maximum tree depth')
        plt.ylabel('Cross validation score')
        plt.title('Random Forest')
        plt.show()

        # Try an extremely randomized forest.
        leafNodeSizeRange = range(1, 100)
        scoreCrossVal = list()
        for minLeafNodeSize in leafNodeSizeRange:
            print("Running model " + str(minLeafNodeSize) + "...")
            self.classifier = ExtraTreesClassifier(n_estimators=200, criterion='gini',
                                       min_samples_leaf=minLeafNodeSize)
            scores = cross_val_score(self.classifier, self.X, self.y, cv=10)
            scoreCrossVal.append(scores.mean())

        index, val = max(enumerate(scoreCrossVal), key=operator.itemgetter(1))
        print("Max cross validation score: " + str(val))
        optimLeafNodeSize = leafNodeSizeRange[index]
        print("Optimal min leaf node size: " + str(optimLeafNodeSize))

        plt.figure()
        plt.plot(leafNodeSizeRange, scoreCrossVal)
        plt.xlabel('Minimum samples in leaf node')
        plt.ylabel('Cross validation score')
        plt.title('Extremely Randomized Forest')
        plt.show()
		


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
        lenStats = sorted([len(phrase) for phrase in self.X + self.testData])
        maxLength = lenStats[int(len(lenStats) * 0.8)]

        # Pad sequences
        self.X = sequence.pad_sequences(np.array(self.X), maxlen=maxLength)
        self.testData = sequence.pad_sequences(np.array(self.testData), maxlen=maxLength)

        # Split validation set
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.y, test_size=0.1,
                                                                            random_state=0)

        # Classify

        self.classify()

    def __init__(self, trainPath, testPath):
        self.trainData, self.y = self.readFile(trainPath)
        self.testData, _ = self.readFile(testPath)
        self.stopWords = set(stopwords.words("english"))
        self.run()


if __name__ == "__main__":
    pre = SentimentAnalysis("./train.tsv", "./test.tsv")