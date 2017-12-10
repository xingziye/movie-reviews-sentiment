
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
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, GRU, SpatialDropout1D
from keras.utils import np_utils

import numpy as np
import csv
import pickle

np.random.seed(0)

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
        vectorizer = CountVectorizer(vocabulary=self.vocabulary)
        self.X = vectorizer.fit_transform(corpus).todense()
        # transformer = TfidfTransformer()
        # self.X = transformer.fit_transform(data_counts)
        
    def classify(self, classifier_name):
        if classifier_name == "RandomForest":
            self.classifier = RandomForestClassifier(n_estimators=100)
            scores = cross_val_score(self.classifier, self.X, self.y, cv=10)
            print("Accuracy: %0.2f" %(scores.mean()))
			
        elif classifier_name == "SGD":
            self.classifier = SGDClassifier(loss="hinge", penalty="l2")
            scores = cross_val_score(self.classifier, self.X, self.y, cv=10)
            print("Accuracy: %0.2f" %(scores.mean()))
			
        elif classifier_name == "GradientBoosting":
            self.classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
            scores = cross_val_score(self.classifier, self.X, self.y, cv=10)
            print("Accuracy: %0.2f" %(scores.mean()))
            
        elif classifier_name == "SVM":
            self.classifier = svm.LinearSVC()
            scores = cross_val_score(self.classifier, self.X, self.y, cv=10)
            print("Accuracy: %0.2f" %(scores.mean()))
            
        elif classifier_name == "DNN":
            numLabels = len(np.unique(self.y))
            labels = np_utils.to_categorical(self.y, numLabels)
            # trainLabels = np_utils.to_categorical(self.trainY, numLabels)
            # testLabels = np_utils.to_categorical(self.testY, numLabels)
            
            self.classifier = Sequential()
            self.classifier.add(Dense(128, activation='relu', input_dim=len(self.vocabulary)))
            # self.classifier.add(Dense(128, input_dim=512))
            self.classifier.add(Dense(numLabels, activation='softmax'))
            
            self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.classifier.fit(self.X, labels, epochs=10, batch_size=256, verbose=2)
            # self.extractFeatures(self.test_corpus)
            # pred = self.classifier.predict(self.X)
            # np.savetxt('result.csv', pred, delimiter=',')
           
           
        # elif classifier_name == "RNN":
            # numLabels = len(np.unique(self.y))
            # labels = np_utils.to_categorical(self.y, numLabels)
                
    def run(self):
        # Preprocessing
        train_corpus = self.preprocess(self.trainData)
        self.test_corpus = self.preprocess(self.testData)
        
        # Generate dictionary
        wordFreq = FreqDist([word for phrase in train_corpus+self.test_corpus for word in phrase.split(" ")])
        self.vocabulary = list(wordFreq.keys())[:2000]
        
        # Extracting features
        self.extractFeatures(train_corpus)
        
        # Polarize matrix
        self.modifyMatrix()
        # pickle.dump(self.X, open("dataMatrix", "wb"))
        # self.X = pickle.load(open("dataMatrix", "rb"))
        
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.y, test_size=0.1, random_state=0)
        del self.X
        
        # Classify
        classifier_name = "DNN"
        self.classify(classifier_name)
        
        
    def prepareDict(self):
        for key in list(self.polarityDict.keys()):
            self.polarityDict[key] = np.exp(np.abs(self.polarityDict[key][0] - self.polarityDict[key][1]))
    
    def modifyMatrix(self):
        for y in range(0, self.X.shape[1]):
            if (y+1)%100 == 0:
                print(y)
            if self.vocabulary[y] in self.polarityDict:
                pol = self.polarityDict[self.vocabulary[y]]
                for x in range(0, self.X.shape[0]):
                    if self.X[x, y] != 0:
                        self.X[x, y] = pol * self.X[x, y]
        
    def __init__(self, trainPath, testPath):
        self.polarityDict = pickle.load(open("polarityDict", "rb"))
        self.prepareDict()
        self.trainData, self.y = self.readFile(trainPath)
        print(len(self.trainData))
        self.testData, _ = self.readFile(testPath)
        print(len(self.testData))
        self.stopWords = set(stopwords.words("english"))
        self.run()

    

if __name__ == "__main__":
    pre = SentimentAnalysis("./train.tsv", "./test.tsv")
    