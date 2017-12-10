
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
import xgboost

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
        wordIds = []
        for phrase in corpus:
            wordIds.append([self.word2id[word] for word in phrase.split(" ")])
        return wordIds
        
    def classify(self, classifier_name):
        if classifier_name == "RandomForest":
            self.classifier = RandomForestClassifier(n_estimators=100)
            scores = cross_val_score(self.classifier, self.X, self.y, cv=5)
            print("Accuracy: %0.2f" %(scores.mean()))
			
        elif classifier_name == "SGD":
            self.classifier = SGDClassifier(loss="hinge", penalty="l2")
            scores = cross_val_score(self.classifier, self.X, self.y, cv=5)
            print("Accuracy: %0.2f" %(scores.mean()))
			
        elif classifier_name == "GradientBoosting":
            self.classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
            scores = cross_val_score(self.classifier, self.X, self.y, cv=5)
            print("Accuracy: %0.2f" %(scores.mean()))
            
        elif classifier_name == "SVM":
            self.classifier = svm.LinearSVC()
            scores = cross_val_score(self.classifier, self.X, self.y, cv=5)
            print("Accuracy: %0.2f" %(scores.mean()))
            
        elif classifier_name == "XGBoost":
            params = {'max_depth':3, 'eta':1.0, 'silent':1, 'colsample_bytree':1, 
             'num_class':5, 'min_child_weight':2, 'objective':'multi:softprob'}
            numRounds = 4
            dataMatrix = xgboost.DMatrix(self.X, label=self.y, missing=-999)
            xgboost.cv(params, dataMatrix, numRounds, nfold=5, metrics={'merror'},
            seed=0, callbacks=[xgboost.callback.print_evaluation(show_stdv=True)])
            
        elif classifier_name == "RNN":
            numLabels = 5
            trainLabels = np_utils.to_categorical(self.trainY, numLabels)
            testLabels = np_utils.to_categorical(self.testY, numLabels)
            
            self.classifier = Sequential()
            self.classifier.add(Embedding(len(self.vocabulary), 128))
            self.classifier.add(SpatialDropout1D(0.2))
            self.classifier.add(Bidirectional(LSTM(128)))
            self.classifier.add(Dense(numLabels))
            self.classifier.add(Activation('softmax'))
            
            self.classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            self.classifier.fit(self.trainX, trainLabels, validation_data=(self.testX, testLabels), epochs=5, batch_size=256, verbose=1)
            
                
    def run(self):
        # Preprocessing
        train_corpus = self.preprocess(self.trainData)
        test_corpus = self.preprocess(self.testData)
        
        # Generate dictionary
        wordFreq = FreqDist([word for phrase in train_corpus+test_corpus for word in phrase.split(" ")])
        self.vocabulary = list(wordFreq.keys())
        self.word2id = {word: i for i, word in enumerate(self.vocabulary)}
        
        # Extracting features
        self.X = self.extractFeatures(train_corpus)
        self.testData = self.extractFeatures(test_corpus)
        
        # Determine max sequence length
        lenStats = sorted([len(phrase) for phrase in self.X+self.testData])
        maxLength = lenStats[int(len(lenStats)*0.8)]
        
        # Pad sequences
        self.X = sequence.pad_sequences(np.array(self.X), maxlen=maxLength)
        self.testData = sequence.pad_sequences(np.array(self.testData), maxlen=maxLength)
        
        # Split validation set
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(self.X, self.y, test_size=0.1, random_state=0)
        
        # Classify
        classifier_name = "XGBoost"
        self.classify(classifier_name)
        
        
    def __init__(self, trainPath, testPath):
        self.trainData, self.y = self.readFile(trainPath)
        self.testData, _ = self.readFile(testPath)
        self.stopWords = set(stopwords.words("english"))
        self.run()

    

if __name__ == "__main__":
    pre = SentimentAnalysis("./train.tsv", "./test.tsv")
    