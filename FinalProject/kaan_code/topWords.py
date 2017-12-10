
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

def readFile(filePath):
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
    
def preprocess(data):
    preprocessedCorpus = []
    
    for phrase in data:
        # All to lower case
        phrase = phrase.lower()
        
        # Split to tokens
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(phrase)
                    
        # Stopword filtering
        nonstopTokens = [token for token in tokens if not token in stopWords]
                    
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


# Paths and file reading    
trainPath = "./train.tsv"
testPath =  "./test.tsv" 
trainData, y = readFile(trainPath)
testData, _ = readFile(testPath)
stopWords = set(stopwords.words("english"))

# Preprocessing
train_corpus = preprocess(trainData)
test_corpus = preprocess(testData)

# Generate dictionary
wordFreq = FreqDist([word for phrase in train_corpus+test_corpus for word in phrase.split(" ")])
vocabulary = list(wordFreq.keys())[:3000]
word2id = {word: i for i, word in enumerate(vocabulary)}

# Extracting features
vectorizer = CountVectorizer(vocabulary=vocabulary)
X = vectorizer.fit_transform(train_corpus).todense()

# Classify
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X, y)
numTopFeatures = 1000
featureImportance = classifier.feature_importances_
featureImportanceList = [(val, idx) for idx, val in enumerate(featureImportance)]
featureImportanceList.sort(reverse=True)
featureImportanceList = featureImportanceList[:min(numTopFeatures, len(featureImportanceList))]
print(featureImportanceList)
