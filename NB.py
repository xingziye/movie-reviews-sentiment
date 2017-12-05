import pandas as pd
import nltk
import re

train = pd.read_csv("train.tsv", header=0, delimiter="\t")
print train.shape
print train.columns.values
print train["Phrase"][247]

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

stops = set(stopwords.words("english"))
#stops = set(['the'])
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def review_to_words(raw_review):
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    words = letters_only.lower().split()
    meaningful_words = [stemmer.stem(w) for w in words if w not in stops]
    return( " ".join( meaningful_words ))

print review_to_words(train["Phrase"][247])

clean_train_reviews = []
num_reviews = train["Phrase"].size
for i in xrange(num_reviews):
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )
    clean_train_reviews.append(review_to_words(train['Phrase'][i]))

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(
    analyzer="word",
    tokenizer=None,
    preprocessor=None,
    stop_words=None)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

print train_data_features.shape

vocab = vectorizer.get_feature_names()
print vocab

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()

from sklearn.metrics import accuracy_score
for train_index, test_index in kf.split(train_data_features, train['Sentiment']):
    X_train, X_test = train_data_features[train_index], train_data_features[test_index]
    y_train, y_test = train['Sentiment'][train_index], train['Sentiment'][test_index]
    clf = clf.fit(X_train, y_train)
    print accuracy_score(y_test, clf.predict(X_test))
