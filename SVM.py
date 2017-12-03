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
    #letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    words = raw_review.lower().split()
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
pipeline_svm = make_pipeline(vectorizer,
                             SVC(probability=True, kernel="linear", class_weight="balanced"))

from sklearn.model_selection import GridSearchCV
grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]},
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,
                    n_jobs=-1)

grid_svm.fit(train_data_features, train['Sentiment'])
