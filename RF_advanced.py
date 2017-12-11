
from collections import defaultdict

from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score
from sklearn.multiclass import fit_ovo
from collections import namedtuple
from os.path import join, dirname, abspath
import nltk


DATA_PATH = abspath(join(dirname(__file__), "..", "data"))

Datapoint = namedtuple("Datapoint", "phraseid sentenceid phrase sentiment")

def target(phrases):
    return [datapoint.sentiment for datapoint in phrases]


class PhraseSentimentPredictor:

    def __init__():
        
        self.limit_train = limit_train
        pipeline = [ExtractText(lowercase)]
        if text_replacements:
            pipeline.append(ReplaceText(text_replacements))

        ext = [build_text_extraction(binary=binary, min_df=min_df, ngram=1, stopwords=stopwords)]
        #ext.append(build_lex_extraction(binary=binary, min_df=min_df, ngram=1))
        pipeline.append(ext)
        classifier_args = {"n_estimators": 200, "min_samples_leaf":10, "n_jobs":-1}
        classifier = RandomForestClassifier(**classifier_args)
        self.pipeline = make_pipeline(*pipeline)
        self.classifier = classifier

    def fit(self, phrases, y=None):

        y = target(phrases)
        Z = self.pipeline.fit_transform(phrases, y)
        if self.limit_train:
            self.classifier.fit(Z[:self.limit_train], y[:self.limit_train])
        else:
            self.classifier.fit(Z, y)
        return self

    def predict(self, phrases):

        Z = self.pipeline.transform(phrases)
        labels = self.classifier.predict(Z)
        return labels

    def score(self, phrases):

        pred = self.predict(phrases)
        return accuracy_score(target(phrases), pred)

    def error_matrix(self, phrases):
        predictions = self.predict(phrases)
        matrix = defaultdict(list)
        for phrase, predicted in zip(phrases, predictions):
            if phrase.sentiment != predicted:
                matrix[(phrase.sentiment, predicted)].append(phrase)
        return matrix


def build_text_extraction(binary, min_df, ngram, stopwords):
    return make_pipeline(CountVectorizer(binary=binary,
                                         tokenizer=lambda x: x.split(),
                                         min_df=min_df,
                                         ngram_range=(1, ngram),
                                         stop_words=stopwords),
                         ClassifierOvOAsFeatures())




class _Baseline:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return ["2" for _ in X]

    def score(self, X):
        gold = target(X)
        pred = self.predict(X)
        return accuracy_score(gold, pred)
class StatelessTransform:
    def fit(self, X, y=None):
        return self


class ExtractText(StatelessTransform):
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    def transform(self, X):
        it = (" ".join(nltk.word_tokenize(datapoint.phrase)) for datapoint in X)
        if self.lowercase:
            return [x.lower() for x in it]
        return list(it)


class ReplaceText(StatelessTransform):
    def __init__(self, replacements):
        self.rdict = dict(replacements)
        self.pat = re.compile("|".join(re.escape(origin) for origin, _ in replacements))

    def transform(self, X):
        if not self.rdict:
            return X
        return [self.pat.sub(self._repl_fun, x) for x in X]

    def _repl_fun(self, match):
        return self.rdict[match.group()]

class ClassifierOvOAsFeatures:

    def fit(self, X, y):
        self.classifiers = fit_ovo(SGDClassifier(), X, numpy.array(y), n_jobs=-1)[0]
        return self

    def transform(self, X, y=None):
        xs = [clf.decision_function(X).reshape(-1, 1) for clf in self.classifiers]
        return numpy.hstack(xs)

def _iter_data_file(filename):
    path = os.path.join(DATA_PATH, filename)
    it = csv.reader(open(path, "r"), delimiter="\t")
    row = next(it)  # Drop column names
    if " ".join(row[:3]) != "PhraseId SentenceId Phrase":
        raise ValueError("Input file has wrong column names: {}".format(path))
    for row in it:
        if len(row) == 3:
            row += (None,)
        yield Datapoint(*row)


def iter_corpus(__cached=[]):
    if not __cached:
        __cached.extend(_iter_data_file("train.tsv"))
    return __cached


def iter_test_corpus():
    return list(_iter_data_file("test.tsv"))


def make_train_test_split(seed, proportion=0.9):
    data = list(iter_corpus())
    ids = list(sorted(set(x.sentenceid for x in data)))
    if len(ids) < 2:
        raise ValueError("Corpus too small to split")
    N = int(len(ids) * proportion)
    if N == 0:
        N += 1
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_ids = set(ids[N:])
    train = []
    test = []
    for x in data:
        if x.sentenceid in test_ids:
            test.append(x)
        else:
            train.append(x)
    return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename")

    predictor = PhraseSentimentPredictor()
    predictor.fit(list(iter_corpus()))
    test = list(iter_test_corpus())
    prediction = predictor.predict(test)

    writer = csv.writer(sys.stdout)
    writer.writerow(("PhraseId", "Sentiment"))
    for datapoint, sentiment in zip(test, prediction):
       writer.writerow((datapoint.phraseid, sentiment))
