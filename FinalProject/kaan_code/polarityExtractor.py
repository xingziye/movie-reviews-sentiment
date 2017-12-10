
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stopWords = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

a = open("sentiwordnet.txt", "r")
dict = {}
for row in a:
    if not row.strip().startswith("#"):
        l = row.strip().split()
        pos = l[2]
        neg = l[3]
        for i in l[4:]:
            if not "#" in i:
                break
            else:
                word = i.split("#")[0].lower()
                if not word in stopWords:
                    word = stemmer.stem(word)
                    dict[word] = (float(pos), float(neg))
                
pickle.dump(dict, open("polarityDict", "wb"))