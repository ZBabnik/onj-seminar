from readXls import ReadXls
from tagger import Tagger
from tokeniser import tokeniser
from pymagnitude import Magnitude
import _pickle as pickle
from os import path, mkdir
from re import sub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


use_pickled_data = True
try:
    mkdir("pickle")
except:
    pass


class Lemmatize:

    def __init__(self, lemmatizer):
        self.lemmatizer = lemmatizer

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        if not path.exists("pickle/lemmas.pkl") or not use_pickled_data:
            # remove all non alphanumeric characters
            no_non_alphanumeric_chars = map(lambda t: sub(r'[^a-zA-Z0-9]+', ' ', str(t)), X)
            # if a letter is repeated 3 times its most likely to emphasize the text so its shortened to 1 repetition
            no_triple_chars = map(lambda t: sub(r'(\w)\1\1*', r'\1', str(t)), no_non_alphanumeric_chars)
            # get lemmas change all lemmas to lower case
            lemmas = map(lambda t: list(map(lambda u: u.lower(), self.lemmatizer.lemmatiser(t, tagging=True)[1])), no_triple_chars)
            # if lemmas are empty mark it with _
            lemmas = list(map(lambda t: ['_'] if (not t) else t, lemmas))
            if use_pickled_data:
                pickle.dump(lemmas, open("pickle/lemmas.pkl", "wb"))
        if use_pickled_data:
            lemmas = pickle.load(open("pickle/lemmas.pkl", "rb"))
        return lemmas


class ToStr:

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        return list(map(str, X))


class Word2vecWiki:

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        if not path.exists("pickle/wikiWV.pkl") or not use_pickled_data:
            wv_embeddings = Magnitude("embeddings/wiki.sl.magnitude")
            word_vec = list(map(lambda t: np.sum(wv_embeddings.query(t), axis=0), X))
            if use_pickled_data:
                pickle.dump(word_vec, open("pickle/wikiWV.pkl", "wb"))
        if use_pickled_data:
            word_vec = pickle.load(open("pickle/wikiWV.pkl", "rb"))
        return word_vec


class Word2vecElmo:
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        if not path.exists("pickle/elmoWV.pkl") or not use_pickled_data:
            wv_embeddings = Magnitude("embeddings/slovenian-elmo.weights.magnitude")
            word_vec = list(map(lambda t: np.sum(wv_embeddings.query(t), axis=0), X))
            if use_pickled_data:
                pickle.dump(word_vec, open("pickle/elmoWV.pkl", "wb"))
        if use_pickled_data:
            word_vec = pickle.load(open("pickle/elmoWV.pkl", "rb"))
        return word_vec


if __name__ == "__main__":
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = xls.get_column_with_name("Message")
    messages_gt = list(filter(lambda t: t, xls.get_column_with_name("Category")))  # ground truth

    X_train, X_test, y_train, y_test = train_test_split(messages, messages_gt, test_size=0.3, random_state=0)

    pipelane_lr1 = Pipeline([('str', ToStr()),
                             ('BOW', CountVectorizer()),
                             ('classify', LogisticRegression(random_state=0))])

    pipelane_lr2 = Pipeline([('scalar1', Lemmatize(Tagger())),
                             ('word2vecW', Word2vecWiki()),
                             ('classify', LogisticRegression(random_state=0))])

    pipelane_lr3 = Pipeline([('scalar2', Lemmatize(Tagger())),
                             ('word2vecE', Word2vecElmo()),
                             ('classify', LogisticRegression(random_state=0))])

    pipelanes = [pipelane_lr1, pipelane_lr2, pipelane_lr3]
    pipelanes_dict = ["BOW", "word2vecWiki", "word2vecElmo"]

    for pipelane in pipelanes:
        pipelane.fit(X_train, y_train)

    for i, model in enumerate(pipelanes):
        scr = 0
        for a,b in zip(y_test, model.predict(X_test)):
            if a == b:
                scr += 1
        print("{} Test Accuracy: {}".format(pipelanes_dict[i], scr/len(y_test)))
