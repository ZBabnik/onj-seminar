from readXls import ReadXls
from tagger import Tagger
from tokeniser import tokeniser
from pymagnitude import Magnitude
from re import sub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class Lemmatize:

    def __init__(self, lemmatizer):
        self.lemmatizer = lemmatizer

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        # remove all non alphanumeric characters
        no_non_alphanumeric_chars = map(lambda t: sub(r'[^a-zA-Z0-9]+', ' ', str(t)), X)
        # if a letter is repeated 3 times its most likely to emphasize the text so its shortened to 1 repetition
        no_triple_chars = map(lambda t: sub(r'(\w)\1\1*', r'\1', str(t)), no_non_alphanumeric_chars)
        # get lemmas change all lemmas to lower case
        lemmas = map(lambda t: list(map(lambda u: u.lower(), tokeniser(t))), no_triple_chars)
        # if lemmas are empty mark it with _
        lemmas = list(map(lambda t: ['_'] if (not t) else t, lemmas))
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
        wv_embeddings = Magnitude("embeddings/wiki.sl.magnitude")
        word_vec = list(map(lambda t: np.sum(wv_embeddings.query(t), axis=0), X))
        return word_vec


class Word2vecElmo:
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        wv_embeddings = Magnitude("embeddings/slovenian-elmo.weights.magnitude")
        word_vec = list(map(lambda t: np.sum(wv_embeddings.query(t), axis=0), X))
        return word_vec


if __name__ == "__main__":
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = xls.get_column_with_name("Message")
    messages_gt = list(filter(lambda t: t, xls.get_column_with_name("Category")))  # ground truth

    X_train, X_test, y_train, y_test = train_test_split(messages, messages_gt, test_size=0.3, random_state=0)

    pipelane_lr1 = Pipeline([('str', ToStr()),
                             ('BOW', CountVectorizer(ngram_range=(1,2))),
                             ('classify', LogisticRegression(random_state=0))])

    pipelane_lr2 = Pipeline([('scalar1', Lemmatize(Tagger())),
                             ('word2vecW', Word2vecWiki()),
                             ('classify', LogisticRegression(random_state=0))])

    pipelane_lr3 = Pipeline([('scalar2', Lemmatize(Tagger())),
                             ('word2vecE', Word2vecElmo()),
                             ('classify', LogisticRegression(random_state=0))])

    pipelanes = [pipelane_lr1, pipelane_lr2, pipelane_lr3]
    pipelanes_dict = ["BOW", "word2vecWiki", "word2vecElmo"]

    for i, pipelane in enumerate(pipelanes):
        pipelane.fit(X_train, y_train)
        print("{} Test Accuracy: {}".format(pipelanes_dict[i], pipelane.score(X_test, y_test)))
