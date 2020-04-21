from readXls import ReadXls
from tagger import Tagger
from pymagnitude import Magnitude
import _pickle as pickle
from os import path, mkdir
from re import sub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


use_pickled_data = True
try:
    mkdir("pickle")
except:
    pass

class Lemmatize:
    def fit(self, X, y=None):
        if not path.exists("pickle/lemmas.pkl") or not use_pickled_data:
            tagger = Tagger()
            # remove all non alphanumeric characters
            no_non_alphanumeric_chars = map(lambda t: sub(r'[^a-zA-Z0-9]+', ' ', str(t)), X)
            # if a letter is repeated 3 times its most likely to emphasize the text so its shortened to 1 repetition
            no_triple_chars = map(lambda t: sub(r'(\w)\1\1*', r'\1', str(t)), no_non_alphanumeric_chars)
            # get lemmas change all lemmas to lower case
            lemmas = map(lambda t: list(map(lambda u: u.lower(), tagger.lemmatiser(t, tagging=True)[1])), no_triple_chars)
            # if lemmas are empty mark it with _
            lemmas = list(map(lambda t: ['_'] if (not t) else t, lemmas))
            if use_pickled_data:
                pickle.dump(lemmas, open("pickle/lemmas.pkl", "wb"))
        if use_pickled_data:
            lemmas = pickle.load(open("pickle/lemmas.pkl", "rb"))
        return lemmas

class Word2vecWiki:
    def fit(self, X, y=None):
        if not path.exists("pickle/wikiWV.pkl") or not use_pickled_data:
            wv_embeddings = Magnitude("embeddings/wiki.sl.magnitude")
            word_vec = list(map(lambda t: wv_embeddings.query(t), X))
            if use_pickled_data:
                pickle.dump(word_vec, open("pickle/wikiWV.pkl", "wb"))
        if use_pickled_data:
            word_vec = pickle.load(open("pickle/wikiWV.pkl", "rb"))
        return word_vec

class Word2vecElmo:
    def fit(self, X, y=None):
        if not path.exists("pickle/elmoWV.pkl") or not use_pickled_data:
            wv_embeddings = Magnitude("embeddings/slovenian-elmo.weights.magnitude")
            word_vec = list(map(lambda t: wv_embeddings.query(t), X))
            if use_pickled_data:
                pickle.dump(word_vec, open("pickle/elmoWV.pkl", "wb"))
        if use_pickled_data:
            word_vec = pickle.load(open("pickle/elmoWV.pkl", "rb"))
        return word_vec

if __name__ == "__main__":
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = xls.get_column_with_name("Message")
    messages_gt = xls.get_column_with_name("Type")  # ground truth

    X_train, X_test, y_train, y_test = train_test_split(messages, messages_gt, test_size=0.2, random_state=0)

    pipelane_lr1 = Pipeline([('scalar1', Lemmatize()),
                             ('word2vecW', Word2vecWiki()),
                             ('classify', LogisticRegression(random_state=0))])

    pipelane_lr2 = Pipeline([('scalar2', Lemmatize()),
                             ('word2vecE', Word2vecElmo()),
                             ('classify', LogisticRegression(random_state=0))])

    pipelanes = [pipelane_lr1, pipelane_lr2]
    pipelanes_dict = ["word2vecWiki", "word2vecElmo"]

    for pipelane in pipelanes:
        pipelane.fit(X_train, y_train)

    for i, model in enumerate(pipelanes):
        print("{} Test Accuracy: {}".format(pipelanes_dict[i], model.score(X_test, y_test)))
