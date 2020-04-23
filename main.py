from readXls import ReadXls
from tagger import Tagger
from tokeniser import tokeniser
from pymagnitude import Magnitude
from re import sub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfTransformer

models = {"embeddings/wiki.sl.magnitude": Magnitude("embeddings/wiki.sl.magnitude"),
          "embeddings/slovenian-elmo.weights.magnitude": Magnitude("embeddings/slovenian-elmo.weights.magnitude")}
countt = 0

class Lemmatize:

    def __init__(self, lemmatizer):
        self.lemmatizer = lemmatizer

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        # remove all non alphanumeric characters
        #no_non_alphanumeric_chars = map(lambda t: sub(r'[^a-zA-Z0-9]+', ' ', str(t)), X)
        # if a letter is repeated 3 times its most likely to emphasize the text so its shortened to 1 repetition
        #no_triple_chars = map(lambda t: sub(r'(\w)\1\1*', r'\1', str(t)), X)
        # get lemmas change all lemmas to lower case
        lemmas = map(lambda t: list(map(lambda u: u.lower(), tokeniser(str(t)))), X)
        # if lemmas are empty mark it with _
        lemmas = list(map(lambda t: ['_'] if (not t) else t, lemmas))
        return lemmas


class ToStr:
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        global countt
        countt += 1
        print(countt)
        return list(map(str, X))


class ToArray:

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        return X.toarray()


@lru_cache(maxsize=None)
def query(model_path, sentence):
    return models[model_path].query(list(sentence))


class WordEmbeddings:
    def __init__(self, model_path):
        self.model_path = model_path

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        global countt
        countt += 1
        print(countt)
        word_vec = list(map(lambda t: np.sum(query(self.model_path, tuple(t)), axis=0), X))
        return word_vec


class GetRelevance:
    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y=None, **kwargs):
        self.pipe.fit(X, kwargs["rel"])
        return self

    def transform(self, X, y=None, **kwargs):
        tmp = np.array(list(map(lambda t: 1 if (t == "Yes") else 0, self.pipe.predict(X))), dtype=float)
        return np.append(X, tmp.reshape(-1, 1), axis=1)

class GetType:
    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y=None, **kwargs):
        self.pipe.fit(X, kwargs["typ"])
        return self

    def transform(self, X, y=None, **kwargs):
        tmp = np.array(list(map(lambda t: 0 if (t == "A") else 2 if (t == "Q") else 1, self.pipe.predict(X))), dtype=float)
        return np.append(X, tmp.reshape(-1, 1), axis=1)


if __name__ == "__main__":
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = np.array(xls.get_column_with_name("Message"))
    relevance = np.array(list(filter(lambda t: t, xls.get_column_with_name("Book relevance"))))  # ground truth
    type = np.array(list(filter(lambda t: t, xls.get_column_with_name("Type"))))  # ground truth
    messages_gt = np.array(list(filter(lambda t: t, xls.get_column_with_name("Category"))))

    X_train, X_test, y_train, y_test, rel_train, rel_test, type_train, type_test\
        = train_test_split(messages, messages_gt, relevance, type, test_size=0.3, random_state=0)

    params = {
        "classify__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        "classify__class_weight": ['balanced', None],
        "classify__penalty": ['l1', 'elasticnet', 'l2', 'none'],
        "classify__l1_ratio": [0.5],
        "classify__multi_class": ["ovr", "multinomial"],
        "classify__max_iter": [500]
    }

    pipeline_lr1 = Pipeline([('str', ToStr()),
                             ('BOW', CountVectorizer(ngram_range=(1,2))),
                             #('tfidf', TfidfTransformer()),
                             ('toArray', ToArray()),
                             ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                             ('type', GetType(pipe=LogisticRegression(random_state=0))),
                             ('classify', LogisticRegression(random_state=0))])

    pipeline_lr2 = Pipeline([('scalar1', Lemmatize(Tagger())),
                             ('word2vecW', WordEmbeddings("embeddings/wiki.sl.magnitude")),
                             ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                             ('type', GetType(pipe=LogisticRegression(random_state=0))),
                             ('classify', LogisticRegression(random_state=0))])

    pipeline_lr3 = Pipeline([('scalar2', Lemmatize(Tagger())),
                             ('word2vecE', WordEmbeddings("embeddings/slovenian-elmo.weights.magnitude")),
                             ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                             ('type', GetType(pipe=LogisticRegression(random_state=0))),
                             ('classify', LogisticRegression(random_state=0))])

    pipelines = [pipeline_lr1, pipeline_lr2, pipeline_lr3]
    pipelines_dict = ["BOW","wiki","elmo"]
    with open("rez.txt", "w") as file:
        for i, pipeline in enumerate(pipelines):
            pipeline.fit(X_train, y_train, relevance__rel=rel_train, type__typ=type_train)
            file.write("{} Test Accuracy: {}\n".format(pipelines_dict[i], pipeline.score(X_test, y_test)))
            print("{} Test Accuracy: {}".format(pipelines_dict[i], pipeline.score(X_test, y_test)))
