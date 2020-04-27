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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from functools import lru_cache
from sklearn.base import TransformerMixin, BaseEstimator

# preload all word vector models
models = {"embeddings/wiki.sl.magnitude": Magnitude("embeddings/wiki.sl.magnitude"),
          "embeddings/slovenian-elmo.weights.magnitude": Magnitude("embeddings/slovenian-elmo.weights.magnitude")}


class Tagg:
    """
    tokenises the messages and also adds pos taggs of tokens.
    """
    def __init__(self, lemmatizer):
        self.lemmatizer = lemmatizer

    def fit(self, X, y=None, **kwargs):
        return self

    def isQuestion(self, sentence):
        for word in ["kdo", "kje", "kaj", "kdaj", "zakaj", "kako", "?", "ali"]:
            if word in sentence:
                return (word + " ") * 5 + sentence
        # for word in sentence.split():
          #  if len(word) > 15:
           #     return "klnlcknlakncklankankasncsincsncklnskncknsaklsnklsnklsnklsnklskankslsanksn"
        return sentence

    def transform(self, X, y=None, *args, **kwargs):
        # remove all non alphanumeric characters
        # no_non_alphanumeric_chars = map(lambda t: sub(r'[^a-zA-Z0-9]+', ' ', str(t)), X[:, 0])
        # if a letter is repeated 3 times its most likely to emphasize the text so its shortened to 1 repetition
        #no_triple_chars = map(lambda t: sub(r'(\w)\1\1*', r'\1', str(t)), X[:, 0])
        # get lemmas change all lemmas to lower case

        weighted_questions = map(lambda t: self.isQuestion(t.lower()),  X[:, 0])

        tokens = np.array(list(map(lambda t: np.array([list(map(lambda u: u.lower(), tokeniser(str(t)))),
                                        " ".join(list(map(lambda u: "a"+u[:2], self.lemmatizer.tagger(str(t))[1])))]),
                                   weighted_questions))) # tokenise and add taggs
        return np.append(tokens, X[:, 1].reshape(-1,1), axis=1)


class ToStr:
    """
    excel data is not in str therefore we need this before BOW
    """
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        return list(map(str, X))


class ToArray(TransformerMixin, BaseEstimator):
    """
    converts BOW output to numpy array
    """
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, *args, **kwargs):
        return X.toarray()


# used for speeding up same sentence queries
@lru_cache(maxsize=None)
def query(model_path, sentence):
    """
    :param model_path: string for finding preloaded model
    :param sentence: array of tokens in string
    :return: word vector for each token
    """
    if len(sentence) == 0:
        # if array is empty put in dummy character.
        sentence = ("_")
    return models[model_path].query(list(sentence))


class WordEmbeddings:
    """
    converts tokens to word vectors and sums them for each sentence
    tags and topic are converted into BOW format and appended to the end of the sum of token word vectors of message
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.BOWtopic = CountVectorizer(stop_words=None)
        self.BOW = CountVectorizer(stop_words=None, ngram_range=(1,2))
        self.BOW.fit(["aVm","aAp","aPq", "aZ", "aNc", "aRg"])  # if we want to search only for specific tags

    def fit(self, X, y=None, **kwargs):
        self.BOWtopic.fit(X[:,2])
        #self.BOW.fit(X[:,1])
        return self

    def transform(self, X, y=None, **kwargs):
        word_vec = list(map(lambda t: np.sum(query(self.model_path, tuple(t)), axis=0), X[:,0]))
        return np.append(np.append(word_vec, self.BOWtopic.transform(X[:, 2]).toarray(), axis=1)
                          ,np.maximum(1,self.BOW.transform(X[:,1]).toarray()), axis=1) # gives bad result


class GetRelevance:
    """
    predicts book relevance and adds the value to the vector
    """
    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y=None, **kwargs):
        self.pipe.fit(X, kwargs["rel"])
        return self

    def transform(self, X, y=None, **kwargs):
        tmp = np.array(list(map(lambda t: 1 if (t == "Yes") else -1, self.pipe.predict(X))), dtype=float)
        return np.append(X, tmp.reshape(-1, 1), axis=1)


class GetType:
    """
    predicts message type and adds the value to the vector
    """

    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y=None, **kwargs):
        self.pipe.fit(X, kwargs["typ"])
        return self

    def transform(self, X, y=None, **kwargs):
        tmp = np.array(list(map(lambda t: -1 if (t == "A") else 1 if (t == "Q") else 0, self.pipe.predict(X))), dtype=float)
        return np.append(X, tmp.reshape(-1, 1), axis=1)


if __name__ == "__main__":
    # read the data
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = np.array(xls.get_column_with_name("Message")).reshape(-1,1)
    topic = np.array(list(map(lambda t: str(t).replace(" ", "")[::11], xls.get_column_with_name("Topic"
                                                                                          )))).reshape(-1,1)
    relevance = np.array(list(filter(lambda t: t, xls.get_column_with_name("Book relevance"))))  # ground truth
    type = np.array(list(filter(lambda t: t, xls.get_column_with_name("Type"))))  # ground truth
    messages_gt = np.array(list(filter(lambda t: t, xls.get_column_with_name("Category"))))
    X = np.append(messages, topic, axis=1)

    # split train / test
    X_train, X_test, y_train, y_test, rel_train, rel_test, type_train, type_test\
        = train_test_split(X, messages_gt, relevance, type, test_size=0.3, random_state=0)

    # params for gridsearchCV
    fit_params = {
        "classify__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        "classify__class_weight": ['balanced', None],
        "classify__penalty": ['l1', 'elasticnet', 'l2', 'none'],
        "classify__l1_ratio": [0.5],
        "classify__multi_class": ["ovr", "multinomial"],
        "classify__max_iter": [500]
    }

    # sklearn pipelines

    pipeline_lr1 = Pipeline([('str', ToStr()),
                             ('BOW', CountVectorizer(ngram_range=(1,2))),
                             ('toArray', ToArray()),
                             ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                             #('type', GetType(pipe=LogisticRegression(random_state=0))),
                             ('classify', LogisticRegression(random_state=0))])

    pipeline_lr2 = Pipeline([('scalar1', Tagg(Tagger())),
                             ('word2vecW', WordEmbeddings("embeddings/wiki.sl.magnitude")),
                             ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                             #('type', GetType(pipe=LogisticRegression(random_state=0))),
                             ('classify', LogisticRegression(random_state=0))])

    # params for gridsearchCV for elmo WV
    fit_params = {
        "classify__solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        "classify__class_weight": ['balanced', None],
        "classify__penalty": ['l1', 'elasticnet', 'l2', 'none'],
        "classify__l1_ratio": [0.5],
        "classify__max_iter": [500]
    }
    pipeline_lr3 = Pipeline([('scalar2', Tagg(Tagger())),
                             ('word2vecW', WordEmbeddings("embeddings/slovenian-elmo.weights.magnitude")),
                             ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                             #('type', GetType(pipe=LogisticRegression(random_state=0))),
                             ('classify', LogisticRegression(random_state=0))])

    pipelines = [pipeline_lr1, pipeline_lr2, pipeline_lr3]
    pipelines_dict = ["BOW", "wiki", "elmo2"]
    # fit and predict
    for i, pipeline in enumerate(pipelines):
        pipeline.fit(X_train, y_train,  relevance__rel=rel_train)
        print("{} Test Accuracy: {}".format(pipelines_dict[i], pipeline.score(X_test, y_test)))
