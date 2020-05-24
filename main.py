from readXls import ReadXls
from tagger import Tagger
from tokeniser import tokeniser
from pymagnitude import Magnitude
import matplotlib.pyplot as plt
from re import sub
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from functools import lru_cache
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
import re


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
        return sentence

    def transform(self, X, y=None, *args, **kwargs):
        # remove all non alphanumeric characters
        # no_non_alphanumeric_chars = map(lambda t: sub(r'[^a-zA-Z0-9]+', ' ', str(t)), X[:, 0])
        # if a letter is repeated 3 times its most likely to emphasize the text so its shortened to 1 repetition
        no_triple_chars = map(lambda t: sub(r'(\w)\1\1*', r'\1', str(t)), X[:, 0])

        weighted_questions = map(lambda t: self.isQuestion(t.lower()),  no_triple_chars)

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

    def addToGibberish(self, sentence):
        if predictGibberishWords(sentence[0]) == -1:
            sentence[0] = sentence[0] + "gibberish"
        return sentence

    def transform(self, X, y=None, *args, **kwargs):
        return list(str(self.addToGibberish(t)) for t in X)


class ToArray:
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
                          ,np.maximum(1,self.BOW.transform(X[:,1]).toarray()), axis=1)


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


class GetGibberish:
    """
    predicts gibberish words and adds the value to the vector
    """
    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y=None, **kwargs):
        self.pipe.fit(X, kwargs["gib"])
        return self

    def transform(self, X, y=None, **kwargs):
        tmp = np.array(list(self.pipe.predict(X)), dtype=float)
        return np.append(X, tmp.reshape(-1, 1), axis=1)


def predictGibberishWords(sentence):
    sentence = sentence.replace(".", "").replace("!", "").replace("?", "").replace(",", " ")
    if len(sentence) < 2:
        return -1
    for i, word in enumerate(sentence.split()):
        if len(word) > 15 or re.search(r'(.)\1\1', word) \
                or re.search(r'[^\w\s,]', word) and i == 0 \
                or any(i.isdigit() for i in word) and not all(i.isdigit() for i in word) and not word[-1].isdigit():
        # if longer than 15 char or sole letter repeats pr has digit in a word, but not in the last place (usernames)
            return -1
    return 1


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


class GetCategory:
    """
    predicts message Category and adds the value to the vector
    """

    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y=None, **kwargs):
        self.pipe.fit(X, kwargs["cat"])
        return self

    def transform(self, X, y=None, **kwargs):
        tmp = np.array(list(map(lambda t: 0 if (t == "CG") else 1 if (t == "CC") else 2 if (t == "CE")
            else 3 if (t == "CF") else 4 if (t == "CO") else 5 if (t == "CB")
            else 6 if (t == "S") else 7 if (t == "DQ") else 8 if (t == "DE")
            else 9 if (t == "DA") else 10 if (t == "DAA") else 11 if (t == "ME")
            else 12 if (t == "MQ") else 13 if (t == "MA") else  15 if (t == "IQ")
            else 16 if (t == "IA") else 17 if (t == "IQA") else 18, self.pipe.predict(X))), dtype=float)
        return np.append(X, tmp.reshape(-1, 1), axis=1)


class GetCategoryBroad:
    """
    predicts message CategoryBroad and adds the value to the vector
    """

    def __init__(self, pipe):
        self.pipe = pipe

    def fit(self, X, y=None, **kwargs):
        self.pipe.fit(X, kwargs["ctb"])
        return self

    def transform(self, X, y=None, **kwargs):
        tmp = np.array(list(map(lambda t: 0 if (t == "C") else 1 if (t == "O") else 2 if (t == "I")
            else 3 if (t == "D") else 4, self.pipe.predict(X))), dtype=float)
        return np.append(X, tmp.reshape(-1, 1), axis=1)


class ResampleLR(LogisticRegression):
    """
    wrapper so that estimator receives resampled values in fitting
    """
    def __init__(self, random_state=None, solver='lbfgs'):
        # initiate over samplers
        self.small_class_sampler = RandomOverSampler(random_state=42, sampling_strategy='minority')
        self.sampler = SMOTE(random_state=42, sampling_strategy='auto')
        super().__init__(random_state=random_state, solver=solver)

    def fit(self, X, y=None, **kwargs):
        # overwrite predictor so that it over samples first when fitting
        print("start sampling")
        while True:
            try:
                X, y = self.sampler.fit_resample(X, y)
                break
            except:
                # if one class has less than 6 elements this happenes we use random over sampler for that class
                # since smote requires at least 6 different elements with same class
                print("small class resampling")
                X, y = self.small_class_sampler.fit_resample(X, y)
        print("fin sampling")
        super().fit(X=X, y=y, **kwargs)


class ResampleMLP(MLPClassifier):
    """
    wrapper so that estimator receives resampled values in fitting
    """
    def __init__(self, hidden_layer_sizes=(100)):
        # initiate over samplers
        self.small_class_sampler = RandomOverSampler(random_state=42, sampling_strategy='minority')
        self.sampler = SMOTE(random_state=42, sampling_strategy='auto')
        super().__init__(hidden_layer_sizes=hidden_layer_sizes)

    def fit(self, X, y=None, **kwargs):
        # overwrite predictor so that it over samples first when fitting
        print("start sampling")
        while True:
            try:
                X, y = self.sampler.fit_resample(X, y)
                break
            except:
                # if one class has less than 6 elements this happenes we use random over sampler for that class
                # since smote requires at least 6 different elements with same class
                print("small class resampling")
                X, y = self.small_class_sampler.fit_resample(X, y)
        print("fin sampling")
        super().fit(X=X, y=y, **kwargs)


class JoinResampleAndNormal:
        """
        predicts message CategoryBroad and adds the value to the vector
        """

        def __init__(self, pipe1, pipe2):
            self.pipe1 = pipe1
            self.pipe2 = pipe2
            self.precision1 = None
            self.precision2 = None

        def fit(self, X, y=None, **kwargs):
            self.pipe1.fit(X, y, **kwargs)
            self.pipe2.fit(X, y, **kwargs)
            self.labels = self.pipe2.classes_
            pred1 = self.pipe1.predict(X)
            pred2 = self.pipe2.predict(X)
            self.precision1 = np.sqrt(precision_score(y, pred1, average=None, labels=self.labels, zero_division=0))
            self.precision2 = np.sqrt(precision_score(y, pred2, average=None, labels=self.labels, zero_division=0))
            return self

        def predict(self, X, y=None):
            pred = self.pipe1.predict_proba(X) * self.precision1 + self.pipe2.predict_proba(X) * self.precision2
            pred = np.apply_along_axis(lambda t: np.argmax(t), 1, pred).flatten()
            return self.labels[pred]


def read_data():
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = np.array(xls.get_column_with_name("Message")).reshape(-1, 1)
    topic = np.array(list(map(lambda t: str(t).replace(" ", "")[::11], xls.get_column_with_name("Topic"
                                                                                                )))).reshape(-1, 1)
    relevance = np.array(list(filter(lambda t: t, xls.get_column_with_name("Book relevance"))))  # ground truth
    type = np.array(list(filter(lambda t: t, xls.get_column_with_name("Type"))))  # ground truth
    category = np.array(list(filter(lambda t: t, xls.get_column_with_name("Category"))))  # ground truth
    category_broad = np.array(list(filter(lambda t: t, xls.get_column_with_name("CategoryBroad"))))
    gibberish = np.array(list(predictGibberishWords(s[0]) for s in messages))

    X = np.append(messages, topic, axis=1)

    return X, relevance, type, category, category_broad, gibberish


def get_all_pipelines(normal_model, resample_model, predicting_relevance):
    pipeline_lr11 = Pipeline([('str', ToStr()),
                              ('BOW', CountVectorizer(ngram_range=(1, 2))),
                              ('toArray', ToArray()),
                              ('gibberish', GetGibberish(pipe=LogisticRegression(random_state=0))),
                              ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                              normal_model])

    pipeline_lr12 = Pipeline([('str', ToStr()),
                              ('BOW', CountVectorizer(ngram_range=(1, 2))),
                              ('toArray', ToArray()),
                              ('gibberish', GetGibberish(pipe=LogisticRegression(random_state=0))),
                              ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                              resample_model])

    if predicting_relevance:
        pipeline_lr11.steps.pop(4)
        pipeline_lr12.steps.pop(4)

    pipeline_comb1 = JoinResampleAndNormal(pipeline_lr11, pipeline_lr12)

    pipeline_lr21 = Pipeline([('scalar1', Tagg(Tagger())),
                              ('word2vecW', WordEmbeddings("embeddings/wiki.sl.magnitude")),
                              ('gibberish', GetGibberish(pipe=LogisticRegression(random_state=0))),
                              ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                              normal_model])

    pipeline_lr22 = Pipeline([('scalar1', Tagg(Tagger())),
                              ('word2vecW', WordEmbeddings("embeddings/wiki.sl.magnitude")),
                              ('gibberish', GetGibberish(pipe=LogisticRegression(random_state=0))),
                              ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                              resample_model])

    if predicting_relevance:
        pipeline_lr21.steps.pop(3)
        pipeline_lr22.steps.pop(3)

    pipeline_comb2 = JoinResampleAndNormal(pipeline_lr21, pipeline_lr22)

    pipeline_lr31 = Pipeline([('scalar1', Tagg(Tagger())),
                              ('word2vecW', WordEmbeddings("embeddings/slovenian-elmo.weights.magnitude")),
                              ('gibberish', GetGibberish(pipe=LogisticRegression(random_state=0))),
                              ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                              normal_model])

    pipeline_lr32 = Pipeline([('scalar1', Tagg(Tagger())),
                              ('word2vecW', WordEmbeddings("embeddings/slovenian-elmo.weights.magnitude")),
                              ('gibberish', GetGibberish(pipe=LogisticRegression(random_state=0))),
                              ('relevance', GetRelevance(pipe=LogisticRegression(random_state=0))),
                              resample_model])

    if predicting_relevance:
        pipeline_lr31.steps.pop(3)
        pipeline_lr32.steps.pop(3)

    pipeline_comb3 = JoinResampleAndNormal(pipeline_lr31, pipeline_lr32)

    return [(pipeline_comb1, "BOW"), (pipeline_comb2, "Wiki"), (pipeline_comb3, "Elmo")]


def show_plot(f1, precision, recall, labels, pipeline_name):
    plt.plot(labels, f1, 'b-', label="F1")
    plt.plot(labels, precision, 'r-',
             label="precision")
    plt.plot(labels, recall, 'g-',
             label="recall")
    plt.xticks(rotation=90)
    plt.title("{} F1, precision, recall".format(pipeline_name))
    plt.xlabel("labels")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel("%")
    plt.legend()
    plt.show()


def evaluate(pipeline_name, test, pred):
    labels = list(set(test))
    print("{} Test Accuracy: {}".format(pipeline_name, accuracy_score(test, pred)))
    print("{} Test F1 micro: {}".format(pipeline_name, f1_score(test, pred, average="micro")))
    print("{} Test F1 macro: {}".format(pipeline_name, f1_score(test, pred, average="macro")))
    print("{} Test F1 weighted: {}".format(pipeline_name, f1_score(test, pred, average="weighted")))

    f1 = f1_score(test, pred, average=None, labels=labels, zero_division=1)
    precision = precision_score(test, pred, average=None, labels=labels, zero_division=0)
    recall = recall_score(test, pred, average=None, labels=labels, zero_division=0)

    pack = sorted(zip(f1, recall, precision, labels), reverse=True)
    f1 = [i[0] * 100 for i in pack]
    precision = [i[2] * 100 for i in pack]
    recall = [i[1] * 100 for i in pack]
    labels = [i[3] for i in pack]

    return f1, precision, recall, labels


if __name__ == "__main__":
    predicting = "relevance"  # options relevance type category category_broad
    classifier = "MLP"  # options MLP (neural_network) logistic_regression
    # read the data
    X, relevance, type, category, category_broad, gibberish = read_data()

    # split train / test
    X_train, X_test, category_train, category_test, relevance_train, relevance_test, type_train, type_test, \
        category_broad_train, category_broad_test, gibberish_train, gibberish_test \
            = train_test_split(X, category, relevance, type, category_broad, gibberish, test_size=0.3, random_state=0)

    # sklearn pipelines
    if classifier == "logistic_regression":
        resample_classifier = ('classify', ResampleLR(random_state=0, solver='lbfgs'))
        normal_classifier = ('classify', LogisticRegression(random_state=0, solver='lbfgs'))
    if classifier == "MLP":
        resample_classifier = ('classify', MLPClassifier(hidden_layer_sizes=(100)))
        normal_classifier = ('classify', ResampleMLP())

    pipelines = get_all_pipelines(normal_classifier, resample_classifier, predicting_relevance=predicting=="relevance")

    if predicting == "category":
        train = category_train
        test = category_test
    elif predicting == "category_broad":
        train = category_broad_train
        test = category_broad_test
    elif predicting == "relevance":
        train = relevance_train
        test = relevance_test
    elif predicting == "type":
        train = type_train
        test = type_test

    # fit and predict
    for pipeline, pipeline_name in pipelines:
        if predicting == "relevance":
            pipeline.fit(X_train, y=train, gibberish__gib=gibberish_train)
        else:
            pipeline.fit(X_train, y=train, relevance__rel=relevance_train, gibberish__gib=gibberish_train)
        pred = pipeline.predict(X_test)

        f1, precision, recall, labels = evaluate(pipeline_name, test, pred)

        show_plot(f1, precision, recall, labels, pipeline_name)
