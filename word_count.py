from collections import Counter

import numpy as np

from readXls import ReadXls
from class_dependency import make_value_dict, get_labels
from tagger import Tagger
from tokeniser import tokeniser

import matplotlib.pyplot as plt


def plot_dist(data, labels, name):

    for i, category in enumerate(data):
        if len(category) < 1:
            continue
        fig, ax = plt.subplots()
        xlabels, values = zip(*sorted(list(category), key=lambda x: x[1], reverse=True)[:10])
        #print(values)
        x = np.arange(len(xlabels))
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.bar(x, values)
        plt.title(str(labels[i])+"-"+name)
        plt.xticks(rotation=90)
        fig.tight_layout()
        plt.show()


def normalize_wc(per_class_wc, prior_wc):

    prior_wc_dict = dict((x, y) for x, y in tuple(prior_wc.items()))

    norm_wc = [[] for _ in range(len(per_class_wc))]
    for i, cls in enumerate(per_class_wc):
        norm_cls_wc = [tuple() for _ in range(len(per_class_wc[i].items()))]
        for j, (key, value) in enumerate(cls.items()):
            if value > 30 and len(key) > 1:
                norm_value = float(value) / float(prior_wc_dict[key])
                norm_cls_wc[j] = (key, norm_value)
        norm_wc[i] = list(filter(None, norm_cls_wc))

    return norm_wc

def category_wordcount(cls, data, name):

    lemmas = list(map(lambda t: list(map(lambda u: u.lower(), tokeniser(str(t)))), data))

    cls_dict = make_value_dict(cls)

    prior_wc = Counter()
    per_class_wc = [Counter() for _, _ in cls_dict.items()]
    for i, lemma in enumerate(lemmas):
        prior_wc += Counter(lemma)
        per_class_wc[cls_dict[cls[i]]] += Counter(lemma)

    norm_per_class_wc = normalize_wc(per_class_wc, prior_wc)

    plot_dist(norm_per_class_wc, get_labels(cls_dict), name)


if __name__ == "__main__":
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = np.array(xls.get_column_with_name("Message"))
    relevance = np.array(list(filter(lambda t: t, xls.get_column_with_name("Book relevance"))))  # ground truth
    type = np.array(list(filter(lambda t: t, xls.get_column_with_name("Type"))))  # ground truth
    messages_gt = np.array(list(filter(lambda t: t, xls.get_column_with_name("Category"))))
    ctg_brd = np.array(list(filter(lambda t: t, xls.get_column_with_name("CategoryBroad"))))

    #category_wordcount(relevance, messages, "Relevance")
    #category_wordcount(type, messages, "Type")
    category_wordcount(ctg_brd, messages, "CategoryBroad")