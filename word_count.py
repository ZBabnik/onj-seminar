from collections import Counter

import numpy as np

from readXls import ReadXls
from class_dependency import make_value_dict, get_labels
from tokeniser import tokeniser

import matplotlib.pyplot as plt

"""
    Plot word distributions
"""
def plot_dist(data, labels, name):

    # Create a figure for all labels of class
    for i, category in enumerate(data):
        if len(category) < 1:
            continue

        # Create figure
        fig, ax = plt.subplots()

        # Get values and labels
        xlabels, values = zip(*sorted(list(category), key=lambda x: x[1], reverse=True)[:10])

        # X tick list
        x = np.arange(len(xlabels))
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)

        # Plot values
        ax.bar(x, values)

        # Set title
        plt.title(str(labels[i])+"-"+name)

        # Rotate labels
        plt.xticks(rotation=90)

        # Show plot
        fig.tight_layout()
        plt.show()

"""
    Normalize data
"""
def normalize_wc(per_class_wc, prior_wc, threshold):

    # Make word: count dictionary
    prior_wc_dict = dict((x, y) for x, y in tuple(prior_wc.items()))

    # List for storing results
    norm_wc = [[] for _ in range(len(per_class_wc))]
    # Iterate over all labels
    for i, cls in enumerate(per_class_wc):
        # List for storing results of one label
        norm_cls_wc = [tuple() for _ in range(len(per_class_wc[i].items()))]
        # Normalize
        for j, (key, value) in enumerate(cls.items()):
            # Remove infrequent words and one letter words
            if value > threshold and len(key) > 1:
                norm_value = float(value) / float(prior_wc_dict[key])
                norm_cls_wc[j] = (key, norm_value)
        norm_wc[i] = list(filter(None, norm_cls_wc))

    return norm_wc

"""
    Return classs label weights
"""
def calculate_weights(cls):

    return {key: (len(cls)/value) / 1000 for key, value in Counter(cls).items()}



"""
    
"""
def category_wordcount(cls, data, name, weighted=False, tau=20):

    if weighted:
        cls_weights = calculate_weights(cls)

    # Split sentences into tokens
    lemmas = list(map(lambda t: list(map(lambda u: u.lower(), tokeniser(str(t)))), data))

    # Get label: int mappings
    cls_dict = make_value_dict(cls)

    # Count all word occurrences
    prior_wc = Counter()
    # Count word occurrences per class
    per_class_wc = [Counter() for _, _ in cls_dict.items()]
    for i, data in enumerate(zip(lemmas, cls)):
        lemma, label = data
        lemma_counter = Counter(lemma)
        if weighted:
            for word in lemma_counter.keys():
                lemma_counter[word] = lemma_counter[word] * cls_weights[label]
        prior_wc += lemma_counter
        per_class_wc[cls_dict[cls[i]]] += lemma_counter

    # Normalize by number of all occurrences
    norm_per_class_wc = normalize_wc(per_class_wc, prior_wc, tau)

    # Plot normalized distributions
    plot_dist(norm_per_class_wc, get_labels(cls_dict), name)


if __name__ == "__main__":

    # Create excel table reader object
    xls = ReadXls("data/AllDiscussionData.xls")
    # Read messages
    messages = np.array(xls.get_column_with_name("Message"))
    # Read relevance class
    relevance = np.array(list(filter(lambda t: t, xls.get_column_with_name("Book relevance"))))
    # Read type class
    type = np.array(list(filter(lambda t: t, xls.get_column_with_name("Type"))))
    # Read category class
    category = np.array(list(filter(lambda t: t, xls.get_column_with_name("Category"))))
    # Read category broad class
    category_broad = np.array(list(filter(lambda t: t, xls.get_column_with_name("CategoryBroad"))))

    # Count words
    category_wordcount(category, messages, "Category", weighted=False, tau=20)

