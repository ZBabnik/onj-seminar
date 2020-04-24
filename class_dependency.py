
import numpy as np
import matplotlib.pyplot as plt

from readXls import ReadXls



def make_value_dict(data):

    value_dict = dict()
    for i, item in enumerate(set(data)):
        value_dict[item] = i

    return value_dict


def plot_prior(data, labels, name):

    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.plot(x, data, marker="o")
    plt.title(name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=90)
    plt.show()


def plot_posterior(data, labels1, labels2, name):

    x = np.arange(len(labels2))

    fig, ax = plt.subplots()

    for i in range(len(data)):
        print(data[i])
        ax.plot(x, data[i], label=labels1[i], marker="o")

    ax.set_xticks(x)
    ax.set_xticklabels(labels2)
    plt.title(name)
    plt.legend()
    plt.xticks(rotation=90)
    fig.tight_layout()
    plt.show()


def get_labels(dict_val):
    labels = ["" for _, _ in dict_val.items()]
    for key, item in dict_val.items():
        labels[item] = key
    return labels


def compare_distributions(data1, data2, cls1_name, cls2_name):

    cls1 = make_value_dict(data1)
    cls2 = make_value_dict(data2)

    cls1_cls2_relation = [[0 for _, _ in cls2.items()] for _, _ in cls1.items()]
    cls2_relation = [0 for _, _ in cls2.items()]
    for d1, d2 in zip(data1, data2):
        cls1_cls2_relation[cls1[d1]][cls2[d2]] += 1
        cls2_relation[cls2[d2]] += 1

    plot_prior(cls2_relation, get_labels(cls2), str(cls2_name)+" distribution")
    plot_posterior(cls1_cls2_relation, get_labels(cls1), get_labels(cls2), str(cls1_name)+"-"+str(cls2_name)+" distribution")


def join_data(relevance, type):
    return [str(r)+"+"+str(l) for r, l in zip(relevance, type)]


if __name__ == "__main__":
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = np.array(xls.get_column_with_name("Message"))
    relevance = np.array(list(filter(lambda t: t, xls.get_column_with_name("Book relevance"))))  # ground truth
    type = np.array(list(filter(lambda t: t, xls.get_column_with_name("Type"))))  # ground truth
    messages_gt = np.array(list(filter(lambda t: t, xls.get_column_with_name("Category"))))
    ctg_brd = np.array(list(filter(lambda t: t, xls.get_column_with_name("CategoryBroad"))))

    #compare_distributions(relevance, ctg_brd, "Relevance", "CategoryBroad")
    #compare_distributions(relevance, messages_gt, "Relevance", "Category")
    #compare_distributions(type, ctg_brd, "Type", "CategoryBroad")
    #compare_distributions(type, messages_gt, "Type", "Category")
    compare_distributions(join_data(relevance, type), messages_gt, "Relevance+Type", "Category")