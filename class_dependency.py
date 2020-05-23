
import numpy as np
import matplotlib.pyplot as plt

from readXls import ReadXls


"""
    Creates dictionary of label: int values
"""
def make_value_dict(data):
    value_dict = dict()
    for i, item in enumerate(sorted(set(data))):
        value_dict[item] = i

    return value_dict

"""
    Return color based on label
"""
def get_color(label):

    if label[0] == "C":
        return "r"
    elif label[0] == "D":
        return "b"
    elif label[0] == "I":
        return "g"
    elif label[0] == "M":
        return "k"
    else:
        return "y"

"""
    Plot the prior distribution
"""
def plot_prior(data, labels, name):

    # Create figure
    fig, ax = plt.subplots()

    # X ticks list
    x = np.arange(len(labels))

    # Get data and labels
    data, labels = zip(*list(sorted(zip(data, labels), key=lambda x: x[0], reverse=True)))

    # Set colors for category_broad labels
    _colors = [get_color(label) for label in labels]

    # Plot data
    ax.bar(x, data, color=_colors)

    # Set x axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Set title
    plt.title(name)

    # Rotate labels
    plt.xticks(rotation=90)

    # Show plot
    plt.show()

    return labels

"""
    Plot the posterior distribution
"""
def plot_posterior(data, labels1, labels2, name, prior_labels):

    # Create figure
    fig, ax = plt.subplots()

    # X ticks list
    x = np.arange(len(labels2))

    # Remove labels with little data
    total_sum = sum([sum(x) for x in data])
    data, labels1 = zip(*[(x, y) for x, y in zip(data, labels1) if sum(x) > total_sum * 0.1])

    # List for data reordering
    _transform = [labels2.index(label) for label in prior_labels]

    # Plot data
    for i in range(len(data)):
        ax.plot(x, [data[i][j] for j in _transform], label=labels1[i], marker="o")

    # Set x ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels([labels2[j] for j in _transform])

    # Set title and legend
    plt.title(name)
    plt.legend()

    # Tilt text
    plt.xticks(rotation=90)

    # Show plot
    fig.tight_layout()
    plt.show()


"""
    Returns order list of labels
"""
def get_labels(dict_val):
    labels = ["" for _, _ in dict_val.items()]
    for key, item in dict_val.items():
        labels[item] = key
    return labels

"""
    Plots prior/posterior distribution
"""
def compare_distributions(data1, data2, cls1_name, cls2_name):

    # Build label: int dictionary for data1/data2
    cls1 = make_value_dict(data1)
    cls2 = make_value_dict(data2)

    # Build list that will hold the results
    cls1_cls2_relation = [[0 for _, _ in cls2.items()] for _, _ in cls1.items()]
    cls2_relation = [0 for _, _ in cls2.items()]
    # Iterate over all data labels
    for d1, d2 in zip(data1, data2):
        # Count prior and posterior distributions
        cls1_cls2_relation[cls1[d1]][cls2[d2]] += 1
        cls2_relation[cls2[d2]] += 1

    # Plot the prior distribution
    labels = plot_prior(cls2_relation, get_labels(cls2), str(cls2_name)+" distribution")
    # Plot posterior distribution
    plot_posterior(cls1_cls2_relation, get_labels(cls1), get_labels(cls2), str(cls1_name)+"-"+str(cls2_name)+" distribution", labels)

"""
    Joins two labels into one
"""
def join_data(data1, data2):
    return [str(r)+"+"+str(l) for r, l in zip(data1, data2)]


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

    # Run category distribution comparison by first splitting the data by relevance+type
    compare_distributions(join_data(relevance, type), category, "Relevance+Type", "Category")
