
import numpy as np
import networkx as nx

from readXls import ReadXls

"""
    Build Markov chains for data
"""
def get_sequences(data, prev):

    # Create directed network
    G = nx.DiGraph()

    # Create list of tuples of all sequences of prev + 1 length
    all_seq = []
    for i in range(len(data)):
        if len(data[i]) >= prev + 1:
            for j in range(prev, len(data[i])):
                all_seq.append(("+".join([data[i][j - (prev - k)] for k in range(prev)]), data[i][j]))

    # Initialize all nodes
    for start, end in all_seq:
        G.add_node(start, nof_outedges=0)
        G.add_node(end, nof_outedges=0)

    # Create all edges
    cummulative_weights = 0                             # Count number of all edges
    for start, end in all_seq:
        if G.has_edge(start, end):
            G[start][end]["label"] += 1
            cummulative_weights += 1
        else:
            G.add_edge(start, end, label=1)
            cummulative_weights += 1
        G.nodes[start]["nof_outedges"] += 1

    # Find less frequent edges
    to_remove_edges = []
    for u, v, data in G.edges(data=True):
        if data["label"] < (cummulative_weights * 0.01):
            to_remove_edges.append((u, v))
    G.remove_edges_from(to_remove_edges)

    # Remove less frequent edges and isolates
    to_remove_nodes = nx.isolates(G)
    G.remove_nodes_from(list(to_remove_nodes))

    # Normalize the edges weights
    for u, v, data in G.edges(data=True):
        data["label"] = round(float(data["label"]) / float(G.nodes[u]["nof_outedges"]), 3)

    # Create .dot file
    nx.drawing.nx_pydot.write_dot(G, 'graph.dot')


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
    # Read topic labels
    topic = np.array(list(filter(lambda t: t, xls.get_column_with_name("Topic"))))
    # Read bookclub labels
    bookclub = np.array(list(filter(lambda t: t, xls.get_column_with_name("Bookclub"))))
    # Read message time labels
    time = np.array(list(filter(lambda t: t, xls.get_column_with_name("Message Time"))))

    # Create a joint label for topic, bookclub, time and category
    joint_data = []
    for i, data in enumerate(zip(topic, bookclub, time, category)):
        # Addressing problems with two different time stamp structures
        if len(data[2].split(" ")) > 1:
            temp_date = data[2].split(" ")[0]
        else:
            temp_date = -1
        joint_data.append((data[0], data[1], data[2], i, temp_date, data[3]))

    # Sort joint data by topic then bookclub then time and then index
    joint_data = sorted(joint_data, key=lambda x: (x[0], x[1], x[2], x[3]))

    # Cut joint data by topic club and date
    currTopic, currClub, currDate = "", "", ""
    data_cut, temp = [], []
    for line in joint_data:
        if line[0] == currTopic and line[1] == currClub and line[4] == currDate:
            temp.append(line[5])
        else:
            data_cut.append(temp)
            currTopic = line[0]
            currClub = line[1]
            currDate = line[4]
            temp = []

    # Remove empty lines
    data_cut = list(filter(None, data_cut))

    # Get Markov chains from data
    get_sequences(data_cut, 2)