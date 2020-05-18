
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from readXls import ReadXls




def get_sequences(data, prev):

    G = nx.DiGraph()

    all_seq = []
    for i in range(len(data)):
        if len(data[i]) >= prev + 1:
            for j in range(prev, len(data[i])):
                all_seq.append(("+".join([data[i][j - (prev - k)] for k in range(prev)]), data[i][j]))

    #for x in all_seq:
    #    print(x)

    for start, end in all_seq:
        G.add_node(start, nof_outedges=0)
        G.add_node(end, nof_outedges=0)

    cummulative_weights = 0
    for start, end in all_seq:
        if G.has_edge(start, end):
            G[start][end]["label"] += 1
            cummulative_weights += 1
        else:
            G.add_edge(start, end, label=1)
            cummulative_weights += 1
        G.nodes[start]["nof_outedges"] += 1

    to_remove_edges = []
    for u, v, data in G.edges(data=True):
        if data["label"] < (cummulative_weights * 0.005):
            to_remove_edges.append((u, v))
    G.remove_edges_from(to_remove_edges)

    to_remove_nodes = nx.isolates(G)
    G.remove_nodes_from(list(to_remove_nodes))

    for u, v, data in G.edges(data=True):
        data["label"] = round(float(data["label"]) / float(G.nodes[u]["nof_outedges"]), 3)

    nx.drawing.nx_pydot.write_dot(G, 'graph.dot')

if __name__ == "__main__":
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = np.array(xls.get_column_with_name("Message"))
    relevance = np.array(list(filter(lambda t: t, xls.get_column_with_name("Book relevance"))))  # ground truth
    type = np.array(list(filter(lambda t: t, xls.get_column_with_name("Type"))))  # ground truth
    messages_gt = np.array(list(filter(lambda t: t, xls.get_column_with_name("Category"))))
    ctg_brd = np.array(list(filter(lambda t: t, xls.get_column_with_name("CategoryBroad"))))
    topic = np.array(list(filter(lambda t: t, xls.get_column_with_name("Topic"))))
    bookclub = np.array(list(filter(lambda t: t, xls.get_column_with_name("Bookclub"))))
    time = np.array(list(filter(lambda t: t, xls.get_column_with_name("Message Time"))))

    joint_data = []
    for i, data in enumerate(zip(topic, bookclub, time, messages_gt)):
        if len(data[2].split(" ")) > 1:
            temp_date = data[2].split(" ")[0]
        else:
            temp_date = -1
        joint_data.append((data[0], data[1], data[2], i, temp_date, data[3]))
    joint_data = sorted(joint_data, key=lambda x: (x[0], x[1], x[2], x[3]))

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

    data_cut = list(filter(None, data_cut))

    #for x in joint_data:
    #    print(x)

    #for x in data_cut:
    #    print(x)

    get_sequences(data_cut, 3)