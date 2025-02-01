import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Define the feature hierarchy
feature_tree = {
    "Implemented Features": {
        "Time-Domain Features": [
            "Mean", "Standard Deviation", "Variance", "Skewness", "Kurtosis",
            "Zero Crossing Rate", "Hjorth Mobility", "Hjorth Complexity", "75th Percentile"
        ],
        "Frequency-Domain Features": [
            "Delta Power", "Theta Power", "Alpha Power", "Beta Power",
            "Delta/Theta Ratio", "Theta/Alpha Ratio", "Alpha/Beta Ratio",
            "(Theta+Delta) / (Alpha+Beta) Ratio",
            "Spectral Edge Frequency", "Median Frequency",
            "Mean Frequency Difference", "Peak Frequency", "Spectral Entropy"
        ],
        "Wavelet-Based (CWT) Features": [
            "CWT Delta Variance", "CWT Theta Variance", "CWT Alpha Variance", "CWT Beta Variance",
            "CWT Delta Skewness", "CWT Theta Skewness", "CWT Alpha Skewness", "CWT Beta Skewness",
            "CWT Delta Entropy", "CWT Theta Entropy", "CWT Alpha Entropy", "CWT Beta Entropy",
            "CWT Activation Duration", "CWT Relative Power",
            "CWT Delta/Theta Ratio", "CWT Theta/Alpha Ratio", "CWT Alpha/Beta Ratio"
        ],
        "Non-Linear Features": [
            "Lempel-Ziv Complexity"
        ]
    }
}

# Create a directed graph
G = nx.DiGraph()

# Recursive function to add nodes and edges
def add_nodes_edges(graph, parent, children):
    for child in children:
        if isinstance(children, dict):  # If child has subcategories
            graph.add_edge(parent, child)
            add_nodes_edges(graph, child, children[child])
        else:  # If child is a list of features
            graph.add_edge(parent, child)

# Construct the tree
add_nodes_edges(G, "Implemented Features", feature_tree["Implemented Features"])

# Draw the tree graph using a spring layout (alternative to Graphviz)
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42, k=0.4)  # Adjust layout for better visibility
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2500, font_size=9, arrows=False)
plt.title("Overview of Implemented Features")
plt.show()
