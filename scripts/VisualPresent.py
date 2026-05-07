import matplotlib.pyplot as plt
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes for input layer
input_nodes = ['Input 1', 'Input 2', '...', 'Input 63']
G.add_nodes_from(input_nodes, layer='input')

# Add nodes for hidden layer
hidden_nodes = ['Hidden 1', 'Hidden 2', '...', 'Hidden 128']
G.add_nodes_from(hidden_nodes, layer='hidden')

# Add nodes for output layer
output_nodes = ['Output 1', 'Output 2', '...', 'Output 11']
G.add_nodes_from(output_nodes, layer='output')

# Add edges between input and hidden layers
for input_node in input_nodes:
    for hidden_node in hidden_nodes:
        G.add_edge(input_node, hidden_node)

# Add edges between hidden and output layers
for hidden_node in hidden_nodes:
    for output_node in output_nodes:
        G.add_edge(hidden_node, output_node)

# Define positions for nodes
pos = {}
layer_distances = [0, 1, 2]  # Distances between layers
layers = [input_nodes, hidden_nodes, output_nodes]

for i, layer in enumerate(layers):
    x = layer_distances[i]
    y = range(len(layer))
    for j, node in enumerate(layer):
        pos[node] = (x, y[j])

# Draw the network
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=False)

# Title
plt.title('Simplified Neural Network Diagram', size=15)

# Show the plot
plt.show()
