import pickle
import networkx as nx
import matplotlib.pyplot as plt

with open("ReSimLogsP2/semilla_2/ModeloDeFinitivo.pkl", "rb") as f:
    genome = pickle.load(f)

G = nx.DiGraph()

for c in genome.connections.values():
    if c.enabled:
        G.add_edge(c.key[0], c.key[1], weight=c.weight)

f = plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_color="lightblue")
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, arrows=True)

def guardar(fig):
    fig.savefig("figura.pdf", dpi=200)
    plt.close(fig)


plt.axis("off")
plt.show()
guardar(f)