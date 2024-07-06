# Function to create a graph from the dependency tree
def create_dependency_graph(doc):
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((token.i, child.i))
    graph = nx.Graph(edges)
    return graph

# Function to find shortest path between entities
def find_shortest_paths(doc, graph, label1, label2):
    entities1 = [ent for ent in doc.ents if ent.label_ == label1]
    entities2 = [ent for ent in doc.ents if ent.label_ == label2]
    shortest_paths = {}
    for ent1 in entities1:
        shortest_distance = float('inf')
        closest_ent2 = None
        for ent2 in entities2:
            try:
                distance = nx.shortest_path_length(graph, source=ent1.root.i, target=ent2.root.i)
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_ent2 = ent2
            except nx.NetworkXNoPath:
                continue
        shortest_paths[ent1] = (closest_ent2, shortest_distance)
    return shortest_paths


def get_ancestors(token):
    ancestors = []
    while token.head != token:
        ancestors.append(token.head)
        token = token.head
    return ancestors

# Function to find shortest path between two tokens in the
# dependency tree
def find_shortest_path(token1, token2):
    ancestors1 = get_ancestors(token1)
    ancestors2 = get_ancestors(token2)
    ancestors2.insert(0,token2)
    #print(f"Ancestors 1 {ancestors1}")
    #print(f"Ancestors 2 {ancestors2}")
    # Find the lowest common ancestor
    common_ancestor = None
    for ancestor in ancestors1:
        if ancestor in ancestors2:
            common_ancestor = ancestor
            break
    if common_ancestor is None:
        return float('inf')
    # Calculate the distance as the number of nodes in the dependency tree
    #print(f"Common ancestor {common_ancestor}")
    distance1 = ancestors1.index(common_ancestor) + 1
    distance2 = ancestors2.index(common_ancestor) + 1
    #print(f"Distance1 = {distance1} and Distance2 = {distance2}")
    distance = distance1 + distance2
    return distance

# Function to find shortest paths between entities
def find_shortest_paths(doc, label1, label2):
    entities1 = [ent for ent in doc.ents if ent.label_ in label1]
    entities2 = [ent for ent in doc.ents if ent.label_ in label2]
    shortest_paths = {}
    for ent1 in entities1:
        shortest_distance = float('inf')
        closest_ent2 = None
        for ent2 in entities2:
            distance = find_shortest_path(ent1.root, ent2.root)
            if distance <= shortest_distance:
                shortest_distance = distance
                closest_ent2 = ent2
        shortest_paths[ent1] = (closest_ent2, shortest_distance)
    return shortest_paths

label1 = ["CARDINAL", "PERCENTAGE"] 
label2 = ['TREATMENT', 'INOCTYPE']