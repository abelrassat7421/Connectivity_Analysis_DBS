import numpy as np
import bct
import networkx as nx

"""
We Implement here the alternative definition of the global and local efficiencies of a networ
as used in T. Xu et al. 2016 (circumvents the problem of inifinity appearing in the classical definition
when the graph is not fully connected)
"""

def charpath_and_global_eff(data, num_subjects=None):
    """
    Summary: computes the characteristic paths and the global efficiencies of a set of graphs 

    Args: 
    data (np.array): set of graphs with shape -> (num_atlas_reg)x(num_atlas_reg)x(num_subjects)
    num_subjects: in case not all of the thresholded graphs and null graphs have been computed

    Returns: 
    tuple (np.array, np.array): 1) characteristic paths of the set of graphs given
                          2) global efficiencies
    """
    num_atlas_reg = data.shape[0]
    if num_subjects == None: 
        num_subjects = data.shape[2]
    charpaths = []
    glob_eff = []
    for i in range(num_subjects):
        inv_distance_mat= 1/bct.distance_bin(data[:, :, i])
        np.fill_diagonal(inv_distance_mat, 0)
        dist_sum = np.sum(inv_distance_mat)
        charpaths.append(num_atlas_reg*(num_atlas_reg-1)/dist_sum)
        glob_eff.append(1/charpaths[i])
    return np.array(charpaths), np.array(glob_eff)

def charpath_and_global_eff_wei(data, num_subjects=None):
    """
    Summary: computes the characteristic paths and the global efficiencies of a set of graphs 

    Args: 
    data (np.array): set of graphs with shape -> (num_atlas_reg)x(num_atlas_reg)x(num_subjects)
    num_subjects: in case not all of the thresholded graphs and null graphs have been computed

    Returns: 
    tuple (np.array, np.array): 1) characteristic paths of the set of graphs given
                          2) global efficiencies
    """
    num_atlas_reg = data.shape[0]
    if num_subjects == None: 
        num_subjects = data.shape[2]
    charpaths = []
    glob_eff = []
    for i in range(num_subjects):
        # may need to tweek the use of distance_wei -> not clear if the function expecting adjacency matrix but with inverse weights i.e., smaller weights means shorter distances
        D, _ = bct.distance_wei(data[:, :, i])
        inv_distance_mat= 1/D
        np.fill_diagonal(inv_distance_mat, 0)
        dist_sum = np.sum(inv_distance_mat)
        charpaths.append(num_atlas_reg*(num_atlas_reg-1)/dist_sum)
        glob_eff.append(1/charpaths[i])
    return np.array(charpaths), np.array(glob_eff)

def path_length(subgraph, j, h):
    try:
        return 1 / nx.shortest_path_length(subgraph, j, h)
    except nx.NetworkXNoPath:
        return 0

# functions for computing modified version of local efficiency of a network from T. Xe et al. (accounting for inifinte distance in the case of diconnected graphs)
def local_efficiency_node(G, node):
    """
    ## Doctests
    >>> G = nx.Graph()
    >>> G.add_node(1)
    >>> local_efficiency_node(G, 1)
    0

    >>> G = nx.Graph()
    >>> G.add_edge(1, 2)
    >>> local_efficiency_node(G, 1)
    0

    >>> G = nx.Graph()
    >>> G.add_edge(1, 2)
    >>> G.add_edge(1, 3)
    >>> G.add_edge(2, 3)
    >>> local_efficiency_node(G, 1)
    1.0
    
    >>> G = nx.erdos_renyi_graph(n=10, p=0.5, seed=42)
    >>> local_efficiency_node(G, 1)
    0.75
    """
    neighbors = list(G.neighbors(node))
    if len(neighbors) < 2:  # For isolated node or leaf node
        return 0
    subgraph = G.subgraph(neighbors)
    efficiency_sum = sum(path_length(subgraph, j, h) if j != h else 0 
                         for j in neighbors for h in neighbors)
    return efficiency_sum / (len(neighbors) * (len(neighbors) - 1))

# Uses local_efficiency_node
def local_efficiency_network(G):
    local_efficiencies = [local_efficiency_node(G, node) for node in G.nodes]
    return np.mean(local_efficiencies)

# Uses local_efficiency_network
def local_eff(data, num_subjects=None):
    """
    Summary: computes the local efficiencies of the set of graphs given

    Args: 
    data (np.array): set of graphs with shape -> (num_atlas_reg)x(num_atlas_reg)x(num_subjects)
    num_subjects: in case not all of the thresholded graphs and null graphs have been computed

    Returns: 
    float: local efficiencies of the set of graphs given
    """
    if num_subjects == None: 
        num_subjects = data.shape[2]
    return np.array([local_efficiency_network(nx.from_numpy_array(data[:, :, i])) for i in range(num_subjects)])