"""
File that contains the class that generates causal graphs.
"""
import numpy as np
import igraph as ig


def generate_synthetic_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == "FC":
        # Fully connected DAG
        M = np.zeros((d, d))
        M[np.triu_indices(d, k=1)] = 1
        B = _random_permutation(M)

    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def generate_random_dag(
    num_variables: int, expected_node_degree: int
) -> np.ndarray:
    """Generate a random DAG.

    This is done by generating a topoligically sorted DAG with an expected node
    degree.

    An Erdos Renyi graph has on average NC2 * p edges

    Note: The DAG returned is topologically sorted. It needs to be permuted.

    Args:
    ----------
    num_variables : int
        The number of variables in the DAG.

    expected_node_degree : int
        The expected node degree of the DAG.

    Returns:
    ----------
    dag : np.ndarray
        The adjacency matrix of the DAG.
    """
    # Generate DAG
    dag = np.zeros((num_variables, num_variables))

    # Prob of edge
    prob_edge = 2 * expected_node_degree / (num_variables - 1)
    if prob_edge > 1:
        raise ValueError(
            "The expected node degree is too high for the number of variables."
            f" Node degree: {expected_node_degree}, prob edge: {prob_edge} "
        )

    # Generate an upper traingular matrix.
    # A_ij where variable i causes variable j.
    for i in range(num_variables):
        for j in range(i + 1, num_variables):
            dag[i, j] = np.random.choice([0, 1], p=[1 - prob_edge, prob_edge])
    return dag


if __name__ == "__main__":
    # Check degree of DAG
    # dag = generate_random_dag(2, )
    # print(np.mean(np.sum(dag, axis=1)))
    # Get 2 node DAG
    for i in range(10):
        dag = generate_random_dag(3, 0.5)
        print(dag)
