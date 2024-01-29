# -- coding: utf-8 --
# @Time    : 2023/7/31 0031 9:52
# @Author  : TangKai
# @Team    : ZheChengData


import networkx as nx


def judge_ring(g=None):
    """判断"""
    if isinstance(g, nx.Graph):
        node_degrees_set = {g.degree[node] for node in g.nodes}
        if node_degrees_set == {2}:
            return True
        else:
            return False
    elif isinstance(g, nx.DiGraph):
        in_degree_set = {g.in_degree[node] for node in g.nodes}
        out_degree_set = {g.out_degree[node] for node in g.nodes}
        if in_degree_set == {1} and out_degree_set == {1}:
            return True
        else:
            return False
    else:
        raise TypeError(r'请传入一个nx图对象...')
    pass


if __name__ == '__main__':
    x = nx.Graph()
    x.add_edges_from([[1, 3], [3, 4], [4, 1], [3, 12]])
    print(judge_ring(x))

    # x = nx.DiGraph()
    # x.add_edges_from([[1, 3], [3, 4], [4, 1]])

    print(judge_ring(x))

    print(list(x.adj[3]))
