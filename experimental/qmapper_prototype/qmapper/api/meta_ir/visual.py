import pydot
from .definitions import MetaNode
from typing import List, Dict

def visual_meta(nodes: List[MetaNode]):
    g = pydot.Dot(graph_type='graph')
    meta_node_map: Dict[MetaNode, pydot.Node] = {}
    static_id = 0
    def get_dot_node(node: MetaNode):
        nonlocal meta_node_map
        if node in meta_node_map:
            return meta_node_map[node]
        else:
            meta_node_map[node] = pydot.Node(f"{node.name}-{node.uuid}")
        return meta_node_map[node]
    for node in nodes:
        g.add_node(get_dot_node(node))
        for input in node.inputs:
            if input.up_node != node:
                input_up_dot_node = get_dot_node(input.up_node)
                this_node = get_dot_node(node)
                g.add_edge(
                    pydot.Edge(input_up_dot_node.get_name(),
                                this_node.get_name()))
    g.write_svg('output.svg')
    g.write_pdf('output.pdf')