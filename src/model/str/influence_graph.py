"""The Influence Graph. Responsible for creating training examples for generative training of WIQA

  V       Z
  |     /
  -   + 
  | /
  X       U
  | \     |
  -   +   -
  |     \ |
  W       Y
  | \   / |
  -   +   -
  | /   \ |
  L       M

Usage:
    influence_graph.py [options]

Options:
    --add_start_end=<BOOL>                  Whether to add examples that use start and end node
    --add_reversed=<BOOL>                   Should the nodes be reversed and added as training examples?
    --add_entire_path=<BOOL>                Add the entire path
    --add_paragraph=<BOOL>                  Should the paragraph be added to the src sentence?
    --para-at-end=<BOOL>                    Should the paragraph be added at the end?
    --reln-in-middle=<BOOL>                 If true, the relation is added to the middle (PARA-RELN-NODE)
    --generation-type=<str>                 The type of generation to be used. Should be one of: natural, question, simple
    --path_to_qa=<str>                      Path to the directory that has qa files. This is used to create the train/test/dev splits
    --outpath=<str>                         Path where the output should be written
    --path_to_influence_graphs=<str>        Path to the influence graphs jsonl file
"""

import json
from collections import Counter, namedtuple
import json
import random
import re
from collections import defaultdict
import sys
import pydot
import itertools
import pandas as pd
from docopt import docopt
from typing import List, Tuple, Set
from IPython.display import Image, display

WhatIfExample = namedtuple(
    "WhatIfExample", ["src", "reln", "dest", "path_length", "src_type", "dest_type"])
WhatIfPath = namedtuple("WhatIfPath", ["nodes", "edges", "length"])

random.seed(42)


class Rels:
    old_special_tokens = {
        "helps": "helps",
        "hurts": "hurts",
        "helped_by": "RELATION-HELPED-BY",
        "hurt_by": "RELATION-HURT-BY",
        "1": "1-HOP",
        "2": "2-HOP",
        "3": "3-HOP",
        "para_sep": "<PARA>",
        "node_sep": "<NODE>"
    }

    special_tokens = {
        "helps": "helps",
        "hurts": "hurts",
        "helped_by": "is helped by",
        "hurt_by": "is hurt by",
        "1": "1 hop",
        "2": "2 hop",
        "3": "3 hop"
    }


class InfluenceGraph:

    nodes = ["V", "Z", "X", "U", "W", "Y",
             "para_outcome_accelerate", "para_outcome_decelerate"]

    def __init__(self, graph: dict, max_para_tokens=75):
        """Creates an influence graph given a dict representation and the max
        number of tokens to be retained in the paragraph.

        Arguments:
            graph_dict {[dict]} -- [the graph representation]
            max_para_tokens {[int]} -- [the max number of tokens to be retained in an influence graph]
        """
        super().__init__()
        self.nodes_dict = {}
        self.graph = graph
        self.prompt = graph["prompt"]
        for node in InfluenceGraph.nodes:
            #  each node is supposed to have a list of sentences. convert to a list if that's not the case.
            if isinstance(graph[node], str):
                graph[node] = [graph[node]]
            # self.nodes_dict[node] = [random.choice(graph[node])] if len(graph[node]) > 0 else graph[node]
            self.nodes_dict[node] = graph[node]
        #  shorter keys for easy ref
        self.nodes_dict["dec"] = self.nodes_dict.pop("para_outcome_decelerate")
        self.nodes_dict["acc"] = self.nodes_dict.pop("para_outcome_accelerate")

        self.para_id = graph["para_id"]
        self.graph_id = graph["graph_id"]

        self.paragraph = " ".join(graph["paragraph"].split()[
                                  :max_para_tokens]).replace("\n", "")

        self.NEG = Rels.special_tokens["hurts"]
        self.POS = Rels.special_tokens["helps"]

        self.edges = {
            "V": [("X", self.NEG)],
            "Z": [("X", self.POS)],
            "X": [("Y", self.POS), ("W", self.NEG)],
            "U": [("Y", self.NEG)]
        }

        if graph["Y_affects_outcome"] == "more":
            self.edges["Y"] = [("acc", self.POS), ("dec", self.NEG)]
            self.edges["W"] = [("acc", self.NEG), ("dec", self.POS)]
        else:
            self.edges["Y"] = [("acc", self.NEG), ("dec", self.POS)]
            self.edges["W"] = [("acc", self.POS), ("dec", self.NEG)]

        self.level = {
            "V": 0, "Z": 0, "X": 1, "U": 1, "W": 2, "Y": 2, "acc": 3, "dec": 3
        }

        # create reverse edges
        self.rev_edges = defaultdict(list)
        for node, links in self.edges.items():
            for (nbr, edge_type) in links:
                self.rev_edges[nbr].append((node, "X-" + edge_type))

        # create all edges
        self.all_edges = {}
        for node in set(list(self.edges.keys()) + list(self.rev_edges.keys())):
            if node in self.edges:
                self.all_edges[node] = self.rev_edges[node] + self.edges[node]
            else:
                self.all_edges[node] = self.rev_edges[node]

        for k, v in self.all_edges.items():
            self.all_edges[k] = sorted(v, key=lambda x: x[1])

        # create text to node

        self.text_to_node = {}
        for node, node_values in self.nodes_dict.items():
            for node_value in node_values:
                self.text_to_node[node_value] = node

        # create node to context
        self.node_context = defaultdict(str)  # captures the context of a node
        for node in self.nodes_dict:
            for (nbr, reln) in self.all_edges[node]:
                nbr_text = reln + " " + " ".join(self.nodes_dict[nbr])
                self.node_context[node] += " " + nbr_text

    def get_pair_context(self, node_1_text: str, node_2_text: str) -> dict:
        """Returns the context for a given node pair. The context is returned after 
        excluding the given nodes

        Arguments:
            node_1_text {str} -- [description]
            node_2_text {str} -- [description]
        Returns:
            dict -- [description]
        """
        # convert text to node
        node1 = self.get_text_to_node(node_1_text)
        node2 = self.get_text_to_node(node_2_text)

        # get neighbors
        excluded_nodes = {node1, node2}
        node1_neighbors = self.get_neighbors(
            node1, excluded_nodes=excluded_nodes)
        node2_neighbors = self.get_neighbors(
            node2, excluded_nodes=excluded_nodes)

        # get context
        return {
            "node1_context": self.make_context_string(node1_neighbors),
            "node2_context": self.make_context_string(node2_neighbors),
        }

    def get_text_to_node(self, node_text: str) -> str:
        node = None
        if node_text in self.text_to_node:
            node = self.text_to_node[node_text]
        elif node_text.lower() in self.text_to_node:
            node = self.text_to_node[node_text.lower()]
        elif node_text.lower()[:-1] in self.text_to_node:
            node = self.text_to_node[node_text.lower()[:-1]]
        return node

    def get_neighbors(self, node: str, excluded_nodes: Set[str]) -> List[Tuple[str, str]]:
        if node is None:
            return None
        all_nbrs = self.all_edges[node]
        return [(nbr, reln) for (nbr, reln) in all_nbrs if nbr not in excluded_nodes]

    def make_context_string(self, neighbors: List[Tuple[str, str]]) -> str:
        if neighbors is None:
            return ""
        nbrs = sorted(neighbors, key=lambda x: x[1])
        #  ctx = " ".join([f"{x[1]} {'|||'.join(self.nodes_dict[x[0]])}" for x in neighbors])
        ctx = ""
        for (nbr, reln) in neighbors:
            nbr_text = self.nodes_dict[nbr]
            if len(nbr_text) == 0:
                continue
            # ctx += " " + f"{reln} {nbr_text[0]}"
            ctx += " " + f"{nbr_text[0]}"
        return ctx.strip()

    def get_examples(self, add_start_end: bool, add_reversed: bool, add_entire_path: bool):
        """Flattens the given influence graph

        Returns:
            [type] -- [description]
        """
        examples = set()
        for src in self.edges:
            paths = self.get_paths(src)
            if add_entire_path:
                examples.update(self.make_whatif_entire_path(paths))
            if add_start_end:
                examples.update(self.make_whatif_start_finish(
                    paths, add_reversed=add_reversed))
        return examples

    def __str__(self, merge_nodes_op="join") -> str:
        def _wrap_quotes(obj: str) -> str:
            return f"\"{obj}\""

        def join_merge(nodes: List) -> str:
            return " [OR] ".join([node.strip() for node in nodes])

        def random_merge(nodes: List) -> str:
            return random.choice(nodes)

        def first_merge(nodes: List) -> str:
            return nodes[0]

        def last_merge(nodes: List) -> str:
            return nodes[-1]

        merge_func = {
            "join": join_merge,
            "random": random_merge,
            "first": first_merge,
            "last": last_merge,
        }

        merge_nodes_op = merge_func[merge_nodes_op]

        hurts = ' - hurts -'
        helps = ' - helps -'
        v = f"V : {merge_nodes_op(self.nodes_dict['V'])}" if len(
            self.nodes_dict['V']) > 0 else ''
        z = f"Z : {merge_nodes_op(self.nodes_dict['Z'])}" if len(
            self.nodes_dict['Z']) > 0 else ''
        x = f"X : {merge_nodes_op(self.nodes_dict['X'])}" if len(
            self.nodes_dict['X']) > 0 else ''
        u = f"U : {merge_nodes_op(self.nodes_dict['U'])}" if len(
            self.nodes_dict['U']) > 0 else ''
        w = f"W : {merge_nodes_op(self.nodes_dict['W'])}" if len(
            self.nodes_dict['W']) > 0 else ''
        y = f"Y : {merge_nodes_op(self.nodes_dict['Y'])}" if len(
            self.nodes_dict['Y']) > 0 else ''
        l = f"L : {merge_nodes_op(self.nodes_dict['dec'])}" if len(
            self.nodes_dict['dec']) > 0 else ''
        m = f"M : {merge_nodes_op(self.nodes_dict['acc'])}" if len(
            self.nodes_dict['acc']) > 0 else ''
        repr = ""
        repr += f"{v} {hurts} {x} | {z} {helps} {x}"
        repr += " | " + f"{x} {hurts} {w} | {x} {helps} {y} | {u} {hurts} {y}"
        if self.graph["Y_affects_outcome"] == "more":
            repr += " | " + f"{w} {helps} {l} | {w} {hurts} {m}"
            repr += " | " + f"{y} {helps} {m} | {y} {hurts} {l}"
        else:
            repr += " | " + f"{w} {hurts} {l} | {w} {helps} {m}"
            repr += " | " + f"{y} {hurts} {m} | {y} {helps} {l}"
        return repr.strip()


    def nodes_to_string(self, merge_nodes_op="join") -> str:
        def _wrap_quotes(obj: str) -> str:
            return f"\"{obj}\""

        def join_merge(nodes: List) -> str:
            return " [OR] ".join([node.strip() for node in nodes])

        def random_merge(nodes: List) -> str:
            nodes = [node.strip() for node in nodes[0].split("[OR]")]
            return random.choice(nodes)

        def first_merge(nodes: List) -> str:
            nodes = [node.strip() for node in nodes[0].split("[OR]")]
            return nodes[0]

        def last_merge(nodes: List) -> str:
            nodes = [node.strip() for node in nodes[0].split("[OR]")]
            return nodes[-1]

        def _prune_tokens(node_label: str, max_num_toks: int = 35):
            node_label_toks = node_label.split()
            if len(node_label_toks) >= max_num_toks:
                return " ".join(node_label_toks[:max_num_toks])
            return " ".join(node_label_toks)

        merge_func = {
            "join": join_merge,
            "random": random_merge,
            "first": first_merge,
            "last": last_merge,
        }

        merge_nodes_op = merge_func[merge_nodes_op]

        v = f"V : {merge_nodes_op(self.nodes_dict['V'])}" if len(
            self.nodes_dict['V']) > 0 else ''
        z = f"Z : {merge_nodes_op(self.nodes_dict['Z'])}" if len(
            self.nodes_dict['Z']) > 0 else ''
        x = f"X : {merge_nodes_op(self.nodes_dict['X'])}" if len(
            self.nodes_dict['X']) > 0 else ''
        u = f"U : {merge_nodes_op(self.nodes_dict['U'])}" if len(
            self.nodes_dict['U']) > 0 else ''
        w = f"W : {merge_nodes_op(self.nodes_dict['W'])}" if len(
            self.nodes_dict['W']) > 0 else ''
        y = f"Y : {merge_nodes_op(self.nodes_dict['Y'])}" if len(
            self.nodes_dict['Y']) > 0 else ''
        l = f"L : {merge_nodes_op(self.nodes_dict['dec'])}" if len(
            self.nodes_dict['dec']) > 0 else ''
        m = f"M : {merge_nodes_op(self.nodes_dict['acc'])}" if len(
            self.nodes_dict['acc']) > 0 else ''
        
        node_labels = [_prune_tokens(node_label) for node_label in [v, z, x, u, w, y, l, m]]
        
        repr = " ;; ".join(node_labels)
        return repr.strip()


    def _get_all_graphs(self, format: str) -> List[str]:
        """Returns *all* the possible graphs that can be generated by considering all
        the combinations of nodess
        Args:
            format ([type]): [either "dot" or "str"]
        Returns:
            A list of tuples, where the first element is the x node and the second is the entire graph represented in format.
        """
        res = []
        transform_func = self._to_dot_given_nodes if format == "dot" else self._to_str_given_nodes
        for (v, z, x, u, w, y, l, m) in itertools.product(self.nodes_dict['V'], self.nodes_dict['Z'], self.nodes_dict['X'], self.nodes_dict['U'], self.nodes_dict['W'], self.nodes_dict['Y'], self.nodes_dict['dec'], self.nodes_dict['acc']):
            res.append(
                (x, transform_func(v=v, z=z, x=x, u=u, w=w, y=y, l=l, m=m)))
        return res

    def _to_str_given_nodes(self, v: str, z: str, x: str, u: str, w: str, y: str, l: str, m: str) -> str:
        hurts = ' - hurts -'
        helps = ' - helps -'
        repr = ""
        repr += f"{v} {hurts} {x} | {z} {helps} {x}"
        repr += " | " + f"{x} {hurts} {w} | {x} {helps} {y} | {u} {hurts} {y}"
        if self.graph["Y_affects_outcome"] == "more":
            repr += " | " + f"{w} {helps} {l} | {w} {hurts} {m}"
            repr += " | " + f"{y} {helps} {m} | {y} {hurts} {l}"
        else:
            repr += " | " + f"{w} {hurts} {l} | {w} {helps} {m}"
            repr += " | " + f"{y} {hurts} {m} | {y} {helps} {l}"
        return repr.strip()

    def _to_dot_given_nodes(self, v: str, z: str, x: str, u: str, w: str, y: str, l: str, m: str) -> str:
        dot_str = "strict digraph { "
        hurts = '[label=hurts]'
        helps = '[label=helps]'
        arrow = '->'
        dot_str += f"{v} {arrow} {x} {hurts};{z} {arrow} {x} {helps};"
        dot_str += f"{x} {arrow} {w} {hurts};{x} {arrow} {y} {helps}; {u} {arrow} {y} {hurts};"
        if self.graph["Y_affects_outcome"] == "more":
            dot_str += f"{w} {arrow} {l} {helps}; {w} {arrow} {m} {hurts};"
            dot_str += f"{y} {arrow} {m} {helps}; {y} {arrow} {l} {hurts};"
        else:
            dot_str += f"{w} {arrow} {l} {hurts}; {w} {arrow} {m} {helps};"
            dot_str += f"{y} {arrow} {m}  {hurts}; {y} {arrow} {l} {helps};"
        dot_str += " }"
        return dot_str.strip()

    @staticmethod
    def from_dot_old(dot_str: str, graph_id: str = ""):
        nodes, edge_dict = InfluenceGraph.parse_dot_influence_graph(
            dot_str, None)
        graph_dict = {}
        for node in nodes:
            graph_dict[node] = [nodes[node]]
        graph_dict["para_outcome_accelerate"] = graph_dict.pop("M")
        graph_dict["para_outcome_decelerate"] = graph_dict.pop("L")
        graph_dict["graph_id"] = graph_id
        graph_dict["para_id"] = graph_id
        graph_dict["paragraph"] = ""
        graph_dict["Y_affects_outcome"] = "more" if edge_dict[(
            "Y", "M")] == "helps" else "less"
        graph_dict["prompt"] = ""
        return InfluenceGraph(graph=graph_dict)

    @staticmethod
    def from_dot(dot_str: str, graph_id: str = ""):
        nodes = InfluenceGraph.parse_dot_str(dot_str, None)
        graph_dict = {}
        for node in ["W", "U", "Z", "V"]:
            graph_dict[node] = [nodes[node]]
        graph_dict["X"] = nodes["X"]
        graph_dict["Y"] = nodes["Y"]
        graph_dict["para_outcome_accelerate"] = nodes["M"]
        graph_dict["para_outcome_decelerate"] = nodes["L"]
        graph_dict["graph_id"] = graph_id
        graph_dict["para_id"] = graph_id
        graph_dict["paragraph"] = ""
        graph_dict["Y_affects_outcome"] = nodes["affects"]
        graph_dict["prompt"] = ""
        return InfluenceGraph(graph=graph_dict)

    @staticmethod
    def parse_dot_influence_graph(igraph_dot_str: str, suffix: str) -> Tuple[dict, dict]:
        """
        Given an influence graph in the DOT format, and a suffix, returns two dictionaries: 
        1) Nodes: mapping from the node label to node text
        2) Edges: Mapping from node label pair to the edge type.
        E.g. output:
        ({'V_suffix': 'less magma is formed [OR] less magma hardens',
        'X_suffix': 'more magma is released',
        'Z_suffix': 'more magma rises [OR] more lava rises',
        'W_suffix': 'less lava reaches the surface [OR] less magma hardens',
        'Y_suffix': 'more lava rises to the surface',
        'U_suffix': 'there is less magma [OR] there is less lava',
        'L_suffix': 'LESS igneous rock forming',
        'M_suffix': 'MORE igneous rock forming'},
        {('V_suffix', 'X_suffix'): 'hurts',
        ('Z_suffix', 'X_suffix'): 'helps',
        ('X_suffix', 'W_suffix'): 'hurts',
        ('X_suffix', 'Y_suffix'): 'helps',
        ('U_suffix', 'Y_suffix'): 'hurts',
        ('W_suffix', 'L_suffix'): 'helps',
        ('W_suffix', 'M_suffix'): 'hurts',
        ('Y_suffix', 'M_suffix'): 'helps',
        ('Y_suffix', 'L_suffix'): 'hurts'})

        Currently, the script attempts to fix node labels, but makes no further cosmetic changes.
        """
        def _clean_node(node_str):
            if len(node_str.strip()) < 3:
                return "N/A", "N/A"
            node_fields = node_str.split(":")
            if len(node_fields) > 2:
                node_label, node_txt, _ = node_fields
            else:
                node_label, node_txt = node_fields
            return node_label.replace("\"", "").strip(), node_txt.replace("\"", "").strip()

        def _infer_node_label(a, b, rel):
            fwd_nbrs = {
                ("V", "hurts"): "X",
                ("Z", "helps"): "X",
                ("X", "helps"): "W",
                ("U", "hurts"): "Y",
                ("W", "hurts"): "L",
                ("W", "helps"): "L",
                ("W", "hurts"): "M",
                ("W", "helps"): "M",
                ("Y", "hurts"): "L",
                ("Y", "helps"): "L",
                ("Y", "hurts"): "M",
                ("Y", "helps"): "M",
            }
            rev_nbrs = {
                (tgt, rel): src for ((src, rel), tgt) in fwd_nbrs.items()
            }

            if b == "N/A":
                return a, fwd_nbrs[(a, rel)]
            elif a == "N/A":
                return rev_nbrs[(b, rel)], b
            else:
                return "N/A", "N/A"
        if suffix is None:
            suffix = ""
        else:
            suffix = f"_{suffix}"
        igraph_dot_str = igraph_dot_str.replace("\'", '')
        igraph = pydot.graph_from_dot_data(igraph_dot_str)[0]
        nodes = dict()
        edge_dict = dict()
        for edge in igraph.get_edge_list():
            src_label, src_text = _clean_node(edge.get_source())
            tgt_label, tgt_text = _clean_node(edge.get_destination())
            if src_label == "N/A" or tgt_label == "N/A":
                src_label, tgt_label = _infer_node_label(
                    src_label, tgt_label, rel=edge.get_label())
            nodes[f"{src_label}{suffix}"] = src_text
            nodes[f"{tgt_label}{suffix}"] = tgt_text
            edge_dict[(f"{src_label}{suffix}",
                       f"{tgt_label}{suffix}")] = edge.get_label()
        return nodes, edge_dict

    def to_dot(self, merge_nodes_op: str = "join") -> str:
        """Returns a dotcode representation for a graph.
        merge_nodes_op: either "join" or "random". If join, all the nodes are concatenated
        with a token [OR]. Otherwise, they are selected at random
        Returns:
            str: Graph representation in DOT.
        """
        def _wrap_quotes(obj: str) -> str:
            return f"\"{obj}\""

        def join_merge(nodes: List) -> str:
            return " [OR] ".join([node.strip() for node in nodes])

        def random_merge(nodes: List) -> str:
            return random.choice(nodes)

        merge_nodes_op = (join_merge if merge_nodes_op ==
                          "join" else random_merge)
        v = _wrap_quotes(f"V : {merge_nodes_op(self.nodes_dict['V'])}" if len(
            self.nodes_dict['V']) > 0 else '')
        z = _wrap_quotes(f"Z : {merge_nodes_op(self.nodes_dict['Z'])}" if len(
            self.nodes_dict['Z']) > 0 else '')
        x = _wrap_quotes(f"X : {merge_nodes_op(self.nodes_dict['X'])}" if len(
            self.nodes_dict['X']) > 0 else '')
        u = _wrap_quotes(f"U : {merge_nodes_op(self.nodes_dict['U'])}" if len(
            self.nodes_dict['U']) > 0 else '')
        w = _wrap_quotes(f"W : {merge_nodes_op(self.nodes_dict['W'])}" if len(
            self.nodes_dict['W']) > 0 else '')
        y = _wrap_quotes(f"Y : {merge_nodes_op(self.nodes_dict['Y'])}" if len(
            self.nodes_dict['Y']) > 0 else '')
        l = _wrap_quotes(f"L : {merge_nodes_op(self.nodes_dict['dec'])}" if len(
            self.nodes_dict['dec']) > 0 else '')
        m = _wrap_quotes(f"M : {merge_nodes_op(self.nodes_dict['acc'])}" if len(
            self.nodes_dict['acc']) > 0 else '')

        dot_str = "strict digraph { "
        hurts = '[label=hurts]'
        helps = '[label=helps]'
        arrow = '->'
        dot_str += f"{v} {arrow} {x} {hurts}; {z} {arrow} {x} {helps}; "
        dot_str += f"{x} {arrow} {w} {hurts}; {x} {arrow} {y} {helps}; {u} {arrow} {y} {hurts}; "
        if self.graph["Y_affects_outcome"] == "more":
            dot_str += f"{w} {arrow} {l} {helps}; {w} {arrow} {m} {hurts}; "
            dot_str += f"{y} {arrow} {m} {helps}; {y} {arrow} {l} {hurts}; "
        else:
            dot_str += f"{w} {arrow} {l} {hurts}; {w} {arrow} {m} {helps}; "
            dot_str += f"{y} {arrow} {m}  {hurts}; {y} {arrow} {l} {helps}; "
        dot_str += "}"
        dot_str = " ".join(dot_str.split()).replace("?", "")
        return dot_str.strip()

    def to_pydot(self):
        return pydot.graph_from_dot_data(self.to_dot())[0]

    def to_png(self):
        return self.to_pydot().create_png()

    def to_svg(self):
        return self.to_pydot().create_svg()

    def to_jupyter_img(self, width: int = 1600, height: int = 1600, retina: bool = False):
        plt = Image(self.to_png(), width=width, height=height, retina=retina)
        return display(plt)

    def to_ascii_drawing(self, merge_nodes_op="join"):
        """Dumps an ASCII drawing on the terminal
        """
        from textwrap import wrap

        def _wrap_quotes(obj: str) -> str:
            return f"\"{obj}\""

        def join_merge(nodes: List) -> str:
            return " [OR] ".join([node.strip() for node in nodes])

        def random_merge(nodes: List) -> str:
            return random.choice(nodes)

        merge_nodes_op = (join_merge if merge_nodes_op ==
                          "join" else random_merge)
        v = _wrap_quotes(f"V : {merge_nodes_op(self.nodes_dict['V'])}" if len(
            self.nodes_dict['V']) > 0 else '')
        z = _wrap_quotes(f"Z : {merge_nodes_op(self.nodes_dict['Z'])}" if len(
            self.nodes_dict['Z']) > 0 else '')
        x = _wrap_quotes(f"X : {merge_nodes_op(self.nodes_dict['X'])}" if len(
            self.nodes_dict['X']) > 0 else '')
        u = _wrap_quotes(f"U : {merge_nodes_op(self.nodes_dict['U'])}" if len(
            self.nodes_dict['U']) > 0 else '')
        w = _wrap_quotes(f"W : {merge_nodes_op(self.nodes_dict['W'])}" if len(
            self.nodes_dict['W']) > 0 else '')
        y = _wrap_quotes(f"Y : {merge_nodes_op(self.nodes_dict['Y'])}" if len(
            self.nodes_dict['Y']) > 0 else '')
        l = _wrap_quotes(f"L : {merge_nodes_op(self.nodes_dict['dec'])}" if len(
            self.nodes_dict['dec']) > 0 else '')
        m = _wrap_quotes(f"M : {merge_nodes_op(self.nodes_dict['acc'])}" if len(
            self.nodes_dict['acc']) > 0 else '')

        r1, r2, r3 = (
            "+", "-", "+") if self.graph["Y_affects_outcome"] == "more" else ("-", "+", "-")
        repr = """
        {0}     {1}
        |     /
        -   + 
        | /
        {2}    {3}
        | \     |
        -   +   -
        |     \ |
        {4}    {5}
        | \   / |
        {8}   {9}   {10}
        | /   \ |
        {6}    {7}
        """.format(v, z, x, u, w, y, l, m, r1, r2, r3)
        return repr

    def get_paths(self, source: str) -> List[Tuple[List, List]]:
        """Runs bfs from the given source node, and returns all the paths

        Returns:
            List[tuple(List, List)] -- [(Nodes, Relations)]
        """
        queue = [(source, [], [source], 0)]
        visited = {source}
        paths = []
        while queue:
            curr_element, path_reln, path_nodes, path_length = queue.pop(0)
            visited.add(curr_element)
            paths.append(WhatIfPath(nodes=path_nodes,
                                    edges=path_reln, length=path_length))
            if curr_element not in self.edges:  # no more traversal possible
                continue
            for (nbr, reln) in self.edges[curr_element]:
                if nbr in visited:
                    continue
                queue.append(
                    (nbr, path_reln + [reln], path_nodes + [nbr], path_length + 1))
        return paths

    def make_whatif_start_finish(self, paths: List[Tuple], add_reversed=False):
        """Given a set of reachable nodes, creates an example using only the starting 
        and the ending nodes. Also captures the reverse relations if indicated
        with the flag.

        Arguments:
            reachable_nodes {[]} -- [A list of paths]
        """
        def get_example_from_path(path: WhatIfPath) -> Set[WhatIfExample]:
            res = set()
            if len(path.nodes) == 1:
                return res
            src = path.nodes[0]
            dest = path.nodes[-1]
            reln = Rels.special_tokens["helps"] if Counter(
                path.edges)[Rels.special_tokens["hurts"]] % 2 == 0 else Rels.special_tokens["hurts"]
            #  even number of hurts is helps
            reverse_reln = Rels.special_tokens["helped_by"] if reln == Rels.special_tokens[
                "helps"] else Rels.special_tokens["hurt_by"]
            for src_sent in self.nodes_dict[src]:
                for dest_sent in self.nodes_dict[dest]:
                    res.add(WhatIfExample(src=src_sent, dest=dest_sent, reln=reln, path_length=Rels.special_tokens[str(path.length)],
                                          src_type=src, dest_type=dest))
                    if add_reversed:
                        res.add(WhatIfExample(src=dest_sent, dest=src_sent, 
                                              src_type=dest, dest_type=src,
                                            reln=reverse_reln, path_length=Rels.special_tokens[str(path.length)]))
            return res

        res = set()
        for path in paths:
            res.update(get_example_from_path(path))
        return res

    def make_whatif_entire_path(self, paths: List[Tuple]):
        """Given a set of reachable nodes, creates an example using only the starting 
        and the ending nodes. Also captures the reverse relations if indicated
        with the flag.

        Arguments:
            reachable_nodes {[]} -- [A list of paths]
        """
        def get_example_from_path(path: WhatIfPath) -> Set[WhatIfExample]:
            res = set()
            if len(path.nodes) == 1:
                return res

            path_nodes_edges = [self.nodes_dict[path.nodes[0]]]
            for i, node in enumerate(path.nodes[1:-1]):
                path_nodes_edges.append([path.edges[i]])
                path_nodes_edges.append(self.nodes_dict[node])

            dest = path.nodes[-1]
            reln = path.edges[-1]
            for source_node in itertools.product(*path_nodes_edges):
                for dest_sent in self.nodes_dict[dest]:
                    src_sent = " ".join(list(source_node))
                    res.add(WhatIfExample(src=src_sent,
                                          dest=dest_sent, reln=reln))
            return res

        res = set()
        for path in paths:
            res.update(get_example_from_path(path))
        return res

    @staticmethod
    def from_str_old(igraph_str: str, graph_id: str = "", para_id: str = "",
                     paragraph: str = "", prompt: str = ""):
        """Makes an influence graph from the string representation.
        Deprecated.
        Args:
            igraph_str ([str]): InfluenceGraph
        """
        def _get_node(node):
            node_str = re.search(f"{node} : .*? [\|-]", igraph_str).group(0)
            return node_str[3:-2].strip()

        if igraph_str[0] != "V":  # for some reason, this is not happening
            igraph_str = f"V : {igraph_str}"

        graph = {}
        for node in ["V", "Z", "X", "U", "Y", "W", "L", "M"]:
            try:
                graph[node] = _get_node(node)
            except:
                graph[node] = ""
        if re.search(r"\| Y : .*? - helps - M.*?", igraph_str) is not None:
            graph["Y_affects_outcome"] = "more"
        else:
            graph["Y_affects_outcome"] = "less"
        graph["para_outcome_accelerate"] = graph.pop("M")
        graph["para_outcome_decelerate"] = graph.pop("L")
        graph["Y_is_outcome"] = ""  # just to have all the keys
        graph["paragraph"] = paragraph
        graph["para_id"] = para_id
        graph["graph_id"] = graph_id
        graph["prompt"] = prompt
        return InfluenceGraph(graph=graph)

    @staticmethod
    def from_str(graph_str, graph_id):
        graph_dict = {}
        graph_dict["Y_affects_outcome"] = "more" if "affects outcome = more" in graph_str else "less"
        graph_str = " ;; ".join(graph_str.split(";;")[:-1])
        for node in ["Z", "V", "X", "U", "W", "Y", "L"]:
            try:
                graph_dict[node] = re.search(
                    f"{node} : .*? ;;", graph_str).group(0).split(":")[1].replace(";;", "").strip()
            except:
                graph_dict[node] = ""
        try:
            graph_dict["para_outcome_accelerate"] = re.search(
                f"M : .*?$", graph_str).group(0).split(":")[1].strip()
        except:
            graph_dict["para_outcome_accelerate"] = ""
        graph_dict["para_outcome_decelerate"] = graph_dict.pop("L")
        graph_dict["graph_id"] = graph_id
        graph_dict["para_id"] = graph_id
        graph_dict["paragraph"] = graph_id
        graph_dict["prompt"] = graph_id
        return InfluenceGraph(graph_dict)

    @staticmethod
    def parse_dot_str(dot_graph_str, prefix_node_label: bool = False):

        def _get_rel(dot_graph_str):
            # broadly: extracts Y -> L OR Y -> M strings from the graph, and looks at the corresponding relation.
            try:
                y_to_l = re.search(
                    "} .*?L.*?Y", dot_graph_str[::-1]).group(0)[::-1]
            except:
                y_to_l = ""
            try:
                y_to_m = re.search(
                    "} .*?M.*?Y", dot_graph_str[::-1]).group(0)[::-1]
            except:
                y_to_m = ""
            if len(y_to_l) < len(y_to_m):
                if "[label=helps]" in y_to_l:
                    return "less"
                else:
                    return "more"
            else:
                if "[label=helps]" in y_to_m:
                    return "more"
                else:
                    return "less"

        nodes = {}
        for node in ["Z", "V", "U"]:
            try:
                nodes[node] = f"{node} = " if prefix_node_label else "" + re.search(
                    f"{node} : .*?->", dot_graph_str).group(0)[3:-2].strip().replace("\"", "")
            except:
                nodes[node] = f"{node} = None" if prefix_node_label else "None"

        for node in ["X", "Y", "W"]:
            try:
                nodes[node] = f"{node} = " if prefix_node_label else "" + re.search(
                    f"{node} : .*?\\[label", dot_graph_str).group(0)[3:-6].strip().replace("\"", "")
            except:
                nodes[node] = f"{node} = None" if prefix_node_label else "None"

        for node in ["L", "M"]:
            try:
                nodes[node] = f"{node} = " if prefix_node_label else "" + re.search(
                    f"{node} : .*?\\[", dot_graph_str).group(0)[3:-1].strip().replace("\"", "")
            except:
                nodes[node] = f"{node} = None" if prefix_node_label else "None"

        nodes["affects"] = _get_rel(dot_graph_str)
        return nodes


def main(args):
    splits = read_splits(args["--path_to_qa"])

    graphs = read_graphs(args["--path_to_influence_graphs"])

    outpath = args["--outpath"]

    res = create_dataset(graphs=graphs, splits=splits,
                         add_start_end=args["--add_start_end"] == "True",
                         add_reversed=args["--add_reversed"] == "True",
                         add_entire_path=args["--add_entire_path"] == "True",
                         add_paragraph=args["--add_paragraph"] == "True",
                         para_at_end=args["--para-at-end"] == "True",
                         generation_type=args["--generation-type"],
                         reln_in_middle=args["--reln-in-middle"] == "True")

    res[res["split"] == "train"].to_json(
        outpath + "/train.jsonl", orient="records", lines=True)

    # if args["--add_reversed"] == "True":
    #     res[res["split"] == "test"].to_json(
    #         outpath + "/test_additional.jsonl", orient="records", lines=True)

    #     res[res["split"] == "dev"].to_json(
    #         outpath + "/dev_additional.jsonl", orient="records", lines=True)

    #     no_rev_res = create_dataset(graphs=graphs, splits=splits,
    #                                 add_start_end=args["--add_start_end"] == "True",
    #                                 add_reversed=False,
    #                                 add_entire_path=args["--add_entire_path"] == "True",
    #                                 add_paragraph=args["--add_paragraph"] == "True",
    #                                 para_at_end=args["--para-at-end"] == "True",
    #                                 generation_type=args["--generation-type"],
    #                                 reln_in_middle=args["--reln-in-middle"] == "True")

    #     no_rev_res[no_rev_res["split"] == "test"].to_json(
    #         outpath + "/test.jsonl", orient="records", lines=True)

    #     no_rev_res[no_rev_res["split"] == "dev"].to_json(
    #         outpath + "/dev.jsonl", orient="records", lines=True)
    # else:
    res[res["split"] == "test"].to_json(
        outpath + "/test.jsonl", orient="records", lines=True)

    res[res["split"] == "dev"].to_json(
        outpath + "/dev.jsonl", orient="records", lines=True)
    

def read_splits(pth):
    """Given a file with split, para[TAB]split, reads it and returns a map

    Arguments:
        pth {[type]} -- [description]
    """
    def read_qa_ids(pth):
        ids = []
        with open(pth, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                ids.append(data["metadata"]["para_id"])
        return ids
    res = {}
    for id in read_qa_ids(f"{pth}/qa-train.jsonl"):
        res[id] = "train"

    for id in read_qa_ids(f"{pth}/qa-test.jsonl"):
        res[id] = "test"

    for id in read_qa_ids(f"{pth}/qa-dev.jsonl"):
        res[id] = "dev"
    split_count = {
        "train": 0,
        "test": 0,
        "dev": 0
    }
    for v in res.values():
        split_count[v] += 1
    print("Done reading splits from WIQA datasets")
    num_train_splits = split_count['train']
    num_test_splits = split_count['test']
    num_dev_splits = split_count['dev']

    print(
        f"Train paras = {num_train_splits}, Test paras = {num_test_splits}, Dev paras = {num_dev_splits}")
    return res


def read_graphs(graph_pth: str) -> List[InfluenceGraph]:
    """Reads influence graphs located on disk to a list of InfluenceGraph objects

    Arguments:
        graph_pth {str} -- [description]

    Returns:
        List[InfluenceGraph] -- [description]
    """
    graphs = []
    with open(graph_pth, "r") as f:
        for line in f:
            graphs.append(InfluenceGraph(json.loads(line.strip())))
    return graphs


def create_dataset(graphs: List[InfluenceGraph], splits: dict, add_start_end: bool,
                   add_reversed: bool, add_entire_path: bool, add_paragraph: bool,
                   para_at_end: bool,
                   reln_in_middle: bool,
                   generation_type: str) -> pd.DataFrame:
    """Create training dataset for GPT2

    Arguments:
        graphs {List[InfluenceGraph]} -- [list of influence graphs]
        splits {dict} -- [train/test/val splits]
        add_start_end {bool} -- [Should start/end nodes be added?]
        add_reversed {bool} -- [Should reverse relations be added?]
        add_entire_path {bool} -- [Should entire path be added?]
        add_paragraph {bool} -- [Should paragraph be added?]
        reln-in-middle{bool} -- If true, the relation is added to the middle (PARA-RELN-NODE)
        simple-generation{bool} -- If true, the generations are in simpler format (PARA - RELN - NODE) without any keywords

    Returns:
        pd.DataFrame -- [Dataset in the format expected by GPT2]
    """
    res = []
    for i, graph in enumerate(graphs):
        split = splits[graph.para_id]
        para = graph.paragraph
        prompt = graph.prompt
        refined_node_types = {
            "Z": "C-",
            "V": "C+",
            "X": "S",
            "W": "M-",
            "U": "S-",
            "Y": "M+",
            "para_outcome_decelerate": "H-",
            "acc": "H+",
            "dec": "H-",
            "para_outcome_accelerate": "H+"}
        node_label_to_node_type = {}
        for k, v in graph.graph.items():
            for node_label in v:
                if k in refined_node_types:
                    node_label_to_node_type[node_label] = refined_node_types[k]

        for example in graph.get_examples(add_start_end=add_start_end,
                                          add_reversed=add_reversed, add_entire_path=add_entire_path):

            if generation_type == 'simple' and not add_paragraph:
                src_sent = SourceTextFormatter.natural_question_formatter(
                    relation=example.reln, path_length=example.path_length, src=example.src, para="")
                ans_sent = example.dest
            elif generation_type == 'simple' and add_paragraph:
                #src_sent = f"{example.reln} {example.path_length} {Rels.special_tokens['node_sep']} {example.src} {Rels.special_tokens['para_sep']} {para}"
                src_sent = f"{para} {Rels.special_tokens['para_sep']} {example.reln} {example.path_length} {Rels.special_tokens['node_sep']} {example.src}"
                if para_at_end:
                    src_sent = SourceTextFormatter.relation_node_para_formatter(
                        relation=example.reln, path_length=example.path_length, src=example.src, para=para)
                else:
                    if reln_in_middle:
                        src_sent = SourceTextFormatter.para_relation_node_formatter(
                            relation=example.reln, path_length=example.path_length, src=example.src, para=para)
                    else:
                        src_sent = SourceTextFormatter.relation_para_node_formatter(
                            relation=example.reln, path_length=example.path_length, src=example.src, para=para)
                ans_sent = example.dest
            elif generation_type == 'question' and add_paragraph:
                src_sent = SourceTextFormatter.natural_question_formatter(
                    relation=example.reln, path_length=example.path_length, src=example.src.lower(), para=para)
                ans_sent = f"{example.src} {example.reln} {example.dest}"

            elif generation_type == 'question_with_prompt':
                if add_paragraph:
                    src_sent = SourceTextFormatter.natural_question_formatter_with_prompt(
                        relation=example.reln, path_length=example.path_length, src=example.src.lower(), para=para,
                        prompt=prompt)
                else:
                    src_sent = SourceTextFormatter.natural_question_formatter_with_prompt(
                        relation=example.reln, path_length=example.path_length, src=example.src.lower(),
                        para="",
                        prompt=prompt)
                ans_sent = f"{example.src} {example.reln} {example.dest}"

            elif generation_type == 'question' and not add_paragraph:
                src_sent = SourceTextFormatter.natural_question_no_para_formatter(
                    relation=example.reln, path_length=example.path_length, src=example.src.lower(), para=para)
                ans_sent = f"{example.src} {example.reln} {example.dest}"
            elif generation_type == 'natural' and add_paragraph:
                src_sent = SourceTextFormatter.natural_sentence_formatter(
                    relation=example.reln, path_length=example.path_length, src=example.src.lower(), para=para)
                ans_sent = example.dest
            elif generation_type == 'natural' and not add_paragraph:
                src_sent = SourceTextFormatter.natural_sentence_no_para_formatter(
                    relation=example.reln, path_length=example.path_length, src=example.src.lower(), para=para)
                ans_sent = example.dest
            else:
                src_sent = f"{example.reln} {example.path_length} {Rels.special_tokens['node_sep']} {example.src}"
                ans_sent = example.dest

            res.append({
                "para_id": graph.para_id,
                "graph_id": graph.graph_id,
                "split": split,
                "reln": example.reln,
                "question": src_sent,
                "answer": ans_sent.lower(),
                "context": para,
                "path_length": example.path_length,
                "id": i,
                "source_node": example.src,
                "target_node": example.dest,
                "source_node_type": refined_node_types[example.src_type],
                "target_node_type": refined_node_types[example.dest_type],
            })
    return pd.DataFrame(res)


class SourceTextFormatter(object):

    @staticmethod
    def natural_question_formatter_with_prompt(relation: str, prompt: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        PARA What RELATION SRC at N HOP?

        Arguments:
            relation {str} -- [Relation]
            prompt {str} -- [Prompt Text]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        #  return f"{para} In the context of {prompt}, What does {src} {relation} at {path_length} ?".strip()
        return f"{para} In the context of {prompt}, What does {src} {relation}?".strip()

    @staticmethod
    def simple_formatter(relation: str, src: str, para: str, path_length: str = None) -> str:
        # return f"{para} <REL> {relation} {Rels.special_tokens['node_sep']} {src}"
        return f"{para} {relation} {src}"

    @staticmethod
    def src_relation_formatter(relation: str, src: str, para: str, path_length: str = None) -> str:
        return f"{src} {relation}"

    @staticmethod
    def natural_question_formatter(relation: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        PARA What does SRC RELATION at N HOP?

        Arguments:
            relation {str} -- [Relation]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"{para} What does {src} {relation} at {path_length} ?"

    def natural_question_no_para_formatter(relation: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        What does SRC RELATION at N HOP?

        Arguments:
            relation {str} -- [Relation]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"What does {src} {relation} at {path_length} ?"

    @staticmethod
    def natural_sentence_formatter(relation: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        PARA What does SRC RELATION at N HOP? SRC RELATION

        Arguments:
            relation {str} -- [Relation]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"{para} What does {src} {relation} at {path_length} ? {src} {relation}"

    @staticmethod
    def natural_sentence_no_para_formatter(relation: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        What does SRC RELATION at N HOP? SRC RELATION

        Arguments:
            relation {str} -- [Relation]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"What does {src} {relation} at {path_length} ? {src} {relation}"

    @staticmethod
    def relation_node_para_formatter(relation: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        RELATION <NODE> SRC <PARA> PARA

        Arguments:
            relation {str} -- [Relation]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"{relation} {path_length} {Rels.special_tokens['node_sep']} {src} {Rels.special_tokens['para_sep']} {para}"

    @staticmethod
    def relation_para_node_formatter(relation: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        RELATION <PARA> PARA <NODE> SRC 

        Arguments:
            relation {str} -- [Relation]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"{relation} {path_length} {Rels.special_tokens['para_sep']} {para} {Rels.special_tokens['node_sep']} {src}"

    @staticmethod
    def para_relation_node_formatter(relation: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        <PARA> PARA RELATION <NODE> SRC 

        Arguments:
            relation {str} -- [Relation]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"{Rels.special_tokens['para_sep']} {para} {relation} {path_length} {Rels.special_tokens['node_sep']} {src}"

    @staticmethod
    def relation_para_node_formatter(relation: str, path_length: str, src: str, para: str) -> str:
        """Formats the source sentence for GPT-2. 
        RELATION <PARA> PARA <NODE> SRC 

        Arguments:
            relation {str} -- [Relation]
            path_length {str} -- [Path length]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"{relation} {path_length} {Rels.special_tokens['para_sep']} {para} {Rels.special_tokens['node_sep']} {src}"

    @staticmethod
    def relation_node_formatter(relation: str, path_length: str, src: str) -> str:
        """Formats the source sentence for GPT-2. 
        RELATION <NODE> SRC 

        Arguments:
            relation {str} -- [Relation]
            src {str} -- [Source sentence]
            para {str} -- [Context paragraph]

        Returns:
            str -- [Source string to be used for training GPT2]
        """
        return f"{relation} {path_length} {Rels.special_tokens['node_sep']} {src}"


if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
