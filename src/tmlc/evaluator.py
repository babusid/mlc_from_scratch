"""Evaluator for TMLC computational graphs."""

import numpy as np
import warnings
from tmlc.ndarray import ndarray
from tmlc.tensor import Tensor


def run(
    inputs: dict[Tensor, ndarray], outputs: list[Tensor]
) -> list[list[ndarray] | ndarray]:
    # eager mode evaluation
    # 1. reverse topo sort graph from outputs to inputs and
    #    build a traversal / target set of nodes that we have to compute
    #    compute them in order until outputs have been computed
    visited: set[Tensor] = set()
    topo_sort: list[Tensor] = []
    for node in outputs:
        _dfs_helper_topo_sort(node, visited, topo_sort)

    if any(input not in visited for input in inputs.keys()):
        warnings.warn(
            "Some provided input tensors not used in computation graph for requested outputs."
        )
    intermediates: dict[Tensor, list[ndarray]] = {}
    for node in topo_sort:
        if node in inputs:
            intermediates[node] = [inputs[node]]
        else:
            input_values = [intermediates[input][0] for input in node.inputs]
            assert len(input_values) == len(node.inputs), (
                "Mismatch in number of input values and node inputs"
            )
            intermediates[node] = node.op.compute(input_values)
    output: list[list[ndarray] | ndarray] = []
    for out in outputs:
        output.append(intermediates[out])

    return output

def gradients():
    # extend graph with gradients for desired node grads
    return

def compile():
    return

def run_compiled():
    return


def _dfs_helper_topo_sort(node: Tensor, visited: set[Tensor], topo_sort: list[Tensor]) -> None:
    """Helper function for topological sort using post-order DFS traversal.

    This ensures all nodes are processed after their children (dependencies).

    Parameters
    ----------
    node: Node
        The current node to process
    visited: Set[Node]
        Set of already visited nodes
    topo_sort: List[Node]
        List to append nodes in topological order
    """
    if node in visited:
        return
    visited.add(node)
    # Process children FIRST
    for input_node in node.inputs:
        _dfs_helper_topo_sort(input_node, visited, topo_sort)
    # THEN add this node
    topo_sort.append(node)
