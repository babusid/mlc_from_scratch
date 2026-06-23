from __future__ import annotations

import copy
from collections import defaultdict
from functools import reduce
from tmlc.tensor.tensor import Tensor
from tmlc.tensor.ops.ops_basic import Constant, Input
from tmlc.tensor.ops.ops_shape import ones_like
from tmlc.util.topo_sort import dfs_helper_topo_sort
from tmlc.ndarray import ndarray
from abc import ABC, abstractmethod

class GraphTransform(ABC):
    '''
    Base class for graph transformations. Subclasses must implement the __call__ method.
    We implement a class for GraphTransform rather than allowing a raw Callable in order to avoid
    variadic arguments in the __call__ method, which would make it difficult to type check the
    transform functions. Instead, the intended pattern is to define custom constructors for
    each transform class that emit custom GraphTransform objects with the appropriate parameters,
    thereby allowing the actual __call__ signature to be uniform across all GraphTransform
    subclasses.
    '''
    @abstractmethod
    def __call__(self, graph: Graph) -> Graph:
        raise NotImplementedError("GraphTransform subclasses must implement __call__")

class Graph:
    """
    Explicit graph object that traces a computation graph built from Tensor objects.
    Must be built in order to actually run the computation graph, as well as to apply
    graph optimizations for increased performance. Building a Graph is the first part
    of the compilation process.
    """
    inputs: list[Tensor]
    outputs: list[Tensor]
    topo_sort: list[Tensor]

    def __init__(self, inputs: list[Tensor], outputs: list[Tensor]) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.topo_sort = self._build_topo_sort()

    def _build_topo_sort(self) -> list[Tensor]:
        """Return a topological sort of the graph's nodes."""
        visited: set[Tensor] = set()
        _topo: list[Tensor] = []
        for node in self.outputs:
            dfs_helper_topo_sort(node, visited, _topo)
        return _topo

    def apply_transforms(self, transform_fns: list[GraphTransform]) -> Graph:
        """Apply a transform pipeline to the graph"""
        init: Graph = copy.deepcopy(self)
        return reduce(lambda graph, fn: fn(graph), transform_fns, init)

    def run(self, inputs: dict[Tensor, ndarray]) -> list[list[ndarray]]:
        outputs = self.outputs
        topo_sort = self.topo_sort
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
        output: list[list[ndarray]] = []
        for out in outputs:
            output.append(intermediates[out])

        return output

    def compile(self) -> None:
        return

    def run_compiled(self) -> None:
        return

def differentiate(graph: Graph, output_node: Tensor, target_nodes: list[Tensor]) -> Graph:
    visited: set[Tensor] = set()
    rev_topo_sort: list[Tensor] = []
    rev_topo_sort = [t for t in graph.topo_sort]
    rev_topo_sort.reverse()

    output_grad = ones_like(output_node, "output_grad")

    # track which nodes in the graph we actually have to
    # compute gradients for. This includes the targets,
    # the output gradient, and everything on the path between them
    target_set = set(target_nodes + [output_node])
    visited = set()

    def generate_target_set(tensor: Tensor) -> bool:
        # if we've already explored this node, just return whether it ended up a target
        if tensor in visited:
            return tensor in target_set
        visited.add(tensor)

        # inputs and constants are leaves: they're only targets if explicitly requested
        if isinstance(tensor.op, Input) or isinstance(tensor.op, Constant):
            return tensor in target_set

        # always recurse, even if `tensor` is already an explicit target, since ancestors
        # further up the graph still need to be connected through to it
        reached_target = tensor in target_set
        for input in tensor.inputs:
            if generate_target_set(input):
                reached_target = True

        if reached_target:
            target_set.add(tensor)
        return reached_target

    _ = generate_target_set(output_node)

    # now we have all nodes we have to compute gradients for in target_set
    # for each node, we have to track what is coming in backwards
    node_grad_incoming: dict[Tensor, list[Tensor]] = defaultdict(list)
    # output node just gets the all one output gradient
    node_grad_incoming[output_node] = [output_grad]
    # map tensor to the aggregate of its input gradients
    node_grad: dict[Tensor, Tensor] = {}
    for node in rev_topo_sort:
        if node not in target_set:
            continue
        # get all incoming partial gradients, and aggregate them
        incoming_grad = node_grad_incoming[node]
        sum_grad = incoming_grad[0]
        for grad in incoming_grad[1:]:
            sum_grad += grad
        node_grad[node] = sum_grad
        # pass in aggregate gradient and calculate the gradients
        # wrt this nodes inputs
        input_grads = node.op.gradients(node, sum_grad)
        for input, input_grad in zip(node.inputs, input_grads):
            # pass in the appropriate gradients
            node_grad_incoming[input].append(input_grad)

    bwd_outputs = [node_grad[target] for target in target_nodes]
    return Graph(graph.inputs, bwd_outputs)


