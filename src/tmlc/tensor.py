from abc import ABC, abstractmethod
from numpy import ndarray
from typing_extensions import override


class Tensor:
    """A Tensor is a node in a computational graph, representing a multi-dimensional array.
    Tensors are the inputs to tensor operations, which output new tensors. A sequence of Tensor
    Operations chained together produces a computational graph, which we can compile and optimize.
    Tensors do NOT actually hold data themseleves, but rather represent the flow of data through
    the graph. The actual data is held in buffers that are supplied at evaluation time.
    """

    inputs: list["Tensor"]
    op: "TensorOp"
    label: str
    shape: tuple[int, ...]
    dtype: str

    def __init__(
        self,
        inputs: list["Tensor"],
        op: "TensorOp",
        shape: tuple[int, ...],
        label: str | None = None,
        dtype: str = "float32",
    ):
        self.inputs = inputs
        self.op = op
        if label is None:
            self.label = self.op.__class__.__name__
        else:
            self.label = label

        self.shape = shape

        # TODO: support more dtypes
        # TODO: inheirit dtypes from input tensors
        # TODO: dtype promotion logic for mismatched input tensor
        self.dtype = dtype

    @override
    def __str__(self):
        inputs = ", ".join(str(tensor) for tensor in self.inputs)
        return f"Tensor(op={self.label}, inputs=[{inputs}])"

    @override
    def __repr__(self):
        return self.__str__()

    def __add__(self, other: "Tensor|float|int") -> "Tensor":
        from tmlc.ops.ops_arithmetic import add
        from tmlc.ops.ops_basic import constant

        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return add(self, other)

    def __radd__(self, other: "Tensor|float|int") -> "Tensor":
        from tmlc.ops.ops_arithmetic import add
        from tmlc.ops.ops_basic import constant

        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return add(self, other)

    def __mul__(self, other: "Tensor|float|int") -> "Tensor":
        from tmlc.ops.ops_arithmetic import mul
        from tmlc.ops.ops_basic import constant

        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return mul(self, other)

    def __rmul__(self, other: "Tensor|float|int") -> "Tensor":
        from tmlc.ops.ops_arithmetic import mul
        from tmlc.ops.ops_basic import constant

        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return mul(self, other)

    def __truediv__(self, other: "Tensor|float|int") -> "Tensor":
        from tmlc.ops.ops_arithmetic import div
        from tmlc.ops.ops_basic import constant

        if isinstance(other, (int, float)):
            other = constant(1 / other, label=str(1 / other))
        return div(self, other)

    def __sub__(self, other: "Tensor|float|int") -> "Tensor":
        from tmlc.ops.ops_arithmetic import add, negate
        from tmlc.ops.ops_basic import constant

        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return add(self, negate(other))

    def __rsub__(self, other: "Tensor|float|int") -> "Tensor":
        from tmlc.ops.ops_arithmetic import add, negate
        from tmlc.ops.ops_basic import constant

        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return add(other, negate(self))

    def __neg__(self) -> "Tensor":
        from tmlc.ops.ops_arithmetic import negate

        return negate(self)

    def __pow__(self, other: "Tensor|float|int") -> "Tensor":
        from tmlc.ops.ops_arithmetic import power
        from tmlc.ops.ops_basic import constant

        if isinstance(other, (int, float)):
            other = constant(other, label=str(other))
        return power(self, other)

    def __matmul__(self, other: "Tensor") -> "Tensor":
        from tmlc.ops.ops_arithmetic import mm

        return mm(self, other)

    @property
    def T(self) -> "Tensor":
        from tmlc.ops.ops_shape import transpose

        return transpose(self)


class ConstantTensor(Tensor):
    """
    Constant is a special type of Tensor that holds a buffer that is managed outside of the graph.
    It is operationally equivalent to an Input node, except that its value is determined at creation
    time, rather than supplied at evaluation time.
    It is a leaf node in the graph and does not have any input tensors.
    The value of a Constant tensor is stored in the `constval` field.
    """

    constval: ndarray

    def __init__(self, value: ndarray, op: "TensorOp", label: str | None = None):
        super().__init__(inputs=[], op=op, label=label, shape=value.shape)
        self.constval = value


class TensorOp(ABC):
    """TensorOp interface represents an operation that can be performed on Tensors."""

    @abstractmethod
    def __call__(
        self,
        inputs: list[Tensor],
        label: str | None = None,
    ) -> Tensor:
        """When a TensorOp is called, it should create a new Tensor that represents the output of
        this operation."""
        raise NotImplementedError("TensorOp subclasses must implement __call__")

    @abstractmethod
    def infer_shape(self, inputs: list[Tensor]) -> tuple[int, ...]:
        """
        Given the input tensors (which contain their shapes), infer the shape of the
        output tensor that this operation will produce.
        """
        raise NotImplementedError("TensorOp subclasses must implement infer_shape()")

    @abstractmethod
    def compute(self, inputs: list[ndarray]) -> list[ndarray]:
        """Given the input arrays, compute the output arrays of this operation.

        This is used by the evaluator to compute the values of the output tensors in the graph. This
        operates on concrete arrays to actually determine a concrete value, and is used for eager
        mode evaluation.
        """
        raise NotImplementedError("TensorOp subclasses must implement compute()")

    @abstractmethod
    def gradients(self, tensor: Tensor, incoming_grad: Tensor) -> list[Tensor]:
        """Given the output of the forward `call` method and the incoming gradient from the
        backwards pass, this method calculates the gradients to propagate to the inputs.

        The calculated gradients must be arranged in a list that corresponds to the original
        ordering of the input tensors.
        """
        raise NotImplementedError("TensorOp subclasses must implement gradients()")

    @abstractmethod
    def emit_ir(self, inputs: list[str]) -> str:
        # TODO: may need to update function signature here. Do we need input tensor labels?
        """If compiling the graph, each TensorOp needs to emit IR that represents this operations
        computation.

        The compiler composes the graphs full IR to optimize and generate the final code.
        """
        raise NotImplementedError("TensorOp subclasses must implement emit_ir()")
