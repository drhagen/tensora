__all__ = ["BucketOutput"]

from dataclasses import dataclass, replace

from ...format import Mode
from ...ir import SourceBuilder, types
from ...ir.ast import Add, Expression, LessThan, Multiply, Variable
from ...kernel_type import KernelType
from .._names import dimension_name
from ..identifiable_expression import TensorLayer
from ..identifiable_expression import ast as ie_ast
from ._base import Output


@dataclass(frozen=True, slots=True)
class BucketOutput(Output):
    output: ie_ast.Tensor
    layers: list[int]
    unfulfilled: set[int]

    def __init__(
        self, output: ie_ast.Tensor, layers: list[int], unfulfilled: set[int] | None = None
    ):
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "layers", layers)
        if unfulfilled is not None:
            object.__setattr__(self, "unfulfilled", unfulfilled)
        else:
            unfulfilled = {layer for layer in layers if output.modes[layer] == Mode.compressed}
            object.__setattr__(self, "unfulfilled", unfulfilled)

    def write_declarations(self, right_hand_side: Expression) -> SourceBuilder:
        source = SourceBuilder("Bucket initialization")
        source.append(self.name().declare(types.Pointer(types.float)).assign(right_hand_side))
        bucket_loop_index = self.loop_name()
        source.append(bucket_loop_index.declare(types.integer).assign(0))
        with source.loop(LessThan(bucket_loop_index, Multiply.join(self.dimension_names()))):
            source.append(self.name().idx(bucket_loop_index).assign(0))
            source.append(bucket_loop_index.increment())
        return source

    def next_output(
        self, iteration_output: TensorLayer | None, kernel_type: KernelType
    ) -> tuple[Output, SourceBuilder, SourceBuilder]:
        if iteration_output is None:
            return self, SourceBuilder(), SourceBuilder()
        else:
            next_unfulfilled = self.unfulfilled - {iteration_output.layer}
            return replace(self, unfulfilled=next_unfulfilled), SourceBuilder(), SourceBuilder()

    def write_assignment(self, right_hand_side: Expression, kernel_type: KernelType):
        source = SourceBuilder()
        bucket_index = self.ravel_indexes(
            self.dimension_names(),
            [Variable(self.output.indexes[layer]) for layer in self.layers],
        )
        source.append(self.name().idx(bucket_index).increment(right_hand_side))
        return source

    def ravel_indexes(self, dimensions: list[Variable], indexes: list[Variable]):
        dimensions_so_far: list[Variable] = []
        terms: list[Expression] = []
        for dim_i, index_i in zip(reversed(dimensions), reversed(indexes), strict=True):
            terms.append(Multiply.join([index_i, *dimensions_so_far]))
            dimensions_so_far.append(dim_i)

        # Reverse it to make it look nice
        return Add.join(list(reversed(terms)))

    def name(self) -> Variable:
        return Variable(f'bucket_{self.output.id}{"".join(f"_{x}" for x in self.layers)}')

    def loop_name(self) -> Variable:
        return Variable(f'i_bucket_{self.output.id}{"".join(f"_{x}" for x in self.layers)}')

    def dimension_names(self):
        return [dimension_name(self.output.indexes[layer]) for layer in self.layers]
