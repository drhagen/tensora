__all__ = ["HashOutput"]

from dataclasses import dataclass, replace

from ...format import Mode
from ...ir import SourceBuilder, types
from ...ir.ast import (
    Address,
    ArrayAllocate,
    ArrayLiteral,
    Break,
    Expression,
    Free,
    FunctionCall,
    IntegerLiteral,
    LessThan,
    ModeLiteral,
    NotEqual,
    Variable,
)
from ...kernel_type import KernelType
from ..identifiable_expression import TensorLayer
from ..identifiable_expression import ast as ie_ast
from ..names import dimension_name, layer_pointer, previous_layer_pointer, vals_name
from ..write_sparse_ir import write_crd_assembly, write_pos_assembly
from .base import Output
from .bucket import BucketOutput


@dataclass(frozen=True, slots=True)
class HashOutput(Output):
    output: ie_ast.Variable
    starting_layer: int
    unfulfilled: set[int]

    def __init__(
        self, output: ie_ast.Variable, starting_layer: int, unfulfilled: set[int] | None = None
    ):
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "starting_layer", starting_layer)
        if unfulfilled is not None:
            object.__setattr__(self, "unfulfilled", unfulfilled)
        else:
            object.__setattr__(
                self, "unfulfilled", set(range(starting_layer, self.final_dense_index()))
            )

    def final_dense_index(self):
        final_dense_index = self.output.order
        for i in reversed(range(self.starting_layer, self.output.order)):
            if self.output.modes[i] == Mode.compressed:
                break
            else:
                final_dense_index = i

        return final_dense_index

    def key_number(self, layer: int):
        number = 0
        for i in range(self.starting_layer, self.output.order):
            if layer == i:
                return number
            if self.output.modes[i] == Mode.compressed:
                number += 1
        return number

    def write_declarations(self) -> SourceBuilder:
        source = SourceBuilder("Hash table initialization")

        modes = [ModeLiteral(mode) for mode in self.output.modes[self.starting_layer :]]
        dims = [
            dimension_name(variable) for variable in self.output.indexes[self.starting_layer :]
        ]

        source.add_dependency("hash")

        # Construct hash table
        source.append(self.name().declare(types.hash_table))
        source.append(
            self.modes_name().declare(types.Array(types.mode)).assign(ArrayLiteral(modes))
        )
        source.append(
            self.dims_name().declare(types.Array(types.integer)).assign(ArrayLiteral(dims))
        )
        source.append(
            FunctionCall(
                Variable("hash_construct"),
                [
                    Address(self.name()),
                    IntegerLiteral(len(modes)),
                    IntegerLiteral(self.final_dense_index() - self.starting_layer),
                    self.modes_name(),
                    self.dims_name(),
                ],
            )
        )

        return source

    def write_assignment(self, right_hand_side: str, kernel_type: KernelType) -> SourceBuilder:
        raise RuntimeError()

    def write_cleanup(self, kernel_type: KernelType) -> SourceBuilder:
        source = SourceBuilder("Hash table cleanup")

        order_name = self.order_name()
        loop_name = self.sort_index_name()

        # Argsort the elements by key
        source.append(
            self.order_name()
            .declare(types.Pointer(types.integer))
            .assign(ArrayAllocate(types.integer, self.name().attr("count")))
        )
        source.append(loop_name.declare(types.integer).assign(0))
        with source.loop(LessThan(loop_name, self.name().attr("count"))):
            source.append(self.order_name().idx(loop_name).assign(loop_name))
            source.append(loop_name.increment())
        source.append(
            FunctionCall(
                Variable("qsort_r"),
                [
                    self.order_name(),
                    self.name().attr("count"),
                    Variable("sizeof(uint32_t)"),  # Temporary hack
                    Variable("hash_comparator"),
                    Address(self.name()),
                ],
            )
        )

        # Extract indexes recursively
        source.append(self.extract_index_name().declare(types.integer).assign(0))
        source.append(self.write_layer_cleanup(self.starting_layer, kernel_type))

        # Free temporaries
        source.append(Free(order_name))

        # Free hash table
        source.append(FunctionCall(Variable("hash_destruct"), [Address(self.name())]))

        return source

    def write_layer_cleanup(self, layer: int, kernel_type: KernelType):
        source = SourceBuilder()

        if layer < self.final_dense_index():
            key_number = self.key_number(layer)
            layer_index = Variable(self.output.indexes[layer])
            dimension_size = dimension_name(self.output.indexes[layer])
            position = layer_pointer(self.output.variable, layer)
            previous_position = previous_layer_pointer(self.output.variable, layer)
            end_position = self.end_position(key_number)
            next_end_position = self.end_position(key_number + 1)

            # Reusable search code
            # This is not applicable for the final key, which has no next key
            search_source = SourceBuilder()
            search_source.append(
                next_end_position.declare(types.integer).assign(self.extract_index_name())
            )
            with search_source.loop(LessThan(next_end_position, end_position)):
                with search_source.branch(
                    NotEqual(
                        self.name()
                        .attr("keys")
                        .idx(self.order_name().idx(next_end_position))
                        .idx(key_number),
                        layer_index,
                    )
                ):
                    search_source.append(Break())
                search_source.append(next_end_position.increment())

            # Keys phase
            if self.output.modes[layer] == Mode.dense:
                source.append(layer_index.declare(types.integer).assign(0))
                with source.loop(LessThan(layer_index, dimension_size)):
                    source.append(
                        position.declare(types.integer).assign(previous_position.plus(layer_index))
                    )
                    source.append(search_source)
                    source.append(self.write_layer_cleanup(layer + 1, kernel_type))
                    source.append(layer_index.increment())

                    if layer == self.final_dense_index() - 1:
                        source.append(self.extract_index_name().increment())

            elif self.output.modes[layer] == Mode.compressed:
                with source.loop(LessThan(self.extract_index_name(), end_position)):
                    source.append(
                        layer_index.declare(types.integer).assign(
                            self.name()
                            .attr("keys")
                            .idx(self.order_name().idx(self.extract_index_name()))
                            .idx(key_number)
                        )
                    )
                    source.append(search_source)
                    source.append(self.write_layer_cleanup(layer + 1, kernel_type))

                    if kernel_type.is_assembly():
                        source.append(write_crd_assembly(TensorLayer(self.output, layer)))
                    source.append(position.increment())

                    if layer == self.final_dense_index() - 1:
                        source.append(self.extract_index_name().increment())

            if kernel_type.is_assembly():
                source.append(write_pos_assembly(TensorLayer(self.output, layer)))
        elif layer < self.output.order:
            # Bucket phase
            layer_index = Variable(self.output.indexes[layer])
            dimension_size = dimension_name(self.output.indexes[layer])
            position = layer_pointer(self.output.variable, layer)
            previous_position = previous_layer_pointer(self.output.variable, layer)
            bucket_position = self.bucket_position(layer)
            previous_bucket_position = self.previous_bucket_position(layer)

            source.append(layer_index.declare(types.integer).assign(0))
            with source.loop(LessThan(layer_index, dimension_size)):
                source.append(
                    position.declare(types.integer).assign(previous_position.plus(layer_index))
                )
                source.append(
                    bucket_position.declare(types.integer).assign(
                        previous_bucket_position.plus(layer_index)
                    )
                )
                source.append(self.write_layer_cleanup(layer + 1, kernel_type))
                source.append(layer_index.increment())
        elif layer == self.output.order:
            # Final phase
            vals = vals_name(self.output.variable.name)
            previous_position = previous_layer_pointer(self.output.variable, layer)
            previous_bucket_position = self.previous_bucket_position(layer)
            bucket = BucketOutput(
                self.output, list(range(self.final_dense_index(), self.output.order))
            )
            source.append(
                vals.idx(previous_position).assign(bucket.name().idx(previous_bucket_position))
            )

        return source

    def next_output(
        self, iteration_output: TensorLayer | None, kernel_type: KernelType
    ) -> tuple[Output, SourceBuilder, SourceBuilder]:
        if iteration_output is None:
            return self, SourceBuilder(), SourceBuilder()
        else:
            next_unfulfilled = self.unfulfilled - {iteration_output.layer}
            if len(next_unfulfilled) == 0:
                final_dense_index = self.final_dense_index()

                next_output = BucketOutput(
                    self.output, list(range(final_dense_index, self.output.order))
                )

                # Write declaration of bucket
                source = SourceBuilder()

                key_names = [
                    Variable(self.output.indexes[layer])
                    for layer in range(self.starting_layer, final_dense_index)
                ]
                key_name = self.key_name()

                source.append(
                    key_name.declare(types.Array(types.integer)).assign(ArrayLiteral(key_names))
                )
                source.append(
                    next_output.write_declarations(
                        FunctionCall(
                            Variable("hash_get_bucket"),
                            [
                                Address(self.name()),
                                key_name,
                            ],
                        )
                    )
                )

                return next_output, source, SourceBuilder()
            else:
                return (
                    replace(self, unfulfilled=next_unfulfilled),
                    SourceBuilder(),
                    SourceBuilder(),
                )

    def name(self) -> Variable:
        return Variable("hash_table")

    def modes_name(self) -> Variable:
        return Variable(f"i_{self.name().name}_modes")

    def dims_name(self) -> Variable:
        return Variable(f"i_{self.name().name}_dims")

    def key_name(self) -> Variable:
        return Variable(f"i_{self.name().name}_key")

    def order_name(self) -> Variable:
        return Variable(f"{self.name().name}_order")

    def sort_index_name(self) -> Variable:
        return Variable(f"i_{self.name().name}_argsort")

    def extract_index_name(self) -> Variable:
        return Variable(f"p_{self.name().name}_order")

    def end_position(self, key_number: int) -> Expression:
        if key_number == 0:
            return self.name().attr("count")
        else:
            return Variable(f"p_{self.name().name}_order_{key_number}_end")

    def bucket_position(self, layer: int):
        return Variable(layer_pointer(self.output.variable, layer).name + "_bucket")

    def previous_bucket_position(self, layer: int):
        if layer == 0:
            return IntegerLiteral(0)
        else:
            return self.bucket_position(layer - 1)
