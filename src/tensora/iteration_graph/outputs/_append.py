__all__ = ["AppendOutput"]

from dataclasses import dataclass

from ...format import Mode
from ...ir import SourceBuilder, types
from ...ir.ast import ArrayAllocate, Expression, IntegerLiteral, Multiply, Variable
from ...kernel_type import KernelType
from .._names import (
    crd_capacity_name,
    crd_name,
    dimension_name,
    layer_pointer,
    pos_capacity_name,
    pos_name,
    previous_layer_pointer,
    vals_capacity_name,
    vals_name,
)
from ..identifiable_expression import TensorLayer
from ..identifiable_expression import ast as ie_ast
from ._base import Output
from ._bucket import BucketOutput

default_array_size = Multiply(IntegerLiteral(1024), IntegerLiteral(1024))


@dataclass(frozen=True, slots=True)
class AppendOutput(Output):
    output: ie_ast.Tensor
    next_layer: int

    def vals_pointer(self) -> Expression:
        return previous_layer_pointer(self.output.id, self.output.order)

    def write_declarations(self, kernel_type: KernelType):
        source = SourceBuilder("Output initialization")

        target_name = self.output.name
        output_tensor = Variable(target_name)

        all_dense = True
        for i, mode in enumerate(self.output.modes):
            if mode == Mode.dense:
                pass
            elif mode == Mode.compressed:
                if kernel_type.is_assemble():
                    # How pos is handled depends on what the previous modes were
                    if all_dense:
                        # If the previous dimensions were all dense, then the size of pos in this dimension is fixed
                        pos_size = Multiply.join(
                            [dimension_name(self.output.indexes[i_prev]) for i_prev in range(i)]
                        ).plus(1)
                    else:
                        # Otherwise, the value will change, so provide a good default
                        pos_size = default_array_size
                    pos_capacity = pos_capacity_name(target_name, i)
                    pos_array = pos_name(target_name, i)
                    source.append(pos_capacity.declare(types.integer).assign(pos_size))
                    source.append(pos_array.assign(ArrayAllocate(types.integer, pos_capacity)))
                    source.append(pos_array.idx(0).assign(0))

                    # crd is always the same
                    crd_capacity = crd_capacity_name(target_name, i)
                    source.append(crd_capacity.declare(types.integer).assign(default_array_size))
                    source.append(
                        crd_name(target_name, i).assign(ArrayAllocate(types.integer, crd_capacity))
                    )

                # This is the only thing that get written when doing compute
                source.append(layer_pointer(self.output.id, i).declare(types.integer).assign(0))

                all_dense = False
            else:
                raise NotImplementedError()

        if kernel_type.is_assemble():
            if all_dense:
                vals_size = Multiply.join(
                    [output_tensor.attr("dimensions").idx(i) for i in range(self.output.order)]
                )
            else:
                vals_size = default_array_size
            vals_capacity = vals_capacity_name(target_name)
            source.append(vals_capacity.declare(types.integer).assign(vals_size))
            source.append(vals_name(target_name).assign(ArrayAllocate(types.float, vals_capacity)))

        return source

    def write_assignment(self, right_hand_side: Expression, kernel_type: KernelType):
        source = SourceBuilder()

        if self.next_layer != self.output.order:
            raise RuntimeError()

        source.append(vals_name(self.output.name).idx(self.vals_pointer()).assign(right_hand_side))

        return source

    def write_cleanup(self, kernel_type: KernelType):
        source = SourceBuilder(f"Assembling output tensor {self.output.name}")

        if kernel_type.is_assemble():
            target_name = self.output.name
            output_tensor = Variable(target_name)

            for i, mode in enumerate(self.output.modes):
                if mode == Mode.dense:
                    pass
                elif mode == Mode.compressed:
                    source.append(
                        output_tensor.attr("indices")
                        .idx(i)
                        .idx(0)
                        .assign(pos_name(target_name, i))
                    )
                    source.append(
                        output_tensor.attr("indices")
                        .idx(i)
                        .idx(1)
                        .assign(crd_name(target_name, i))
                    )
                else:
                    raise NotImplementedError()
            source.append(output_tensor.attr("vals").assign(vals_name(target_name)))

        return source

    def next_output(
        self, iteration_output: TensorLayer | None, kernel_type: KernelType
    ) -> tuple[Output, SourceBuilder, SourceBuilder]:
        if iteration_output is not None and self.next_layer == iteration_output.layer:
            return AppendOutput(self.output, self.next_layer + 1), SourceBuilder(), SourceBuilder()
        else:
            # No layer or wrong layer was encountered
            dense_only_remaining = all(
                mode == Mode.dense for mode in self.output.modes[self.next_layer :]
            )
            if dense_only_remaining:
                next_output = BucketOutput(
                    self.output, list(range(self.next_layer, len(self.output.modes)))
                )
                dimension_names = [
                    dimension_name(index) for index in self.output.indexes[self.next_layer :]
                ]
                bucket = vals_name(self.output.name).plus(
                    previous_layer_pointer(self.output.id, self.next_layer).times(
                        Multiply.join(dimension_names)
                    )
                )
                return next_output, next_output.write_declarations(bucket), SourceBuilder()
            else:
                raise NotImplementedError(
                    "Encountered a sparse output layer preceded by a contraction layer or a later "
                    "output layer. This requires a hash table to store intermediate outputs, "
                    "which is not currently implemented."
                )
