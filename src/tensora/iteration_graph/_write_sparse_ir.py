__all__ = [
    "write_sparse_initialization",
    "write_crd_assembly",
    "write_pos_assembly",
    "write_pos_allocation",
]

from ..format import Mode
from ..ir import SourceBuilder, types
from ..ir.ast import ArrayReallocate, GreaterThanOrEqual, Max, Multiply, Variable
from ._names import dimension_name
from .identifiable_expression import TensorLayer


def write_sparse_initialization(leaf: TensorLayer) -> SourceBuilder:
    source = SourceBuilder()

    index_variable = leaf.layer_pointer()
    start_index = leaf.previous_layer_pointer()
    end_index = leaf.sparse_end_name()
    pos_array = leaf.pos_name()

    source.append(index_variable.declare(types.integer).assign(pos_array.idx(start_index)))
    source.append(end_index.declare(types.integer).assign(pos_array.idx(start_index.plus(1))))

    return source


def write_crd_assembly(output: TensorLayer) -> SourceBuilder:
    source = SourceBuilder("crd assembly")

    pointer = output.layer_pointer()
    capacity = output.crd_capacity_name()
    crd = output.crd_name()
    loop_variable = Variable(output.tensor.indexes[output.layer])

    with source.branch(GreaterThanOrEqual(pointer, capacity)):
        source.append(capacity.assign(capacity.times(2)))
        source.append(crd.assign(ArrayReallocate(crd, types.integer, capacity)))

    source.append(crd.idx(pointer).assign(loop_variable))

    return source


def write_pos_assembly(output: TensorLayer) -> SourceBuilder:
    source = SourceBuilder("pos assembly")

    pointer = output.layer_pointer()
    pos = output.pos_name()
    previous_pointer = output.previous_layer_pointer()

    source.append(pos.idx(previous_pointer.plus(1)).assign(pointer))

    return source


def write_pos_allocation(output: TensorLayer) -> SourceBuilder:
    dense_dimensions = []
    for i_layer in range(output.layer + 1, output.tensor.order):
        index_variable_i = output.tensor.indexes[i_layer]
        mode_i = output.tensor.modes[i_layer]
        if mode_i == Mode.compressed:
            break
        dense_dimensions.append(dimension_name(index_variable_i))

    layer_being_allocated = output.layer + len(dense_dimensions) + 1
    if layer_being_allocated == len(output.tensor.indexes):
        comment = "vals allocation"
        capacity = output.vals_capacity_name()
        array = output.vals_name()
        type = types.float
        bonus = 0
    else:
        comment = "pos allocation for next sparse layer"
        target_leaf = TensorLayer(output.tensor, output.layer + len(dense_dimensions) + 1)
        capacity = target_leaf.pos_capacity_name()
        array = target_leaf.pos_name()
        type = types.integer
        bonus = 1  # pos array is 1 longer

    source = SourceBuilder(comment)

    # TODO: The minimum capacity formulas do not seem consistent, double check them
    if len(dense_dimensions) == 0:
        # Peephole optimization cannot figure out that doubling is always bigger with no dense dimensions, so the
        # dropping of max() must be done manually.
        minimum_capacity = output.layer_pointer().plus(bonus)
        with source.branch(GreaterThanOrEqual(minimum_capacity, capacity)):
            source.append(capacity.assign(capacity.times(2)))
            source.append(array.assign(ArrayReallocate(array, type, capacity)))
    else:
        minimum_capacity = (
            output.layer_pointer().plus(1).times(Multiply.join(dense_dimensions)).plus(bonus)
        )

        with source.branch(GreaterThanOrEqual(minimum_capacity, capacity)):
            source.append(capacity.assign(Max(capacity.times(2), minimum_capacity)))
            source.append(array.assign(ArrayReallocate(array, type, capacity)))

    return source
