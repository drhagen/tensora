from __future__ import annotations

__all__ = ["generate_ir"]

from functools import singledispatch

from ..format import Mode
from ..ir import SourceBuilder, types
from ..ir.ast import (
    And,
    Block,
    BooleanToInteger,
    Branch,
    Equal,
    Expression,
    GreaterThan,
    IntegerLiteral,
    LessThan,
    Min,
    Return,
    Variable,
)
from ..kernel_type import KernelType
from .definition import Definition
from .identifiable_expression import to_ir
from .identifiable_expression.tensor_layer import TensorLayer
from .iteration_graph import IterationGraph, IterationVariable, Sum, TerminalExpression
from .names import crd_name, dimension_name, pos_name, vals_name
from .outputs import AppendOutput, Output
from .write_sparse_ir import (
    write_crd_assembly,
    write_pos_allocation,
    write_pos_assembly,
    write_sparse_initialization,
)


@singledispatch
def to_ir_iteration_graph(
    self: IterationGraph, output: Output, kernel_type: KernelType
) -> SourceBuilder:
    raise NotImplementedError(
        f"iteration_graph_to_c_code not implemented for {type(self)}: {self}"
    )


@to_ir_iteration_graph.register(TerminalExpression)
def to_ir_terminal_expression(self: TerminalExpression, output: Output, kernel_type: KernelType):
    source = SourceBuilder("*** Computation of expression ***")

    if kernel_type.is_compute():
        source.append(output.write_assignment(to_ir(self.expression), kernel_type))

    return source


def generate_subgraphs(graph: IterationVariable) -> list[IterationVariable]:
    # The 0th element is just the full graph
    # Each element is derived from a previous element by zeroing a tensor
    # Zeroing a tensor always results in a strictly simpler graph
    # Zeroing a tensor will cause tensors multiplied by it to be zeroed as well
    # This means that zeroing two different tensors can result in the same graph
    # Subgraphs can be keyed by their set of remaining sparse tensors to eliminate duplicates
    all_subgraphs = {graph.compressed_dimensions(): graph}
    old_subgraphs = all_subgraphs.copy()

    while True:
        new_graphs = {}

        for sparse_layers, old_graph in old_subgraphs.items():
            # Reverse sparse_dimensions so that last values dropped first
            for sparse_layer in reversed(sparse_layers):
                new_graph = old_graph.exhaust_tensor(sparse_layer)
                new_graphs[new_graph.compressed_dimensions()] = new_graph

        if len(new_graphs) == 0:
            break
        else:
            all_subgraphs.update(new_graphs)
            old_subgraphs = new_graphs

    return list(all_subgraphs.values())


@to_ir_iteration_graph.register(IterationVariable)
def to_ir_iteration_variable(self: IterationVariable, output: Output, kernel_type: KernelType):
    source = SourceBuilder(f"*** Iteration over {self.index_variable} ***")

    if not kernel_type.is_compute() and not self.has_assemble():
        # If not doing compute and no assembly is left, return early
        return source

    loop_variable = Variable(self.index_variable)

    # If this node is sparse, then we can use efficient sparse iteration. Sparse graphs produce
    # sparse subgraphs, so sparsity is a constant for this iteration. A node is sparse if its input
    # graph is sparse and either its output graph is sparse or it is a contraction.
    is_sparse = self.is_sparse_input() and (self.output is None or self.is_sparse_output())

    # Compute the next output
    next_output, next_output_declarations, next_output_cleanup = output.next_output(
        self.output, kernel_type
    )
    source.append(next_output_declarations)

    ##################
    # Initialization #
    ##################
    if not is_sparse:
        # If this loop is dense, we need to initialize the index variable
        source.append(loop_variable.declare(types.integer).assign(0))

    for leaf in self.sparse_leaves():
        source.append(write_sparse_initialization(leaf))

    ############
    # Subnodes #
    ############
    subnodes = generate_subgraphs(self)
    for subnode in subnodes:
        if is_sparse and len(subnode.compressed_dimensions()) == 0:
            # If the loop is sparse and the subnode has no sparse leaves, then
            # it is a sparse zero and should be skipped. Such node is
            # necessarily the last one, so break or continue would do the same
            # thing.
            continue

        sparse_subnode_leaves = subnode.sparse_leaves()
        dense_subnode_leaves = subnode.dense_leaves()

        ########
        # Loop #
        ########
        if len(sparse_subnode_leaves) > 0:
            while_criteria = And.join(
                [
                    LessThan(leaf.layer_pointer(), leaf.sparse_end_name())
                    for leaf in sparse_subnode_leaves
                ]
            )
        else:
            # Last subnode may be dense, so the last loop should finish out the index variable
            while_criteria = LessThan(loop_variable, dimension_name(self.index_variable))

        with source.loop(while_criteria):
            ##########################
            # Extract sparse indexes #
            ##########################
            index_variables = []
            for leaf in sparse_subnode_leaves:
                index_variable = leaf.value_from_crd()
                index_variables.append(index_variable)
                source.append(
                    index_variable.declare(types.integer).assign(
                        leaf.crd_name().idx(leaf.layer_pointer())
                    )
                )

            if is_sparse:
                # If the loop is sparse, the index variable is the closest value
                # among all the sparse layers
                source.append(
                    loop_variable.declare(types.integer).assign(Min.join(index_variables))
                )

            #########################
            # Compute dense indexes #
            #########################
            # Earlier layers may not be iterated over yet if they are dense. The position of
            # earlier layers are prerequisites for computing the position of this layer. Defer
            # computing the position of this layer until all earlier layers have been iterated.
            # This means that we must check if later layers have already been iterated over and can
            # now be satisfied. If so, compute the position of those layers now.

            if (
                self.output is not None
                and self.output.mode == Mode.dense
                and isinstance(output, AppendOutput)
            ):
                maybe_dense_output = [self.output]
            else:
                maybe_dense_output = []

            for leaf in maybe_dense_output + dense_subnode_leaves:
                needed_indexes: set[str] = set()
                for i_layer in reversed(range(0, leaf.layer)):
                    if leaf.tensor.modes[i_layer] == Mode.dense:
                        needed_indexes.add(leaf.tensor.indexes[i_layer])
                    else:
                        break

                layers_to_write = []
                for i_layer in range(leaf.layer, leaf.tensor.order):
                    if leaf.tensor.modes[i_layer] != Mode.dense:
                        # Only consider adjacent dense layers
                        break

                    needed_indexes.add(leaf.tensor.indexes[i_layer])
                    available_before_here = needed_indexes - self.later_indexes()
                    available_after_here = available_before_here | {self.index_variable}

                    if not needed_indexes.issubset(
                        available_before_here
                    ) and needed_indexes.issubset(available_after_here):
                        # Layers whose prerequisites are satisfied with this layer, but not
                        # satisfied without it are written now.
                        layers_to_write.append(TensorLayer(leaf.tensor, i_layer))

                for layer in layers_to_write:
                    pointer = layer.layer_pointer()
                    previous_pointer = layer.previous_layer_pointer()
                    index_variable_i = layer.tensor.indexes[layer.layer]
                    pointer_value = previous_pointer.times(dimension_name(index_variable_i)).plus(
                        index_variable_i
                    )
                    source.append(pointer.declare(types.integer).assign(pointer_value))

            ###############
            # Subsubnodes #
            ###############
            subsubnodes = generate_subgraphs(subnode)
            subsubnode_leaves: list[tuple[Expression, Block]] = []
            for subsubnode in subsubnodes:
                if is_sparse and len(subsubnode.compressed_dimensions()) == 0:
                    # If the loop is sparse and the subsubnode has no sparse
                    # leaves, then it is a sparse zero and should be skipped.
                    # Such node is necessarily the last one, so break or
                    # continue would do the same thing.
                    continue

                sparse_subsubnode_leaves = subsubnode.sparse_leaves()

                ############################
                # Branch on sparse matches #
                ############################
                condition = And.join(
                    [
                        Equal(leaf.value_from_crd(), loop_variable)
                        for leaf in sparse_subsubnode_leaves
                    ]
                )
                block = SourceBuilder()

                ###################################
                # Allocate space for pos and vals #
                ###################################
                # This is done at this level, rather than up a level, because none
                # of the if-statements may hit. In that case, no allocation should
                # be done.
                if (
                    kernel_type.is_assemble()
                    and self.is_sparse_output()
                    and isinstance(next_output, AppendOutput)
                ):
                    block.append(write_pos_allocation(self.output))

                ################################
                # Store position of next layer #
                ################################
                # This allows the current sparse layer to remember if the next sparse layer
                # had any nonzeros. This is conceptually part of the next layer, but it is used
                # by this layer's crd assembly, so we put it here.
                if self.is_sparse_output() and self.next.is_sparse_output():
                    with block.block("Save next layer's position"):
                        next_output_layer: TensorLayer = self.next.output
                        next_pointer = next_output_layer.layer_pointer()
                        next_pointer_begin = next_output_layer.layer_begin_name().declare(
                            types.integer
                        )
                        block.append(next_pointer_begin.assign(next_pointer))

                #####################
                # Invoke next layer #
                #####################
                block.append(to_ir_iteration_graph(subsubnode.next, next_output, kernel_type))

                ########################################################
                # Write sparse output index and advance output pointer #
                ########################################################
                if self.is_sparse_output() and isinstance(next_output, AppendOutput):
                    if self.next.is_sparse_output():
                        # Only advance the index for this sparse layer if the next sparse layer
                        # had any nonzeros.
                        next_output_layer: TensorLayer = self.next.output
                        next_pointer = next_output_layer.layer_pointer()
                        next_pointer_begin = next_output_layer.layer_begin_name()
                        with block.branch(GreaterThan(next_pointer, next_pointer_begin)):
                            if kernel_type.is_assemble():
                                block.append(write_crd_assembly(self.output))
                            block.append(self.output.layer_pointer().increment())
                    else:
                        if kernel_type.is_assemble():
                            block.append(write_crd_assembly(self.output))
                        block.append(self.output.layer_pointer().increment())

                subsubnode_leaves.append((condition, block.finalize()))

            source.append(Branch.join(subsubnode_leaves))

            #######################
            # Increment positions #
            #######################
            for leaf in sparse_subnode_leaves:
                source.append(
                    leaf.layer_pointer().increment(
                        BooleanToInteger(Equal(leaf.value_from_crd(), loop_variable))
                    )
                )
            if not is_sparse:
                # If this loop is dense, we need to increment the index variable
                source.append(loop_variable.increment())

    ################################
    # Write sparse output position #
    ################################
    if (
        kernel_type.is_assemble()
        and self.is_sparse_output()
        and isinstance(next_output, AppendOutput)
    ):
        source.append(write_pos_assembly(self.output))

    source.append(next_output_cleanup)

    return source


@to_ir_iteration_graph.register(Sum)
def to_ir_sum(self: Sum, output: Output, kernel_type: KernelType):
    source = SourceBuilder("*** Sum ***")

    if kernel_type.is_compute():
        # No assembly is currently allowed downstream of a Sum node
        next_output, next_output_declarations, next_output_cleanup = output.next_output(
            None, kernel_type
        )
        source.append(next_output_declarations)

        for term in self.terms:
            source.append(to_ir_iteration_graph(term, next_output, kernel_type))

    return source


def generate_ir(definition: Definition, graph: IterationGraph, kernel_type: KernelType):
    source = SourceBuilder()

    # Function declaration
    parameters = {name: types.Pointer(types.tensor) for name in definition.formats.keys()}
    with source.function_definition(kernel_type.name, parameters, types.integer):
        # Dimensions of all index variables
        with source.block("Extract dimensions"):
            for index_name, tensor_layer in definition.indexes.items():
                declaration = dimension_name(index_name).declare(types.integer)
                value = Variable(tensor_layer.name).attr("dimensions").idx(tensor_layer.dimension)
                source.append(declaration.assign(value))

        # Unpack tensors
        with source.block("Unpack tensors"):
            for tensor_name, format in definition.formats.items():
                for i, mode in enumerate(format.modes):
                    match mode:
                        case Mode.dense:
                            pass
                        case Mode.compressed:
                            pos_declaration = pos_name(tensor_name, i).declare(
                                types.Pointer(types.integer)
                            )
                            pos_value = Variable(tensor_name).attr("indices").idx(i).idx(0)
                            source.append(pos_declaration.assign(pos_value))
                            crd_declaration = crd_name(tensor_name, i).declare(
                                types.Pointer(types.integer)
                            )
                            crd_value = Variable(tensor_name).attr("indices").idx(i).idx(1)
                            source.append(crd_declaration.assign(crd_value))

                vals_declaration = vals_name(tensor_name).declare(types.Pointer(types.float))
                vals_value = Variable(tensor_name).attr("vals")
                source.append(vals_declaration.assign(vals_value))

        output = AppendOutput(definition.output_variable, 0)
        source.append(output.write_declarations(kernel_type))

        source.append(to_ir_iteration_graph(graph, output, kernel_type))

        source.append(output.write_cleanup(kernel_type))

        source.append(Return(IntegerLiteral(0)))

    return source
