from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto
from functools import singledispatch
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .identifiable_expression import ast as ie_ast
from .identifiable_expression.to_ir import to_ir
from .iteration_graph import IterationGraph, IterationVariable, TerminalExpression, Add as GraphAdd
from .merge_lattice import LatticeLeaf
from .names import dimension_name, pos_name, crd_name, vals_name, crd_capacity_name, pos_capacity_name, \
    vals_capacity_name, layer_pointer, value_from_crd, bucket_name, previous_layer_pointer, \
    bucket_loop_name
from ..format import Mode
from ..ir.ast import Variable, Multiply, IntegerLiteral, ArrayAllocate, Expression, Return, GreaterThanOrEqual, \
    ArrayReallocate, Max, LessThan, And, Min, BooleanToInteger, Equal, Block, GreaterThan, Branch, Add
from .problem import Problem
from ..ir.builder import SourceBuilder
from ..ir import types

default_array_size = Multiply(IntegerLiteral(1024), IntegerLiteral(1024))


class KernelType(Enum):
    assembly = auto()
    compute = auto()
    evaluate = auto()

    def is_assembly(self):
        return self == KernelType.assembly or self == KernelType.evaluate


class Output:
    @abstractmethod
    def write_assignment(self, right_hand_side: Expression, kernel_type: KernelType) -> SourceBuilder:
        raise NotImplementedError()

    @abstractmethod
    def next_output(self, iteration_output: Optional[LatticeLeaf]) -> Tuple[Output, SourceBuilder]:
        raise NotImplementedError()


@dataclass(frozen=True)
class AppendOutput(Output):
    output: ie_ast.Variable
    next_layer: int

    def vals_pointer(self) -> Expression:
        return previous_layer_pointer(self.output.variable, self.output.order)

    def write_declarations(self, kernel_type: KernelType):
        source = SourceBuilder('Output initialization')

        target_name = self.output.variable.name
        output_tensor = Variable(target_name)

        all_dense = True
        for i, mode in enumerate(self.output.modes):
            if mode == Mode.dense:
                pass
            elif mode == Mode.compressed:
                if kernel_type.is_assembly():
                    # How pos is handled depends on what the previous modes were
                    if all_dense:
                        # If the previous dimensions were all dense, then the size of pos in this dimension is fixed
                        pos_size = Multiply.join(
                            [output_tensor.attr('dimensions').idx(i_prev) for i_prev in range(i)]
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
                    source.append(crd_name(target_name, i).assign(ArrayAllocate(types.integer, crd_capacity)))

                # This is the only thing that get written when doing compute
                source.append(layer_pointer(self.output.variable, i).declare(types.integer).assign(0))

                all_dense = False
            else:
                raise NotImplementedError

        if kernel_type.is_assembly():
            if all_dense:
                vals_size = Multiply.join([output_tensor.attr('dimensions').idx(i) for i in range(self.output.order)])
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

        source.append(vals_name(self.output.variable.name).idx(self.vals_pointer()).assign(right_hand_side))

        return source

    def write_cleanup(self, kernel_type: KernelType):
        source = SourceBuilder(f'Assembling output tensor {self.output.name}')

        if kernel_type.is_assembly():
            target_name = self.output.variable.name
            output_tensor = Variable(target_name)

            for i, mode in enumerate(self.output.modes):
                if mode == Mode.dense:
                    pass
                elif mode == Mode.compressed:
                    source.append(output_tensor.attr('indices').idx(i).idx(0).assign(pos_name(target_name, i)))
                    source.append(output_tensor.attr('indices').idx(i).idx(1).assign(crd_name(target_name, i)))
                else:
                    raise NotImplementedError
            source.append(output_tensor.attr('vals').assign(vals_name(target_name)))

        return source

    def next_output(self, iteration_output: Optional[LatticeLeaf]) -> Tuple[Output, SourceBuilder]:
        if iteration_output is not None and self.next_layer == iteration_output.layer:
            return AppendOutput(self.output, self.next_layer + 1), SourceBuilder()
        else:
            # No layer or wrong layer was encountered
            dense_only_remaining = all(mode == Mode.dense for mode in self.output.modes[self.next_layer:])
            if dense_only_remaining:
                next_output = BucketOutput(self.output, list(range(self.next_layer, len(self.output.modes))))
                dense_product = Multiply.join(next_output.dimension_names())
                bucket = vals_name(self.output.variable.name).plus(self.vals_pointer().times(dense_product))
                return next_output, next_output.write_declarations(bucket)
            else:
                next_output = HashOutput(self.output, self.next_layer)
                return next_output, next_output.write_declarations()


@dataclass(frozen=True)
class HashOutput(Output):
    output: ie_ast.Variable
    starting_layer: int
    unfulfilled: Set[int]

    def __init__(self, output: ie_ast.Variable, starting_layer: int, unfulfilled: Set[int] = None):
        object.__setattr__(self, 'output', output)
        object.__setattr__(self, 'starting_layer', starting_layer)
        if unfulfilled is not None:
            object.__setattr__(self, 'unfulfilled', unfulfilled)
        else:
            object.__setattr__(self, 'unfulfilled', set(range(starting_layer, self.final_dense_index())))

    def final_dense_index(self):
        final_dense_index = self.output.order
        for i in reversed(range(self.starting_layer, self.output.order)):
            if self.output.modes[i] == Mode.compressed:
                break
            else:
                final_dense_index = i

        return final_dense_index

    def write_declarations(self) -> SourceBuilder:
        source = SourceBuilder()

        modes = self.output.modes[self.starting_layer:]
        dim_names = [dimension_name(variable) for variable in self.output.indexes[self.starting_layer:]]

        source.add_dependency('hash', Path(__file__).parent.joinpath('tensora_hash_table.c').read_text())

        source.append(f'hash_table_t {self.name()};')
        source.append(f'taco_mode_t[{len(modes)}] {self.name()}_modes = {{{", ".join(mode.name for mode in modes)}}};')
        source.append(f'uint32_t[{len(modes)}] {self.name()}_dims = {{{", ".join(dim_names)}}};')
        source.append(f'hash_construct(&{self.name()}, {len(modes)}, '
                      f'{self.final_dense_index() - self.starting_layer}, &{self.name()}_modes, '
                      f'&{self.name()}_dims);')
        source.append('')

        return source

    def write_assignment(self, right_hand_side: str, kernel_type: KernelType) -> SourceBuilder:
        raise RuntimeError()

    def write_cleanup(self, kernel_type: KernelType) -> SourceBuilder:
        source = SourceBuilder()

        # Argsort the elements by key
        source.append(f'uint32_t[] {self.name()}_order = malloc(sizeof(uint32_t) * {self.name()}->count);')
        source.append(f'for (uint32_t i = 0; i < {self.name()}->count; i++) {{')
        with source.indented():
            source.append(f'{self.name()}_order[i] = i;')
        source.append('}')
        source.append(f'qsort_r({self.name()}_order, {self.name()}->count, sizeof(uint32_t), hash_comparator, '
                      f'&{self.name()});')

        # Extract indexes recursively
        source.append('i_order = 0;')
        source.include(self.write_layer_cleanup(self.starting_layer, kernel_type))

        # Free temporaries
        source.append(f'free({self.name()}_order);')

        return source

    def write_layer_cleanup(self, layer: int, kernel_type: KernelType):
        source = SourceBuilder()

        if layer < self.final_dense_index():
            layer_index = value_from_crd(self.output.variable, layer)
            dimension_size = dimension_name(self.output.indexes[layer])
            position = layer_pointer(self.output.variable, layer)
            if layer == 0:
                previous_position = '0'
            else:
                previous_position = layer_pointer(self.output.variable, layer - 1)
            end_position = {self.end_position(layer)}
            next_end_position = {self.end_position(layer + 1)}

            # Reusable search code
            search_source = SourceBuilder()
            search_source.append(f'uint32_t {next_end_position} = 0;')
            search_source.append(f'while ({next_end_position} < {end_position}) {{')
            with search_source.indented():
                search_source.append(f'if ({self.name()}->keys[{self.name()}_order[i_order]][{layer}] '
                                     f'!= {layer_index}) {{')
                with search_source.indented():
                    search_source.append('break;')
                search_source.append('}')
                search_source.append(f'{next_end_position}++;')
            search_source.append('}')

            # Keys phase
            if self.output.modes[layer] == Mode.dense:
                source.append(f'for (uint32_t {layer_index} = 0; {layer_index} < {dimension_size}; {layer_index}++)')
                with source.indented():
                    source.append(f'uint32_t {position} = {previous_position} + {layer_index};')
                    source.include(search_source)
                    source.include(self.write_layer_cleanup(layer + 1, kernel_type))

            elif self.output.modes[layer] == Mode.compressed:
                source.append(f'while (i_order < {end_position}) {{')
                with source.indented():
                    source.append(f'uint23_t {layer_index} = '
                                  f'{self.name()}->keys[{self.name()}_order[i_order]][{layer}];')
                    source.include(search_source)
                    source.include(self.write_layer_cleanup(layer + 1, kernel_type))

                if kernel_type.is_assembly():
                    source.include(write_crd_assembly(LatticeLeaf(self.output, layer)))

            if kernel_type.is_assembly():
                source.include(write_pos_assembly(LatticeLeaf(self.output, layer)))

            if layer == self.final_dense_index() - 1:
                with source.indented():
                    source.append(f'i_order++')
            source.append('}')  # end loop
        elif layer < self.output.order:
            # Bucket phase
            layer_index = value_from_crd(self.output.variable, layer)
            dimension_size = dimension_name(self.output.indexes[layer])
            position = layer_pointer(self.output.variable, layer)
            bucket_position = f'{position}_bucket'
            if layer == 0:
                previous_position = '0'
                previous_bucket_position = '0'
            else:
                previous_position = layer_pointer(self.output.variable, layer - 1)
                previous_bucket_position = f'{previous_position}_bucket'

            source.append(f'for (uint32_t {layer_index} = 0; {layer_index} < {dimension_size}; {layer_index}++) {{')
            with source.indented():
                source.append(f'uint32_t {position} = {previous_position} + {layer_index};')
                source.append(f'uint32_t {bucket_position} = {previous_bucket_position} + {layer_index};')
                source.include(self.write_layer_cleanup(layer + 1, kernel_type))
            source.append('}')
        elif layer == self.output.order:
            # Final phase
            vals = vals_name(self.output.variable.name)
            if layer == 0:
                previous_position = '0'
                previous_bucket_position = '0'
            else:
                previous_position = layer_pointer(self.output.variable, layer - 1)
                previous_bucket_position = f'{previous_position}_bucket'
            bucket = f'{self.name()}_bucket'
            source.append(f'{vals}[{previous_position}] = {bucket}[{previous_bucket_position}];')

        return source

    def next_output(self, iteration_output: Optional[LatticeLeaf]) -> Tuple[Output, SourceBuilder]:
        if iteration_output is None:
            return self, SourceBuilder()
        else:
            next_unfulfilled = self.unfulfilled - {iteration_output.layer}
            if len(next_unfulfilled) == 0:
                final_dense_index = self.final_dense_index()

                next_output = BucketOutput(self.output, list(range(final_dense_index, self.output.order)))

                # Write declaration of bucket
                source = SourceBuilder()

                key_names = [value_from_crd(self.output.variable, layer)
                             for layer in range(self.starting_layer, final_dense_index)]

                source.append(f'uint32_t[] {self.name()}_key = {{{", ".join(key_names)}}};')
                source.include(next_output.write_declarations(f'hash_get_bucket(&{self.name()}, &{self.name()}_key)'))
                source.append('')

                return next_output, source
            else:
                return replace(self, unfulfilled=next_unfulfilled), SourceBuilder()

    def name(self):
        return f'hash_table_{layer_pointer(self.output.variable, self.starting_layer)}'

    def start_position(self, layer: int):
        if layer < 0:
            return '0'
        else:
            return f'p_{self.name()}_{layer}_start'

    def end_position(self, layer: int):
        if layer < 0:
            return f'{self.name()}->count'
        else:
            return f'p_{self.name()}_{layer}_start'


@dataclass(frozen=True)
class BucketOutput(Output):
    output: ie_ast.Variable
    layers: List[int]
    unfulfilled: Set[int]

    def __init__(self, output: ie_ast.Variable, layers: List[int], unfulfilled: Set[int] = None):
        object.__setattr__(self, 'output', output)
        object.__setattr__(self, 'layers', layers)
        if unfulfilled is not None:
            object.__setattr__(self, 'unfulfilled', unfulfilled)
        else:
            unfulfilled = {layer for layer in layers if output.modes[layer] == Mode.compressed}
            object.__setattr__(self, 'unfulfilled', unfulfilled)

    def write_declarations(self, right_hand_side: Expression) -> SourceBuilder:
        source = SourceBuilder('Bucket initialization')
        source.append(self.name().declare(types.Pointer(types.float)).assign(right_hand_side))
        bucket_loop_index = bucket_loop_name(self.output.variable, self.layers)
        source.append(bucket_loop_index.declare(types.integer).assign(0))
        with source.loop(LessThan(bucket_loop_index, Multiply.join(self.dimension_names()))):
            source.append(self.name().idx(bucket_loop_index).assign(0))
            source.append(bucket_loop_index.increment())
        return source

    def next_output(self, iteration_output: Optional[LatticeLeaf]) -> Tuple[Output, SourceBuilder]:
        if iteration_output is None:
            return self, SourceBuilder()
        else:
            next_unfulfilled = self.unfulfilled - {iteration_output.layer}
            return replace(self, unfulfilled=next_unfulfilled), SourceBuilder()

    def write_assignment(self, right_hand_side: Expression, kernel_type: KernelType):
        source = SourceBuilder()
        bucket_index = self.ravel_indexes(
            self.dimension_names(),
            [layer_pointer(self.output.variable, layer) for layer in self.layers],
        )
        source.append(self.name().idx(bucket_index).increment(right_hand_side))
        return source

    def ravel_indexes(self, dimensions: List[Variable], indexes: List[Variable]):
        dimensions_so_far: List[Variable] = []
        terms: List[Expression] = []
        for dim_i, index_i in zip(dimensions, indexes):
            terms.append(Multiply.join([index_i] + dimensions_so_far))
            dimensions_so_far.append(dim_i)

        return Add.join(terms)

    def write_cleanup(self, kernel_type: KernelType):
        return SourceBuilder()

    def name(self) -> Variable:
        return bucket_name(self.output.variable, self.layers)

    def dimension_names(self):
        return [dimension_name(self.output.indexes[layer]) for layer in self.layers]


@singledispatch
def iteration_graph_to_c_code(graph: IterationGraph, output: Output, kernel_type: KernelType) -> SourceBuilder:
    raise NotImplementedError()


def generate_subgraphs(graph: IterationVariable) -> List[IterationVariable]:
    # The 0th element is just the full lattice
    # Each element is derived from a previous element by zeroing a tensor
    # Zeroing a tensor always results in a strictly smaller lattice
    # Zeroing a tensor will cause tensors multiplied by it to be zeroed as well
    # This means that zeroing two different tensors can result in the same lattice
    # Lattices can be keyed by their set of remaining sparse tensors to eliminate duplicates
    all_lattices = {graph.lattice.compressed_dimensions(): graph}
    old_lattices = {graph.lattice.compressed_dimensions(): graph}

    while True:
        new_lattices = {}

        for old_graph in old_lattices.values():
            sparse_dimensions = old_graph.lattice.compressed_dimensions()

            # Reverse sparse_dimensions so that last values dropped first
            for sparse_dimension in reversed(sparse_dimensions):
                new_graph = old_graph.exhaust_tensor(sparse_dimension)

                if new_graph is not None:
                    new_lattices[new_graph.lattice.compressed_dimensions()] = new_graph

        if len(new_lattices) == 0:
            break
        else:
            all_lattices.update(new_lattices)
            old_lattices = new_lattices

    return list(all_lattices.values())


def write_sparse_initialization(leaf: LatticeLeaf):
    source = SourceBuilder()

    index_variable = leaf.layer_pointer()
    start_index = leaf.previous_layer_pointer()
    end_index = leaf.sparse_end_name()
    pos_array = leaf.pos_name()

    source.append(index_variable.declare(types.integer).assign(pos_array.idx(start_index)))
    source.append(end_index.declare(types.integer).assign(pos_array.idx(start_index.plus(1))))

    return source


def write_crd_assembly(output: LatticeLeaf):
    source = SourceBuilder('crd assembly')

    pointer = output.layer_pointer()
    capacity = output.crd_capacity_name()
    crd = output.crd_name()
    loop_variable = Variable(output.tensor.indexes[output.layer])

    with source.branch(GreaterThanOrEqual(pointer, capacity)):
        source.append(capacity.assign(capacity.times(2)))
        source.append(crd.assign(ArrayReallocate(crd, types.integer, capacity)))

    source.append(crd.idx(pointer).assign(loop_variable))

    return source


def write_pos_assembly(output: LatticeLeaf) -> SourceBuilder:
    source = SourceBuilder('pos assembly')

    pointer = output.layer_pointer()
    pos = output.pos_name()
    previous_pointer = output.previous_layer_pointer()

    source.append(pos.idx(previous_pointer.plus(1)).assign(pointer))

    return source


def write_pos_allocation(output: LatticeLeaf):
    dense_dimensions = []
    for i_layer in range(output.layer):
        index_variable_i = output.tensor.indexes[i_layer]
        mode_i = output.tensor.modes[i_layer]
        if mode_i == Mode.compressed:
            break
        dense_dimensions.append(dimension_name(index_variable_i))

    layer_being_allocated = output.layer + len(dense_dimensions) + 1
    if layer_being_allocated == len(output.tensor.indexes):
        comment = 'vals allocation'
        capacity = output.vals_capacity_name()
        array = output.vals_name()
        type = types.float
        bonus = 0
    else:
        comment = 'pos allocation for next sparse layer'
        target_leaf = LatticeLeaf(output.tensor, output.layer + len(dense_dimensions) + 1)
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
        minimum_capacity = output.layer_pointer().plus(1).times(Multiply.join(dense_dimensions)).plus(bonus)

        with source.branch(GreaterThanOrEqual(minimum_capacity, capacity)):
            source.append(capacity.assign(Max(capacity.times(2), minimum_capacity)))
            source.append(array.assign(ArrayReallocate(array, type, capacity)))

    return source


@iteration_graph_to_c_code.register(IterationVariable)
def iteration_variable_to_c_code(graph: IterationVariable, output: Output, kernel_type: KernelType):
    source = SourceBuilder(f'*** Iteration over {graph.index_variable} ***')

    loop_variable = Variable(graph.index_variable)

    # If this node is_dense, then every index needs to be iterated over
    is_dense = graph.lattice.is_dense()

    # Compute the next output
    next_output, next_output_declarations = output.next_output(graph.output)
    source.append(next_output_declarations)

    ##################
    # Initialization #
    ##################
    if is_dense:
        source.append(loop_variable.declare(types.integer).assign(0))

    for leaf in graph.lattice.sparse_leaves():
        source.append(write_sparse_initialization(leaf))

    ############
    # Subnodes #
    ############
    subnodes = generate_subgraphs(graph)
    for subnode in subnodes:
        sparse_subnode_leaves = subnode.lattice.sparse_leaves()
        dense_subnode_leaves = subnode.lattice.dense_leaves()

        ########
        # Loop #
        ########
        if len(sparse_subnode_leaves) == 0:
            # If there are no sparse dimensions, the only thing to stop iteration is the dense dimension
            while_criteria = LessThan(loop_variable, dimension_name(graph.index_variable))
        else:
            while_criteria = And.join([LessThan(leaf.layer_pointer(), leaf.sparse_end_name())
                                       for leaf in sparse_subnode_leaves])

        with source.loop(while_criteria):
            ##########################
            # Extract sparse indexes #
            ##########################
            index_variables = []
            for leaf in sparse_subnode_leaves:
                index_variable = leaf.value_from_crd()
                index_variables.append(index_variable)
                source.append(index_variable.declare(types.integer).assign(leaf.crd_name().idx(leaf.layer_pointer())))

            # Print closest index
            if not is_dense:
                source.append(loop_variable.declare(types.integer).assign(Min.join(index_variables)))

            #########################
            # Compute dense indexes #
            #########################
            maybe_dense_output = [graph.output] if graph.output is not None else []
            for leaf in maybe_dense_output + dense_subnode_leaves:
                pointer = leaf.layer_pointer()
                previous_pointer = leaf.previous_layer_pointer()
                pointer_value = previous_pointer.times(dimension_name(graph.index_variable)).plus(loop_variable)
                source.append(pointer.declare(types.integer).assign(pointer_value))

            ###############
            # Subsubnodes #
            ###############
            subsubnodes = generate_subgraphs(subnode)
            subsubnode_leaves: List[Tuple[Expression, Block]] = []
            for i, subsubnode in enumerate(subsubnodes):
                ############################
                # Branch on sparse matches #
                ############################
                sparse_subsubnode_leaves = subsubnode.lattice.sparse_leaves()
                condition = And.join([Equal(leaf.value_from_crd(), loop_variable) for leaf in sparse_subsubnode_leaves])
                block = SourceBuilder()

                ###################################
                # Allocate space for pos and vals #
                ###################################
                if kernel_type.is_assembly() and graph.is_sparse_output():
                    block.append(write_pos_allocation(graph.output))

                ################################
                # Store position of next layer #
                ################################
                if kernel_type.is_assembly() and graph.is_sparse_output() and graph.next.is_sparse_output():
                    with block.block("Save next layer's position"):
                        next_pointer = graph.next.output.layer_pointer()
                        next_pointer_begin = graph.next.output.layer_begin_name()
                        block.append(next_pointer_begin.assign(next_pointer))

                #####################
                # Invoke next layer #
                #####################
                block.append(iteration_graph_to_c_code(subsubnode.next, next_output, kernel_type))

                ########################################################
                # Write sparse output index and advance output pointer #
                ########################################################
                if graph.is_sparse_output():
                    if graph.next.is_sparse_output():
                        next_pointer = graph.next.output.layer_pointer()
                        next_pointer_begin = graph.next.output.layer_begin_name()
                        with block.branch(GreaterThan(next_pointer, next_pointer_begin)):
                            if kernel_type.is_assembly():
                                block.append(write_crd_assembly(graph.output))
                            block.append(graph.output.layer_pointer().increment())
                    else:
                        if kernel_type.is_assembly():
                            block.append(write_crd_assembly(graph.output))
                        block.append(graph.output.layer_pointer().increment())

                subsubnode_leaves.append((condition, block.finalize()))

            source.append(Branch.join(subsubnode_leaves))

            #######################
            # Increment positions #
            #######################
            for leaf in sparse_subnode_leaves:
                source.append(leaf.layer_pointer().increment(
                    BooleanToInteger(Equal(leaf.value_from_crd(), loop_variable))
                ))
            if graph.lattice.is_dense():
                source.append(loop_variable.increment())

    ################################
    # Write sparse output position #
    ################################
    if kernel_type.is_assembly() and graph.is_sparse_output():
        source.append(write_pos_assembly(graph.output))

    return source


@iteration_graph_to_c_code.register(GraphAdd)
def add_node_to_c_code(graph: GraphAdd, output: Output, kernel_type: KernelType):
    source = SourceBuilder('*** Add ***')

    next_output, next_output_declarations = output.next_output(None)
    source.append(next_output_declarations)

    for term in graph.terms:
        source.append(iteration_graph_to_c_code(term, next_output, kernel_type))

    return source


@iteration_graph_to_c_code.register(TerminalExpression)
def to_c_code_terminal_expression(graph: TerminalExpression, output: Output, kernel_type: KernelType):
    source = SourceBuilder('*** Computation of expression ***')

    source.append(output.write_assignment(to_ir(graph.expression), kernel_type))

    return source


def generate_c_code(problem: Problem, graph: IterationGraph, kernel_type: KernelType):
    source = SourceBuilder()

    target_name = problem.assignment.target.name

    tensor_names = [target_name] + list(problem.input_formats.keys())

    # Function declaration
    parameters = {name: types.Pointer(types.tensor) for name in tensor_names}
    with source.function_definition('evaluate', parameters, types.integer):
        # Dimensions of all index variables
        with source.block('Extract dimensions'):
            for index_name, (tensor_name, tensor_dimension) in problem.index_dimensions().items():
                declaration = dimension_name(index_name).declare(types.integer)
                value = Variable(tensor_name).attr('dimensions').idx(tensor_dimension)
                source.append(declaration.assign(value))

        # Unpack tensors
        with source.block('Unpack tensors'):
            for tensor_name, format in {target_name: problem.output_format, **problem.input_formats}.items():
                for i, mode in enumerate(format.modes):
                    if mode == Mode.dense:
                        pass
                    elif mode == Mode.compressed:
                        pos_declaration = pos_name(tensor_name, i).declare(types.Pointer(types.integer))
                        pos_value = Variable(tensor_name).attr('indices').idx(i).idx(0)
                        source.append(pos_declaration.assign(pos_value))
                        crd_declaration = crd_name(tensor_name, i).declare(types.Pointer(types.integer))
                        crd_value = Variable(tensor_name).attr('indices').idx(i).idx(1)
                        source.append(crd_declaration.assign(crd_value))
                    else:
                        raise NotImplementedError
                vals_declaration = vals_name(tensor_name).declare(types.Pointer(types.float))
                vals_value = Variable(tensor_name).attr('vals')
                source.append(vals_declaration.assign(vals_value))

        output = AppendOutput(problem.assignment.target, 0)
        source.append(output.write_declarations(kernel_type))

        source.append(iteration_graph_to_c_code(graph, output, kernel_type))

        source.append(output.write_cleanup(kernel_type))

        source.append(Return(IntegerLiteral(0)))

    return source
