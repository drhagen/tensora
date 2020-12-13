from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from enum import Enum, auto
from functools import singledispatch
from pathlib import Path
from typing import List, Optional, Set, Tuple

from .identifiable_expression.ast import Tensor
from .iteration_graph import IterationGraph, IterationVariable, TerminalExpression, Add
from .identifiable_expression import to_c_code
from .merge_lattice import LatticeLeaf
from .names import dimension_name, pos_name, crd_name, vals_name, crd_capacity_name, pos_capacity_name, \
    vals_capacity_name, layer_pointer, value_from_crd, bucket_name
from ..format import Mode
from .problem import Problem
from ..source_builder import SourceBuilder


class KernelType(Enum):
    assembly = auto()
    compute = auto()
    evaluate = auto()

    def is_assembly(self):
        return self == KernelType.assembly or self == KernelType.evaluate


class Output:
    @abstractmethod
    def write_assignment(self, right_hand_side: str, kernel_type: KernelType) -> SourceBuilder:
        raise NotImplementedError()

    @abstractmethod
    def next_output(self, iteration_output: Optional[LatticeLeaf]) -> Tuple[Output, SourceBuilder]:
        raise NotImplementedError()


@dataclass(frozen=True)
class AppendOutput(Output):
    output: Tensor
    next_layer: int

    def vals_pointer(self):
        return layer_pointer(self.output.variable, len(self.output.indexes) - 1)

    def write_declarations(self, kernel_type: KernelType):
        source = SourceBuilder()

        target_name = self.output.variable.name

        if kernel_type.is_assembly():
            all_dense = True
            for i, mode in enumerate(self.output.modes):
                if mode == Mode.dense:
                    pass
                elif mode == Mode.compressed:
                    # How pos is handled depends on what the previous modes were
                    if all_dense:
                        # If the previous dimensions were all dense, then the size of pos in this dimension is fixed
                        if i == 0:
                            nnz_string = '1'
                        else:
                            nnz_string = " * ".join(f"{target_name}->dimensions[{i_prev}]" for i_prev in range(i))

                        capacity = f'({nnz_string} + 1)'
                    else:
                        capacity = pos_capacity_name(target_name, i)
                        source.append(f'int32_t {capacity} = 1024*1024;')
                    source.append(f'{pos_name(target_name, i)} = (int32_t*)malloc(sizeof(int32_t) * {capacity});')
                    source.append(f'{pos_name(target_name, i)}[0] = 0;')

                    capacity = crd_capacity_name(target_name, i)
                    source.append(f'int32_t {capacity} = 1024*1024;')
                    source.append(f'{crd_name(target_name, i)} = (int32_t*)malloc(sizeof(int32_t) * {capacity}')
                    source.append(f'int32_t {layer_pointer(self.output.variable, i)} = 0;')

                    all_dense = False
                else:
                    raise NotImplementedError

            if all_dense:
                vals_size = ' * '.join(f'{target_name}->dimensions[{i}]' for i in range(self.output.order))
            else:
                vals_size = '1024*1024'
            vals_capacity = vals_capacity_name(target_name)
            source.append(f'int32_t {vals_capacity} = {vals_size};')
            source.append(f'{target_name}_vals = (double*)malloc(sizeof(double) * {vals_capacity});')
        else:
            # Provide the vals pointer even when not doing assembly
            source.append(f'int32_t {self.vals_pointer()} = 0;')

        return source

    def write_assignment(self, right_hand_side: str, kernel_type: KernelType):
        source = SourceBuilder()

        if self.next_layer != self.output.order:
            raise RuntimeError()

        source.append(f'{vals_name(self.output.variable.name)}'
                      f'[{layer_pointer(self.output.variable, self.next_layer)}] = {right_hand_side};')

        return source

    def write_cleanup(self, kernel_type: KernelType):
        source = SourceBuilder()

        target_name = self.output.variable.name

        if kernel_type.is_assembly():
            for i, mode in enumerate(self.output.modes):
                if mode == Mode.dense:
                    pass
                elif mode == Mode.compressed:
                    source.append(f'{target_name}->indices[{i}][0] = (unit8_t*){pos_name(target_name, i)};')
                    source.append(f'{target_name}->indices[{i}][1] = (unit8_t*){crd_name(target_name, i)};')
                else:
                    raise NotImplementedError
            source.append(f'{target_name}->vals = (uint8_t*){vals_name(target_name)};')

        return source

    @abstractmethod
    def next_output(self, iteration_output: Optional[LatticeLeaf]) -> Tuple[Output, SourceBuilder]:
        if iteration_output is None:
            next_output = HashOutput(self.output, self.next_layer)
            return next_output, next_output.write_declarations()
        elif self.next_layer == iteration_output.layer:
            return AppendOutput(self.output, self.next_layer + 1), SourceBuilder()
        else:
            # Wrong layer was encountered
            dense_only_remaining = all(mode == Mode.dense for mode in self.output.modes[self.next_layer:])
            if dense_only_remaining:
                next_output = BucketOutput(self.output, list(range(self.next_layer, len(self.output.modes))))
            else:
                next_output = HashOutput(self.output, self.next_layer)
            return next_output, next_output.write_declarations()


@dataclass(frozen=True)
class HashOutput(Output):
    output: Tensor
    starting_layer: int
    unfulfilled: Set[int]

    def __init__(self, output: Tensor, starting_layer: int, unfulfilled: Set[int] = None):
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

                key_names = [value_from_crd(self.output.name, layer)
                             for layer in range(self.starting_layer, final_dense_index)]

                source.append(f'uint32_t[] {self.name()}_key = {{{", ".join(key_names)}}};')
                source.include(next_output.write_declarations(f'hash_get_bucket(&{self.name()}, &{self.name()}_key)'))
                source.append('')

                return next_output, source
            else:
                return replace(self, unfulfilled=next_unfulfilled), SourceBuilder()

    def name(self):
        return f'hash_table_{layer_pointer(self.output.variable.name, self.starting_layer)}'

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
    output: Tensor
    layers: List[int]
    unfulfilled: Set[int]

    def __init__(self, output: Tensor, layers: List[int], unfulfilled: Set[int] = None):
        object.__setattr__(self, 'output', output)
        object.__setattr__(self, 'layers', layers)
        if unfulfilled is not None:
            object.__setattr__(self, 'unfulfilled', unfulfilled)
        else:
            unfulfilled = {layer for layer in layers if output.modes[layer] == Mode.compressed}
            object.__setattr__(self, 'unfulfilled', unfulfilled)

    def write_declarations(self, right_hand_side: str) -> SourceBuilder:
        source = SourceBuilder()
        source.append(f'double[] {self.name()} = {right_hand_side};')
        return source

    def next_output(self, iteration_output: Optional[LatticeLeaf]) -> Tuple[Output, SourceBuilder]:
        if iteration_output is None:
            return self, SourceBuilder()
        else:
            next_unfulfilled = self.unfulfilled - {iteration_output.layer}
            return replace(self, unfulfilled=next_unfulfilled), SourceBuilder()

    def write_assignment(self, right_hand_side: str, kernel_type: KernelType):
        source = SourceBuilder()
        source.include(self.ravel_indexes(
            [dimension_name(self.output.indexes[layer]) for layer in self.layers],
            [layer_pointer(self.output.variable.name, layer) for layer in self.layers],
            self.position_name(),
        ))
        source.append(f'{self.name()}[{self.position_name()}] += {right_hand_side};')
        return source

    def ravel_indexes(self, dimensions: List[str], indexes: List[str], index: str):
        source = SourceBuilder()
        dimensions_so_far = []
        cumulative_sum = []
        for dim_i, index_i in zip(dimensions, indexes):
            cumulative_sum.append(' * '.join([index_i] + dimensions_so_far))
            dimensions_so_far.append(dim_i)

        if len(cumulative_sum) == 0:
            ravel_index = '0'
        else:
            ravel_index = ' + '.join(cumulative_sum)
        source.append(f'int32_t {index} = {ravel_index};')

        return source

    def write_cleanup(self, kernel_type: KernelType):
        return SourceBuilder()

    def name(self):
        return bucket_name(self.output.variable.name, self.layers)

    def position_name(self):
        return f'p_{self.name()}'


@singledispatch
def iteration_graph_to_c_code(graph: IterationGraph, output: Output, kernel_type: KernelType) -> SourceBuilder:
    raise NotImplementedError()


def minimum_expression(expressions: List[str]):
    if len(expressions) == 0:
        raise ValueError('Cannot take minimum of empty array')
    elif len(expressions) == 1:
        return expressions[0]
    else:
        return f'TACO_MIN({expressions[0]}, {minimum_expression(expressions[1:])})'


def maximum_expression(expressions: List[str]):
    if len(expressions) == 0:
        raise ValueError('Cannot take maximum of empty array')
    elif len(expressions) == 1:
        return expressions[0]
    else:
        return f'TACO_MAX({expressions[0]}, {maximum_expression(expressions[1:])})'


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

    if start_index == '0':
        end_index = '1'
    else:
        end_index = f'{start_index}+1'

    source.append(f'int32_t {index_variable} = {leaf.pos_name()}[{start_index}];')
    source.append(f'int32_t {index_variable}_end = {leaf.pos_name()}[{end_index}];')

    return source


def write_crd_assembly(output: LatticeLeaf):
    source = SourceBuilder()

    pointer = output.layer_pointer()
    capacity = output.crd_capacity_name()
    crd = output.crd_name()
    index_variable = output.tensor.indexes[output.layer]

    source.append(f'if ({pointer} >= {capacity}) {{')
    with source.indented():
        source.append(f'{capacity} *= 2;')
        source.append(f'{crd} = (int32_t*)realloc({crd}, sizeof(int32_t) * {capacity});')
    source.append('}')

    source.append(f'{crd}[{pointer}] = {index_variable};')
    source.append(f'{pointer}++;')

    return source


def write_pos_assembly(output: LatticeLeaf) -> SourceBuilder:
    source = SourceBuilder()
    pointer = output.layer_pointer()
    pos = output.pos_name()
    previous_pointer = output.previous_layer_pointer()

    if previous_pointer == '0':
        end_index = '1'
    else:
        end_index = f'{previous_pointer}+1'

    source.append(f'{pos}[{end_index}] = {pointer};')

    return source


def write_pos_allocation(output: LatticeLeaf):
    source = SourceBuilder()

    dense_dimensions = []
    for i_layer in range(output.layer):
        index_variable_i = output.tensor.indexes[i_layer]
        mode_i = output.tensor.modes[i_layer]
        if mode_i == Mode.compressed:
            break
        dense_dimensions.append(dimension_name(index_variable_i))

    layer_being_allocated = output.layer + len(dense_dimensions) + 1
    if layer_being_allocated == len(output.tensor.indexes):
        capacity = output.vals_capacity_name()
        array = output.vals_name()
        type = 'double'
        bonus = ''
    else:
        target_leaf = LatticeLeaf(output.tensor, output.layer + len(dense_dimensions) + 1)
        capacity = target_leaf.pos_capacity_name()
        array = target_leaf.pos_name()
        type = 'int32_t'
        bonus = ' + 1'

    if len(dense_dimensions) == 0:
        source.append(f'if ({output.layer_pointer()}{bonus} >= {capacity}) {{')
        with source.indented():
            source.append(f'{capacity} *= 2;')
            source.append(f'{array} = ({type}*)realloc({array}, sizeof({type}) * {capacity});')
        source.append('}')
    else:
        double = f'{capacity} * 2'
        required = f'({output.layer_pointer()} + 1) * {" * ".join(dense_dimensions)}{bonus}'
        source.append(f'if ({required} >= {capacity}) {{')
        with source.indented():
            source.append(f'{capacity} = {maximum_expression([double, required])};')
            source.append(f'{array} = ({type}*)realloc({array}, sizeof({type}) * {capacity});')
        source.append('}')

    return source


@iteration_graph_to_c_code.register(IterationVariable)
def iteration_variable_to_c_code(graph: IterationVariable, output: Output, kernel_type: KernelType):
    source = SourceBuilder()

    # If this node is_dense, then every index needs to be iterated over
    is_dense = graph.lattice.is_dense()

    # Compute the next output
    next_output, next_output_declarations = output.next_output(graph.output)
    source.include(next_output_declarations)

    ##################
    # Initialization #
    ##################
    if is_dense:
        source.append(f'int32_t {graph.index_variable} = 0;')

    for leaf in graph.lattice.sparse_leaves():
        source.include(write_sparse_initialization(leaf))

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
            while_criteria = [f'{graph.index_variable} < {graph.index_variable}_dim']
        else:
            while_criteria = [f'{leaf.layer_pointer()} < {leaf.layer_pointer()}_end' for leaf in sparse_subnode_leaves]
        source.append(f'while ({" && ".join(while_criteria)}) {{')

        with source.indented():
            ##########################
            # Extract sparse indexes #
            ##########################
            index_variables = []
            for leaf in sparse_subnode_leaves:
                index_variable = leaf.value_from_crd()
                index_variables.append(index_variable)
                source.append(f'int32_t {index_variable} = {leaf.crd_name()}[{leaf.layer_pointer()}];')

            # Print closest index
            if not is_dense:
                source.append(f'int32_t {graph.index_variable} = {minimum_expression(index_variables)};')

            #########################
            # Compute dense indexes #
            #########################
            for leaf in dense_subnode_leaves:
                pointer = leaf.layer_pointer()
                previous_pointer = leaf.previous_layer_pointer()
                if previous_pointer == '0':
                    source.append(f'int32_t {pointer} = {graph.index_variable};')
                else:
                    source.append(f'int32_t {pointer} = {previous_pointer} * {dimension_name(graph.index_variable)};')
            if len(dense_subnode_leaves) > 0:
                source.append('')

            ###############
            # Subsubnodes #
            ###############
            subsubnodes = generate_subgraphs(subnode)
            for i, subsubnode in enumerate(subsubnodes):
                sparse_subsubnode_leaves = subsubnode.lattice.sparse_leaves()

                ###################################
                # Allocate space for pos and vals #
                ###################################
                if kernel_type.is_assembly() and graph.is_sparse_output():
                    source.include(write_pos_allocation(graph.output))

                ############################
                # Branch on sparse matches #
                ############################
                if len(sparse_subsubnode_leaves) == 0:
                    # Elide branch if there are no sparse layers
                    source.include(iteration_graph_to_c_code(subsubnode.next, next_output, kernel_type))
                else:
                    equalities = [f'{leaf.value_from_crd()} == {graph.index_variable}'
                                  for leaf in sparse_subsubnode_leaves]
                    if i == 0:
                        if_statement = f'if ({" && ".join(equalities)}) {{'
                    elif len(equalities) != 0:
                        if_statement = f'else if ({" && ".join(equalities)}) {{'
                    else:
                        if_statement = 'else {'
                    source.append(if_statement)

                    with source.indented():
                        ################################
                        # Store position of next layer #
                        ################################
                        if kernel_type.is_assembly() and graph.is_sparse_output() and graph.next.is_sparse_output():
                            next_pointer = graph.next.output.layer_pointer()
                            source.append(f'int32_t {next_pointer}_begin = {next_pointer};')

                        #####################
                        # Invoke next layer #
                        #####################
                        source.include(iteration_graph_to_c_code(subsubnode.next, next_output, kernel_type))

                        ##########################
                        # Advance output pointer #
                        ##########################
                        if kernel_type.is_assembly() and graph.is_sparse_output():
                            if graph.next.is_sparse_output():
                                next_pointer = graph.next.output.layer_pointer()
                                source.append(f'if ({next_pointer} > {next_pointer}_begin) {{')
                                with source.indented():
                                    source.include(write_crd_assembly(graph.output))
                                source.append('}')
                            else:
                                source.include(write_crd_assembly(graph.output))
                        elif graph.is_sparse_output() \
                                and (not isinstance(graph.next, IterationVariable) or graph.next.output is None):
                            source.append(f'{graph.output.layer_pointer()}++;')

                    source.append('}')  # End branch

            #######################
            # Increment positions #
            #######################
            for leaf in sparse_subnode_leaves:
                source.append(f'{leaf.layer_pointer()} '
                              f'+= (int32_t)({leaf.value_from_crd()} == {graph.index_variable});')
            if graph.lattice.is_dense():
                source.append(f'{graph.index_variable}++;')

        source.append('}')  # End loop

    #########################
    # Write output position #
    #########################
    if kernel_type.is_assembly() and graph.is_sparse_output():
        source.include(write_pos_assembly(graph.output))

    return source


@iteration_graph_to_c_code.register(Add)
def add_node_to_c_code(graph: Add, output: Output, kernel_type: KernelType):
    new_output = AccumulatorOutput(graph.name)
    source = SourceBuilder()
    source.include(new_output.write_declarations())
    source.append('')

    for term in graph.terms:
        source.include(iteration_graph_to_c_code(term, new_output, kernel_type))
        source.append('')

    source.include(output.write_assignment('output'))
    source.append('')

    return source


@iteration_graph_to_c_code.register(TerminalExpression)
def to_c_code_terminal_expression(graph: TerminalExpression, output: Output, kernel_type: KernelType):
    source = SourceBuilder()

    source.include(output.write_assignment(to_c_code(graph.expression), kernel_type))
    return source


def generate_c_code(problem: Problem, graph: IterationGraph, kernel_type: KernelType):
    source = SourceBuilder()

    target_name = problem.assignment.target.name

    tensor_names = [target_name] + list(problem.input_formats.keys())

    # Function declaration
    source.append(f"int evaluate({', '.join(f'taco_tensor_t *{name}' for name in tensor_names)}) {{")

    with source.indented():
        # Dimensions of all index variables
        for name, dimension in problem.index_dimensions().items():
            source.append(f'int32_t {dimension_name(name)} = {dimension[0]}->dimensions[{dimension[1]}];')

        # Unpack tensors
        for name, format in {target_name: problem.output_format, **problem.input_formats}.items():
            for i, mode in enumerate(format.modes):
                if mode == Mode.dense:
                    pass
                elif mode == Mode.compressed:
                    source.append(f'int32_t* restrict {pos_name(name, i)} = (int32_t*)({name}->indices[{i}][0]);')
                    source.append(f'int32_t* restrict {crd_name(name, i)} = (int32_t*)({name}->indices[{i}][1]);')
                else:
                    raise NotImplementedError
            source.append(f'double* restrict {vals_name(name)} = (double*)({name}->vals);')

        source.append('')

        output = AppendOutput(problem.assignment.target, 0)
        source.include(output.write_declarations(kernel_type))
        source.append('')

        source.include(iteration_graph_to_c_code(graph, output, kernel_type))
        source.append('')

        source.include(output.write_cleanup(kernel_type))
        source.append('')

        source.append('return 0;')

    source.append('}')  # end function

    return source
