from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from functools import singledispatch
from typing import List

from .iteration_graph import IterationGraph, IterationVariable, TerminalExpression, Contract, Add, Scratch
from .identifiable_expression import to_c_code, layer_pointer
from .merge_lattice import LatticeLeaf
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
    def write_declarations(self):
        raise NotImplementedError()

    @abstractmethod
    def write_assignment(self, right_hand_side: str):
        raise NotImplementedError()


@dataclass(frozen=True)
class TensorOutput(Output):
    output: LatticeLeaf

    def write_declarations(self):
        source = SourceBuilder()
        source.append(f'int32_t {self.output.layer_pointer()} = 0;')
        return source

    def write_assignment(self, right_hand_side: str):
        source = SourceBuilder()
        source.append(f'{self.output.vals_name()}[{self.output.layer_pointer()}] = {right_hand_side};')
        source.append(f'{self.output.layer_pointer()}++;')
        return source


@dataclass(frozen=True)
class AccumulatorOutput(Output):
    name: str

    def write_declarations(self):
        source = SourceBuilder()
        source.append(f'double {self.name} = 0.0;')
        return source

    def write_assignment(self, right_hand_side: str):
        source = SourceBuilder()
        source.append(f'{self.name} += {right_hand_side};')
        return source


@dataclass(frozen=True)
class ScratchOutput(Output):
    layers: List[LatticeLeaf]

    def write_declarations(self):
        return SourceBuilder()

    def write_assignment(self, right_hand_side: str):
        source = SourceBuilder()
        source.include(self.ravel_indexes([layer.layer_pointer() for layer in self.layers], self.sums_position_name()))
        source.append(f'{self.sums_name()}[{self.sums_position_name()}] += {right_hand_side};')

        source.append(f'if ({self.next_name()}[{self.sums_position_name()}] < 0) {{')
        with source.indented():
            source.append(f'{self.next_name()}[{self.sums_position_name()}] = {self.head_name()};')
            source.append(f'{self.head_name()} = {self.sums_position_name()};')
            source.append(f'{self.nnz_name()}++;')
        source.append('}')
        return source

    def ravel_indexes(self, indexes: List[str], index: str):
        source = SourceBuilder()
        dimensions_so_far = []
        cumulative_sum = []
        for layer, index_i in zip(self.layers, indexes):
            cumulative_sum.append(' * '.join([index_i] + dimensions_so_far))
            dimensions_so_far.append(layer.dimension_name())

        ravel_index = ' + '.join(cumulative_sum)
        source.append(f'int32_t {index} = {ravel_index};')

        return source

    def unravel_indexes(self, index: str, indexes: List[str]):
        source = SourceBuilder()
        source.append(f'int32_t remaining_{index} = {index};')
        for layer, index_i in zip(self.layers, indexes):
            source.append(f'int32_t {index_i} = remaining_{index} % {layer.dimension_name()};')
            source.append(f'remaining_{index} = remaining_{index} / {layer.dimension_name()};')

        return source

    def next_name(self):
        return 'next_' + '_'.join(layer.tensor.variable.to_string() for layer in self.layers)

    def sums_name(self):
        return 'sums_' + '_'.join(layer.tensor.variable.to_string() for layer in self.layers)

    def head_name(self):
        return 'head_' + '_'.join(layer.tensor.variable.to_string() for layer in self.layers)

    def old_head_name(self):
        return 'old_head_' + '_'.join(layer.tensor.variable.to_string() for layer in self.layers)

    def nnz_name(self):
        return 'nnz_' + '_'.join(layer.tensor.variable.to_string() for layer in self.layers)

    def sums_position_name(self):
        return f'p_{self.sums_name()}'

    def old_sums_position_name(self):
        return f'old_p_{self.sums_name()}'


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


def write_crd_assembly(output: LatticeLeaf, index_variable: str):
    source = SourceBuilder()

    pointer = output.layer_pointer()
    capacity = output.crd_capacity_name()
    crd = output.crd_name()

    source.append(f'if ({pointer} >= {capacity}) {{')
    with source.indented():
        source.append(f'{capacity} *= 2;')
        source.append(f'{crd} = (int32_t*)realloc({crd}, sizeof(int32_t) * {capacity});')
    source.append('}')

    source.append(f'{crd}[{pointer}] = {index_variable};')
    source.append(f'{pointer}++;')

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
            source.append(f'{capacity} *= {capacity};')
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
        source.append(f'while({" && ".join(while_criteria)}) {{')

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

            ###############
            # Subsubnodes #
            ###############
            subsubnodes = generate_subgraphs(subnode)
            for i, subsubnode in enumerate(subsubnodes):
                sparse_subsubnode_leaves = subsubnode.lattice.sparse_leaves()

                ############################
                # Branch on sparse matches #
                ############################
                if len(sparse_subsubnode_leaves) == 0:
                    # Elide branch if there are no sparse layers
                    source.include(iteration_graph_to_c_code(subsubnode.next, output, kernel_type))
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

                        ###################################
                        # Allocate space for pos and vals #
                        ###################################
                        if kernel_type.is_assembly() and graph.is_sparse_output():
                            source.include(write_pos_allocation(graph.output))

                        #####################
                        # Invoke next layer #
                        #####################
                        source.include(iteration_graph_to_c_code(subsubnode.next, output, kernel_type))

                        ##########################
                        # Advance output pointer #
                        ##########################
                        if kernel_type.is_assembly() and graph.is_sparse_output():
                            if graph.next.is_sparse_output():
                                next_pointer = graph.next.output.layer_pointer()
                                source.append(f'if ({next_pointer} > {next_pointer}_begin) {{')
                                with source.indented():
                                    source.include(write_crd_assembly(graph.output, graph.index_variable))
                                source.append('}')
                            else:
                                source.include(write_crd_assembly(graph.output, graph.index_variable))
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
        pointer = graph.output.layer_pointer()
        pos = graph.output.pos_name()
        previous_pointer = graph.output.previous_layer_pointer()

        if previous_pointer == '0':
            end_index = '1'
        else:
            end_index = f'{previous_pointer}+1'

        source.append(f'{pos}[{end_index}] = {pointer};')

    return source


@iteration_graph_to_c_code.register(Scratch)
def scratch_node_to_code(graph: Scratch, output: Output, kernel_type: KernelType):
    source = SourceBuilder()

    new_output = ScratchOutput(graph.layers)
    source.include(new_output.write_declarations())

    if kernel_type.is_assembly():
        source.append(f'int32_t {new_output.head_name()} = -2;')
        source.append(f'int32_t {new_output.nnz_name()} = 0;')
        source.append('')

    source.include(iteration_graph_to_c_code(graph.next, new_output, kernel_type))

    source.append('')

    if kernel_type.is_assembly():
        source.append(f'while (1) {{')
        with source.indented():
            source.append(f'if ({new_output.head_name()} < 0) {{')
            with source.indented():
                source.append('break;')
            source.append('}')
            source.append('else {')
            with source.indented():
                indexes = [layer.layer_pointer() for layer in new_output.layers]
                source.include(new_output.unravel_indexes(new_output.sums_position_name(), indexes))
                for layer, index in zip(graph.layers, indexes):
                    source.append(f'{layer.crd_name()}[{layer.layer_pointer()}] = {index};')

                # Reset scratch space
                source.append(f'int32_t {new_output.old_head_name()} = {new_output.head_name()};')
                source.append(f'{new_output.head_name()} = {new_output.next_name()}[{new_output.old_head_name()}];')
                source.append(f'{new_output.next_name()}[{new_output.old_head_name()}] = -1;')

            source.append('}')
        source.append('}')

    # Sort indexes

    if len(graph.layers) > 1:
        raise NotImplementedError('Scratch space larger than 1 dimension is not fully supported')

    # Compress indexes
    # TODO: All but the top level need to be compressed

    # Write vals
    # TODO: ravel indexes to properly clear higher dimensional scratch space
    only_layer = graph.layers[0]
    source.include(write_sparse_initialization(only_layer))
    source.append(f'while ({only_layer.layer_pointer()} < {only_layer.layer_pointer()}_end) {{')
    with source.indented():
        source.include(output.write_assignment(f'{new_output.sums_name()}[{new_output.head_name()}]'))

        # Reset scratch space
        source.append(f'{new_output.sums_name()}[{only_layer.layer_pointer()}] = 0.0;')
        source.append(f'{only_layer.layer_pointer()}++;')
    source.append('}')

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


@iteration_graph_to_c_code.register(Contract)
def iteration_variable_to_c_code(graph: Contract, output: Output, kernel_type: KernelType):
    new_output = AccumulatorOutput('output')
    source = SourceBuilder()
    source.include(new_output.write_declarations())
    source.include(iteration_graph_to_c_code(graph.next, new_output, kernel_type))
    source.include(output.write_assignment('output'))
    return source


@iteration_graph_to_c_code.register(TerminalExpression)
def to_c_code_terminal_expression(graph: TerminalExpression, output: Output, kernel_type: KernelType):
    source = SourceBuilder()

    source.include(output.write_assignment(to_c_code(graph.expression)))
    return source


def dimension_name(index_variable: str):
    return f'{index_variable}_dim'


def pos_name(tensor: str, layer: int):
    return f'{tensor}_{layer}_pos'


def crd_name(tensor: str, layer: int):
    return f'{tensor}_{layer}_crd'


def vals_name(tensor: str):
    return f'{tensor}_vals'


def crd_capacity_name(tensor: str, layer: int):
    return f'{tensor}_{layer}_crd_capacity'


def pos_capacity_name(tensor: str, layer: int):
    return f'{tensor}_{layer}_pos_capacity'


def vals_capacity_name(tensor: str):
    return f'{tensor}_vals_capacity'


def generate_c_code(problem: Problem, graph: IterationGraph, kernel_type: KernelType):
    target_name = problem.assignment.target.name

    tensor_names = [target_name] + list(problem.input_formats.keys())

    # Emit headers
    source = SourceBuilder()
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

        # Allocate memory for target
        if kernel_type.is_assembly():
            all_dense = True
            for i, mode in enumerate(problem.output_format.modes):
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
                        source.append(f'int32_t {capacity} = 1048576;')
                    source.append(f'{pos_name(target_name, i)} = (int32_t*)malloc(sizeof(int32_t) * {capacity});')
                    source.append(f'{target_name}->indices[{i}][0] = (unit8_t*){pos_name(target_name, i)};')
                    source.append(f'{pos_name(target_name, i)}[0] = 0;')

                    capacity = crd_capacity_name(target_name, i)
                    source.append(f'int32_t {capacity} = 1048576;')
                    source.append(f'{crd_name(target_name, i)} = (int32_t*)malloc(sizeof(int32_t) * {capacity}')
                    source.append(f'{target_name}->indices[{i}][1] = (unit8_t*){crd_name(target_name, i)};')
                    source.append(f'int32_t {layer_pointer(problem.assignment.target.variable, i)} = 0;')

                    all_dense = False
                else:
                    raise NotImplementedError

            if all_dense:
                vals_size = ' * '.join(f'{target_name}->dimensions[{i}]' for i in range(problem.output_format.order))
            else:
                vals_size = vals_capacity_name(target_name)
                source.append(f'int32_t {vals_size} = 1048576;')
            source.append(f'{target_name}_vals = (double*)malloc(sizeof(double) * {vals_size});')
            source.append(f'{target_name}->vals = (uint8_t*){target_name}_vals;')

            source.append('')

        output = TensorOutput(LatticeLeaf(problem.assignment.target, problem.output_format.order - 1))
        source.include(output.write_declarations())
        source.include(iteration_graph_to_c_code(graph, output, kernel_type))

        source.append('')

        source.append('return 0;')

    source.append('}')

    return source
