__all__ = ['tensor_method', 'evaluate', 'PureTensorMethod']

from functools import lru_cache
from inspect import Signature, Parameter
from typing import Tuple, Dict

from .expression import Assignment
from .format import Format, parse_format
from .compile import taco_kernel, allocate_taco_structure, take_ownership_of_arrays
from .tensor import Tensor


class PureTensorMethod:
    """A function taking specific tensor arguments.
    """

    def __init__(self, assignment: Assignment, input_formats: Dict[str, Format], output_format: Format):
        if assignment.is_mutating():
            raise ValueError(f'{assignment} mutates its target and is so is not a pure function')

        variable_orders = assignment.expression.variable_orders()

        # Ensure that all parameters are defined
        for variable_name in variable_orders.keys():
            if variable_name not in input_formats:
                raise ValueError(f'Variable {variable_name} in {assignment} not listed in parameters')

        # Ensure that no extraneous parameters are defined
        for parameter_name in input_formats.keys():
            if parameter_name not in variable_orders:
                raise ValueError(f'Parameter {parameter_name} not in {assignment} variables')

        # Verify that parameters have the correct order
        for parameter_name, format in input_formats.items():
            if format.order != variable_orders[parameter_name]:
                raise ValueError(f'Parameter {parameter_name} has order {format.order}, but this variable in the '
                                 f'assignment has order {variable_orders[parameter_name]}')

        if output_format.order != assignment.target.order:
            raise ValueError(f'Output parameter has order {output_format.order}, but the output variable in the '
                             f'assignment has order {assignment.target.order}')

        # Store validated attributes
        self.assignment = assignment
        self.input_formats = input_formats
        self.output_format = output_format

        # Create Python signature of the function
        self.signature = Signature([Parameter(parameter_name, Parameter.POSITIONAL_OR_KEYWORD)
                                    for parameter_name in input_formats.keys()])

        # Compile taco function
        all_formats = {self.assignment.target.name: output_format, **input_formats}
        format_strings = frozenset((parameter_name, format_to_taco_format(format))
                                   for parameter_name, format in all_formats.items()
                                   if format.order != 0)  # Taco does not like formats for scalars
        self.parameter_order, self.cffi_lib = taco_kernel(assignment.deparse(), format_strings)

    def __call__(self, *args, **kwargs):
        # Handle arguments like normal Python function
        bound_arguments = self.signature.bind(*args, **kwargs).arguments

        # Validate tensor arguments
        for name, argument, format in zip(bound_arguments.keys(), bound_arguments.values(),
                                          self.input_formats.values()):
            if argument.order != format.order:
                raise ValueError(f'Argument {name} must have order {format.order} not {argument.order}')
            if tuple(argument.modes) != tuple(format.modes):
                raise ValueError(f'Argument {name} must have modes '
                                 f'{tuple(mode.name for mode in format.modes)} not '
                                 f'{tuple(mode.name for mode in argument.modes)}')
            if tuple(argument.mode_ordering) != tuple(format.ordering):
                raise ValueError(f'Argument {name} must have mode ordering '
                                 f'{format.ordering} not {argument.mode_ordering}')

        # Validate dimensions
        index_participants = self.assignment.expression.index_participants()
        index_sizes = {}
        for index, participants in index_participants.items():
            # Extract the size of dimension referenced by this index on each tensor that uses it; record the variable
            # name and dimension for a better error
            actual_sizes = [(variable, dimension, bound_arguments[variable].dimensions[dimension])
                            for variable, dimension in participants]

            reference_size = actual_sizes[0][2]
            index_sizes[index] = reference_size

            for variable, dimension, size in actual_sizes[1:]:
                if size != reference_size:
                    expected = ', '.join(f'{variable}.dimensions[{dimension}] == {size}'
                                         for variable, dimension, size in actual_sizes)
                    raise ValueError(f'{self.assignment} expected all these dimensions of these tensors to be the same '
                                     f'because they share the index {index}: {expected}')

        # Determine output dimensions
        output_dimensions = tuple(index_sizes[index] for index in self.assignment.target.indexes)

        cffi_output = allocate_taco_structure(tuple(mode.c_int for mode in self.output_format.modes),
                                              output_dimensions, self.output_format.ordering)

        output = Tensor(cffi_output)

        all_arguments = {self.assignment.target.name: output, **bound_arguments}

        cffi_args = [all_arguments[name].cffi_tensor for name in self.parameter_order]

        return_value = self.cffi_lib.evaluate(*cffi_args)

        take_ownership_of_arrays(cffi_output)

        if return_value != 0:
            raise RuntimeError(f'Taco function failed with error code {return_value}')

        return output


def tensor_method(assignment: str, input_formats: Dict[str, str], output_format: str) -> PureTensorMethod:
    return cachable_tensor_method(assignment, tuple(input_formats.items()), output_format)


@lru_cache()
def cachable_tensor_method(assignment: str, input_formats: Tuple[Tuple[str, str], ...], output_format: str
                           ) -> PureTensorMethod:
    from .expression.parser import parse_assignment
    parsed_assignment = parse_assignment(assignment).or_die()

    parsed_input_formats = {name: parse_format(format).or_die() for name, format in input_formats}

    parsed_output = parse_format(output_format).or_die()

    if parsed_assignment.is_mutating():
        raise NotImplementedError(f'Mutating tensor assignments like {assignment} not implemented yet.')
    else:
        return PureTensorMethod(parsed_assignment, parsed_input_formats, parsed_output)


def evaluate(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    input_formats = {name: tensor.format.deparse() for name, tensor in inputs.items()}

    function = tensor_method(assignment, input_formats, output_format)

    return function(**inputs)


def format_to_taco_format(format: Format):
    return ''.join(mode.character for mode in format.modes) + ':' + ','.join(map(str, format.ordering))
