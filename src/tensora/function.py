__all__ = ["tensor_method", "evaluate", "evaluate_taco", "evaluate_tensora", "TensorMethod"]

from functools import lru_cache
from inspect import Parameter, Signature

from returns.functions import raise_exception

from .compile import allocate_taco_structure, generate_library, take_ownership_of_arrays
from .expression import parse_assignment
from .format import parse_format
from .generate import TensorCompiler
from .problem import Problem, make_problem
from .tensor import Tensor


class TensorMethod:
    """A function taking specific tensor arguments."""

    def __init__(self, problem: Problem, compiler: TensorCompiler = TensorCompiler.tensora):
        # Store validated attributes
        self._problem = problem
        self._output_name = problem.assignment.target.name
        self._input_formats = {
            name: format for name, format in problem.formats.items() if name != self._output_name
        }
        self._output_format = problem.formats[self._output_name]

        # Create Python signature of the function
        self.signature = Signature(
            [
                Parameter(parameter_name, Parameter.KEYWORD_ONLY, annotation=Tensor)
                for parameter_name in self._input_formats.keys()
            ]
        )

        # Compile taco function
        self._parameter_order, self._cffi_lib = generate_library(problem, compiler)

    def __call__(self, *args, **kwargs):
        # Handle arguments like normal Python function
        bound_arguments = self.signature.bind(*args, **kwargs).arguments

        # Validate tensor arguments
        for name, argument, format in zip(
            bound_arguments.keys(), bound_arguments.values(), self._input_formats.values()
        ):
            if not isinstance(argument, Tensor):
                raise TypeError(f"Argument {name} must be a Tensor not {type(argument)}")

            if argument.order != format.order:
                raise ValueError(
                    f"Argument {name} must have order {format.order} not {argument.order}"
                )
            if tuple(argument.modes) != tuple(format.modes):
                raise ValueError(
                    f"Argument {name} must have modes "
                    f"{tuple(mode.name for mode in format.modes)} not "
                    f"{tuple(mode.name for mode in argument.modes)}"
                )
            if tuple(argument.mode_ordering) != tuple(format.ordering):
                raise ValueError(
                    f"Argument {name} must have mode ordering "
                    f"{format.ordering} not {argument.mode_ordering}"
                )

        # Validate dimensions
        index_participants = self._problem.assignment.expression.index_participants()
        index_sizes = {}
        for index, participants in index_participants.items():
            # Extract the size of dimension referenced by this index on each tensor that uses it; record the variable
            # name and dimension for a better error
            actual_sizes = [
                (variable, dimension, bound_arguments[variable].dimensions[dimension])
                for variable, dimension in participants
            ]

            reference_size = actual_sizes[0][2]
            index_sizes[index] = reference_size

            for _, _, size in actual_sizes[1:]:
                if size != reference_size:
                    expected = ", ".join(
                        f"{variable}.dimensions[{dimension}] == {size}"
                        for variable, dimension, size in actual_sizes
                    )
                    raise ValueError(
                        f"{self._problem.assignment} expected all these dimensions of these tensors to be the same "
                        f"because they share the index {index}: {expected}"
                    )

        # Determine output dimensions
        output_dimensions = tuple(
            index_sizes[index] for index in self._problem.assignment.target.indexes
        )

        cffi_output = allocate_taco_structure(
            tuple(mode.c_int for mode in self._output_format.modes),
            output_dimensions,
            self._output_format.ordering,
        )

        output = Tensor(cffi_output)

        all_arguments = {self._output_name: output, **bound_arguments}

        cffi_args = [all_arguments[name].cffi_tensor for name in self._parameter_order]

        return_value = self._cffi_lib.evaluate(*cffi_args)

        take_ownership_of_arrays(cffi_output)

        if return_value != 0:
            raise RuntimeError(f"Kernel evaluation failed with error code {return_value}")

        return output


@lru_cache()
def cachable_tensor_method(problem: Problem, compiler: TensorCompiler) -> TensorMethod:
    return TensorMethod(problem, compiler)


def tensor_method(
    assignment: str,
    formats: dict[str, str],
    compiler: TensorCompiler = TensorCompiler.tensora,
) -> TensorMethod:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    parsed_formats = {
        name: parse_format(format).alt(raise_exception).unwrap()
        for name, format in formats.items()
    }

    problem = make_problem(parsed_assignment, parsed_formats).alt(raise_exception).unwrap()

    return cachable_tensor_method(problem, compiler)


def evaluate_taco(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    input_formats = {name: tensor.format for name, tensor in inputs.items()}
    parsed_output_format = parse_format(output_format).alt(raise_exception).unwrap()

    formats = {parsed_assignment.target.name: parsed_output_format} | input_formats

    problem = make_problem(parsed_assignment, formats).alt(raise_exception).unwrap()

    function = cachable_tensor_method(problem, TensorCompiler.taco)

    return function(**inputs)


def evaluate_tensora(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    parsed_assignment = parse_assignment(assignment).alt(raise_exception).unwrap()
    input_formats = {name: tensor.format for name, tensor in inputs.items()}
    parsed_output_format = parse_format(output_format).alt(raise_exception).unwrap()

    formats = {parsed_assignment.target.name: parsed_output_format} | input_formats

    problem = make_problem(parsed_assignment, formats).alt(raise_exception).unwrap()

    function = cachable_tensor_method(problem, TensorCompiler.tensora)

    return function(**inputs)


def evaluate(assignment: str, output_format: str, **inputs: Tensor) -> Tensor:
    return evaluate_tensora(assignment, output_format, **inputs)
