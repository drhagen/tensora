__all__ = [
    "TensorMethod",
    "BackendCompiler",
    "BroadcastTargetIndexError",
    "UnsupportedBackendCompilerError",
]

from dataclasses import dataclass
from enum import Enum
from inspect import Parameter, Signature

from returns.result import Failure, Success

from ..expression.ast import Assignment
from ..generate import Language, TensorCompiler, generate_code, generate_module_tensora
from ..kernel_type import KernelType
from ..problem import Problem
from ..tensor import Tensor
from ._cffi_ownership import allocate_taco_structure, take_ownership_of_arrays, tensor_cdefs


@dataclass(frozen=True, slots=True)
class BroadcastTargetIndexError(Exception):
    index: str
    assignment: Assignment

    def __str__(self):
        return (
            f"Expected index variable {self.index} on the target variable to be mentioned on the "
            f"right-hand side, but it was not: {self.assignment}. Such broadcasting makes sense "
            f"in a kernel and those kernels can be generated, but they cannot be used in "
            f"`evaluate` or `tensor_method` because those functions get the output dimensions "
            f"from the the dimensions of the input tensors."
        )


class BackendCompiler(Enum):
    """The tool to generate the machine code.

    Attributes
    ----------
    llvm
        Generate LLVM IR and compile with the llvmlite package.
        Not available with TensorCompiler.taco.
    cffi
        Generate C code and compile with the cffi package.
        Not available on Windows.
    """

    llvm = "llvm"
    cffi = "cffi"


@dataclass(frozen=True, slots=True)
class UnsupportedBackendCompilerError(Exception):
    backend_compiler: BackendCompiler
    tensor_compiler: TensorCompiler

    def __str__(self):
        return f"Backend compiler {self.backend_compiler} is not supported for tensor compiler {self.tensor_compiler}"


class TensorMethod:
    """A function taking specific tensor arguments."""

    def __init__(
        self,
        problem: Problem,
        compiler: TensorCompiler = TensorCompiler.tensora,
        backend: BackendCompiler = BackendCompiler.llvm,
    ):
        # Reject broadcasting to outputs because there is no way to specify output dimensions that
        # do not have a corresponding input dimension
        input_indexes = set(problem.assignment.expression.index_participants().keys())
        for output_index in problem.assignment.target.indexes:
            if output_index not in input_indexes:
                raise BroadcastTargetIndexError(output_index, problem.assignment)

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

        match backend:
            case BackendCompiler.llvm:
                match compiler:
                    case TensorCompiler.tensora:
                        match generate_module_tensora(problem, [KernelType.evaluate]):
                            case Failure(error):
                                raise error
                            case Success(tensora_module):
                                from ._compile_llvm import compile_module

                                self._lib = compile_module(tensora_module)

                                # Convert ctypes function to cffi function
                                function_type = (
                                    f"int32_t (*)({', '.join(['void *'] * len(problem.formats))})"
                                )
                                function_pointer = self._lib.get_function_address("evaluate")
                                self._evaluate = tensor_cdefs.cast(function_type, function_pointer)
                    case TensorCompiler.taco:
                        raise UnsupportedBackendCompilerError(backend, compiler)
            case BackendCompiler.cffi:
                match generate_code(problem, [KernelType.evaluate], compiler, Language.c):
                    case Failure(error):
                        raise error
                    case Success(c_code):
                        from ._compile_cffi import compile_evaluate

                        self._lib = compile_evaluate(c_code)
                        self._evaluate = self._lib.evaluate

    def __call__(self, *args, **kwargs):
        # Handle arguments like normal Python function
        bound_arguments = self.signature.bind(*args, **kwargs).arguments

        # Validate tensor arguments
        for name, argument, format in zip(
            bound_arguments.keys(),
            bound_arguments.values(),
            self._input_formats.values(),
            strict=True,
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

        cffi_args = [all_arguments[name].cffi_tensor for name in self._problem.formats.keys()]

        return_value = self._evaluate(*cffi_args)

        take_ownership_of_arrays(cffi_output)

        if return_value != 0:
            raise RuntimeError(f"Kernel evaluation failed with error code {return_value}")

        return output
