__all__ = ["compile_evaluate"]

import re
import tempfile
import threading
from typing import Any

from cffi import FFI

from ._cffi_ownership import taco_type_header, tensor_cdefs

lock = threading.Lock()

taco_define_header = """
    #ifndef TACO_C_HEADERS
    #define TACO_C_HEADERS
    #define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
    #define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
    #endif
"""


def compile_evaluate(source: str) -> Any:
    """Compile evaluate kernel in C code using CFFI.

    Args:
        source: C code containing one evaluate function

    Returns:
        The compiled FFILibrary which has a single method `evaluate` which
        expects cffi pointers to taco_tensor_t instances.
    """
    # Extract signature
    # This needs to be provided alone to cdef
    signature_match = re.search(r"int(32_t)? evaluate\(([^)]*)\)", source)
    signature = signature_match.group(0)

    # Use cffi to compile the kernels
    ffibuilder = FFI()
    ffibuilder.include(tensor_cdefs)
    ffibuilder.cdef(signature + ";")
    ffibuilder.set_source(
        "taco_kernel",
        taco_define_header + taco_type_header + source,
        extra_compile_args=["-Wno-unused-variable", "-Wno-unknown-pragmas"],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Lock because FFI.compile is not thread safe: https://foss.heptapod.net/pypy/cffi/-/issues/490
        with lock:
            # Create shared object in temporary directory
            lib_path = ffibuilder.compile(tmpdir=temp_dir)

        # Load the shared object
        lib = ffibuilder.dlopen(lib_path)

    # Return the entire library rather than just the function because it appears that the memory containing the compiled
    # code is freed as soon as the library goes out of scope: https://stackoverflow.com/q/55323592/1485877
    return lib
