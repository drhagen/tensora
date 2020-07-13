__all__ = ['taco_kernel', 'allocate_taco_structure', 'taco_structure_to_cffi', 'take_ownership_of_arrays',
           'tensor_cdefs']

import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, FrozenSet, Any
from weakref import WeakKeyDictionary

from cffi import FFI

taco_binary = Path(__file__).parent.joinpath('taco/bin/taco')

global_weakkeydict = WeakKeyDictionary()

# order: The number of dimensions of the tensor
# dimensions: The size of each dimension of the tensor; has length `order`
# csize: No idea what this is
# mode_ordering: The dimension that each level refers to; has length `order` and
#   has exactly the numbers 0 to `order` - 1; e.g. if `mode_ordering` is (2, 0, 1), then
#   the first level describes the last dimension, the second level describes the first
#   dimension, etc.
# mode_types: The type (dense, compressed sparse) of each level; importantly, the ith
#   element in `mode_types` describes the ith element in `indices`, not `dimensions`
# indices: A complex data structure storing all the index information of the structure;
#   it has length `order`; each element is one level of the data structure; each level
#   conceptually stores one dimension worth of indexes;
#   *   if a level is dense, then the element in indices is a null pointer or a pointer to
#       a length 0 array or a pointer to a length 1 array, which contains a pointer to a
#       length 1 array, which contains the size of this level's dimension. It does not
#       really matter what goes here because it is never used.
#   *   if a level is compressed (sparse), then the element in indices is a pointer to a
#       length 2 array
#       *   the first element is the `pos` arrays of compressed sparse matrix
#           representation; it has a number of elements equal to the number of indexes
#           in the previous level plus 1; each element `i` in `pos` is the starting
#           position in `idx` for indexes in this dimension associated with the `i`th
#           index of the previous dimension; the last element is the total length of
#           `idx` (that is, the first index not in `idx`)
#       *   the second element is the `idx` array of compressed sparse matrix
#           representation; each element is an index that has a value in this dimension
# vals: The actual values of the sparse matrix; has a length equal to the number of indexes
#   in the last dimension; one has to traverse the `indices` structure to determine the
#   coordinate of each value
# vals_size: Deprecated https://github.com/tensor-compiler/taco/issues/208#issuecomment-476314322

taco_define_header = """
    #ifndef TACO_C_HEADERS
    #define TACO_C_HEADERS
    #define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
    #define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
    #endif
"""

taco_type_header = """
    typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;

    typedef struct {
      int32_t      order;         // tensor order (number of modes)
      int32_t*     dimensions;    // tensor dimensions
      int32_t      csize;         // component size
      int32_t*     mode_ordering; // mode storage ordering
      taco_mode_t* mode_types;    // mode storage types
      uint8_t***   indices;       // tensor index data (per mode)
      uint8_t*     vals;          // tensor values
      int32_t      vals_size;     // values array size
    } taco_tensor_t;

    void free(void *ptr);
"""

# Define ffi for building the tensors.
tensor_cdefs = FFI()

# This `FFI` is `include`d in each kernel so objects created with its `new` can be passed to the kernels.
tensor_cdefs.cdef(taco_type_header)

# This library only has definitions, in order to call `dlopen`, `set_source` must be called with empty `source` first
tensor_cdefs.set_source('_main', '')

tensor_lib = tensor_cdefs.dlopen(None)


def taco_kernel(expression: str, formats: FrozenSet[Tuple[str, str]]) -> Tuple[List[str], Any]:
    """Call taco with expression and compile resulting function.

    Given an expression and a set of formats:
    (1) call out to taco to get the source code for the evaluate function that runs that expression for those formats
    (2) parse the signature in the source to determine the order of arguments
    (3) compile the source with cffi
    (4) return the list of parameter names and the compiled library

    Because compilation can take a non-trivial amount of time, the results of this function is cached by a
    `functools.lru_cache`, which is configured to store the results of the 256 most recent calls to this function.

    Args:
        expression: An expression that can parsed by taco.
        formats: A frozen set of pairs of strings. It must be a frozen set because `lru_cache` requires that the
        arguments be hashable and therefore immutable. The first element of each pair is a variable name; the second
        element is the format in taco format (e.g. 'dd:1,0', 'dss:0,1,2'). Scalar variables must not be listed because
        taco does not understand them having a format.

    Returns:
        A tuple where the first element is the list of variable names in the order they appear in the function
        signature, and the second element is the compiled FFILibrary which has a single method `evaluate` which expects
        cffi pointers to taco_tensor_t instances in order specified by the list of variable names.
    """
    # Call taco to write the kernels to standard out
    result = subprocess.run([taco_binary, expression, '-print-evaluate', '-print-nocolor']
                            + [f'-f={name}:{format}' for name, format in formats],
                            capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    source = result.stdout

    # Determine signature
    # 1) Find function by name and capture its parameter list
    # 2) Find each parameter by `*` and capture its name
    signature_match = re.search(r'int evaluate\(([^)]*)\)', source)
    signature = signature_match.group(0)
    parameter_list_matches = re.finditer(r'\*([^,]*)', signature_match.group(1))
    parameter_names = [match.group(1) for match in parameter_list_matches]

    # Use cffi to compile the kernels
    ffibuilder = FFI()
    ffibuilder.include(tensor_cdefs)
    ffibuilder.cdef(signature + ';')
    ffibuilder.set_source('taco_kernel', taco_define_header + taco_type_header + source)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create shared object in temporary directory
        lib_path = ffibuilder.compile(tmpdir=temp_dir)

        # Load the shared object
        lib = ffibuilder.dlopen(lib_path)

    # Return the parameter names because we need to know the order in which to send the arguments. It appears that this
    # order is always the order in which the name first appears in the expression left-to-right, but it is not clear
    # that this is guaranteed.
    # Return the entire library rather than just the function because it appears that the memory containing the compiled
    # code is freed as soon as the library goes out of scope: https://stackoverflow.com/q/55323592/1485877
    return parameter_names, lib


def allocate_taco_structure(mode_types: Tuple[int, ...], dimensions: Tuple[int, ...], mode_ordering: Tuple[int, ...]):
    """Allocate all parts of a taco tensor except growable arrays.

    All int32_t[] tensor.indices[*][*] and double[] tensor.vals are NULL pointers. All other properties are immutable.

    For level i, tensor.indices[i] is
    * if tensor.mode_types[i] is dense: NULL or int32_t*[0]
    * if tensor.mode_types[i] is compressed: int32_t*[2]

    Args:
        mode_types: A list of length order of integer representations of taco_mode_t for each level.
        dimensions: A list of length order of the size of each dimension.
        mode_ordering: A list of length order indicating that the ith mode corresponds to the mode_ordering[i] dimension

    Returns:
        A cffi taco_tensor_t*
    """
    # Validate inputs
    if not (len(mode_types) == len(dimensions) == len(mode_ordering)):
        raise ValueError(f'Must all be the same length: mode_types = {mode_types}, dimensions = {dimensions}, '
                         f'mode_ordering = {mode_ordering}')

    for mode_type in mode_types:
        if mode_type not in (0, 1):
            raise ValueError(f'mode_types must only contain elements 0 or 1: {mode_types}')

    for dimension in dimensions:
        if dimension < 0:
            # cffi will reject integers too big to fit into an int32_t
            raise ValueError(f'All values in dimensions must be positive: {dimensions}')

    if set(mode_ordering) != set(range(len(mode_types))):
        raise ValueError(f'mode_ordering must contain each number in the set {{0, 1, ..., order - 1}} exactly once: '
                         f'{mode_ordering}')

    # This structure mimics the taco structure and holds the objects owning the memory of the arrays, ensuring that the
    # pointers stay valid as long as the cffi taco structure has not been garbage collected.
    memory_holder = {}

    cffi_tensor = tensor_cdefs.new('taco_tensor_t*')

    cffi_tensor.order = len(mode_types)

    cffi_dimensions = tensor_cdefs.new('int32_t[]', dimensions)
    memory_holder['dimensions'] = cffi_dimensions
    cffi_tensor.dimensions = cffi_dimensions

    cffi_mode_ordering = tensor_cdefs.new('int32_t[]', mode_ordering)
    memory_holder['mode_ordering'] = cffi_mode_ordering
    cffi_tensor.mode_ordering = cffi_mode_ordering

    cffi_mode_types = tensor_cdefs.new('taco_mode_t[]', mode_types)
    memory_holder['mode_types'] = cffi_mode_types
    cffi_tensor.mode_types = cffi_mode_types

    converted_levels = []
    memory_holder_levels = []
    memory_holder_levels_arrays = []
    for mode in mode_types:
        converted_arrays = []
        if mode == tensor_lib.taco_mode_dense:
            converted_arrays = []
        elif mode == tensor_lib.taco_mode_sparse:
            converted_arrays = [tensor_cdefs.NULL, tensor_cdefs.NULL]
        cffi_level = tensor_cdefs.new('int32_t*[]', converted_arrays)
        memory_holder_levels.append(cffi_level)
        memory_holder_levels_arrays.append(converted_arrays)
        converted_levels.append(cffi_level)
    cffi_levels = tensor_cdefs.new('int32_t**[]', converted_levels)
    memory_holder['indices'] = cffi_levels
    memory_holder['*indices'] = memory_holder_levels
    memory_holder['**indices'] = memory_holder_levels_arrays
    cffi_tensor.indices = tensor_cdefs.cast('uint8_t***', cffi_levels)

    cffi_tensor.vals = tensor_cdefs.NULL

    cffi_tensor.vals_size = 0

    global_weakkeydict[cffi_tensor] = memory_holder

    return cffi_tensor


def taco_structure_to_cffi(indices: List[List[List[int]]], vals: List[float], *,
                           mode_types: Tuple[int, ...], dimensions: Tuple[int, ...], mode_ordering: Tuple[int, ...]):
    """Build a cffi taco tensor from Python data.

    This takes Python data with a one-to-one mapping to taco tensor attributes and builds a cffi taco tensor from it.

    Args:
        indices: The list of length order containing the index data required by the taco structure.
        vals: The explicit values required by the taco structure.
        mode_types: A list of length order of integer representations of taco_mode_t for each level.
        dimensions: A list of length order of the size of each dimension.
        mode_ordering: A list of length order indicating that the ith mode corresponds to the mode_ordering[i] dimension

    Returns:
        A cffi taco_tensor_t*. This object owns all the data under it. The associated arrays will be freed when this
        object is garbage collected.
    """
    cffi_tensor = allocate_taco_structure(mode_types, dimensions, mode_ordering)

    # Validate inputs
    if len(indices) != len(mode_types):
        raise ValueError(f'Length of indices ({len(indices)}) must be equal to the length of mode_types, dimensions, '
                         f'and mode_ordering ({len(mode_types)})')

    nnz = 1
    for i_level in range(cffi_tensor.order):
        if mode_types[i_level] == 0:
            if len(indices[i_level]) != 0:
                raise ValueError(f'Level {i_level} is a dense mode and therefore expects indices[{i_level}] to be '
                                 f'empty: {indices[i_level]}')
            nnz *= dimensions[mode_ordering[i_level]]
        elif mode_types[i_level] == 1:
            if len(indices[i_level]) != 2:
                raise ValueError(f'Level {i_level} is a compressed mode and therefore expects indices[{i_level}] to be '
                                 f'length 2 not length {len(indices[i_level])}: {indices[i_level]}')
            pos = indices[i_level][0]
            crd = indices[i_level][1]
            if len(pos) != nnz + 1:
                raise ValueError(f'The pos array of level {i_level} (indices[{i_level}][0]) must have length {nnz}, '
                                 f'the number of explicit indexes so far, not length {len(pos)}: {pos}')
            if pos[0] != 0:
                raise ValueError(f'The first element of the pos array of level {i_level} must be 0: {pos}')
            if not weakly_increasing(pos):
                raise ValueError(f'The pos array of level {i_level} (indices[{i_level}][0]) must be weakly '
                                 f'monotonically increasing: {pos}')
            if len(crd) != pos[-1]:
                raise ValueError(f'The crd array of level {i_level} (indices[{i_level}][1]) must have length '
                                 f"{pos[-1]}, the last element of this level's pos array, not length {len(crd)}: {crd}")
            if not all(0 <= x < dimensions[mode_ordering[i_level]] for x in crd):
                raise ValueError(f'All values in the crd array of level {i_level } (indices[{i_level}][1]) must be '
                                 f'nonnegative and less than the size of this dimension: {crd}')
            nnz = len(crd)

    if len(vals) != nnz:
        raise ValueError(f'Length of vals must be equal to the number of indexes implicitly defined by indices {nnz} '
                         f'not {len(vals)}: {vals}')

    # Get the partial constructed memory holder stored by allocate_taco_structure
    memory_holder = global_weakkeydict[cffi_tensor]

    cffi_indices = tensor_cdefs.cast('int32_t***', cffi_tensor.indices)
    for i_level, (mode, level) in enumerate(zip(mode_types, indices)):
        if mode == tensor_lib.taco_mode_dense:
            pass
        elif mode == tensor_lib.taco_mode_sparse:
            for i_array, array in enumerate(level):
                cffi_array = tensor_cdefs.new('int32_t[]', array)
                memory_holder['**indices'][i_level][i_array] = cffi_array
                cffi_indices[i_level][i_array] = cffi_array

    cffi_vals = tensor_cdefs.new('double[]', vals)
    memory_holder['vals'] = cffi_vals
    cffi_tensor.vals = tensor_cdefs.cast('uint8_t*', cffi_vals)

    cffi_tensor.vals_size = len(vals)

    global_weakkeydict[cffi_tensor] = memory_holder

    return cffi_tensor


def take_ownership_of_arrays(cffi_tensor) -> None:
    """Take ownership of arrays allocated in taco kernel.

    The `evaluate` function created by taco `malloc`s several arrays on the output tensor, namely the `pos` and `crd`
    arrays of compressed levels and the `vals` array. This function looks up the tensor in `global_weakkeydict` and
    attaches the allocated arrays so that they get deallocated when the tensor is garbage collected. This function must
    be called on the output tensor of any evaluated taco kernel.

    Args:
        cffi_tensor: A cffi taco_tensor_t*.
    """

    memory_holder = global_weakkeydict[cffi_tensor]

    order = cffi_tensor.order

    modes = cffi_tensor.mode_types[0:order]

    cffi_levels = tensor_cdefs.cast('int32_t***', cffi_tensor.indices)
    for i_dimension, mode in enumerate(modes):
        if mode == tensor_lib.taco_mode_dense:
            pass
        if mode == tensor_lib.taco_mode_sparse:
            memory_holder['**indices'][i_dimension][0] = tensor_cdefs.gc(cffi_levels[i_dimension][0], tensor_lib.free)
            memory_holder['**indices'][i_dimension][1] = tensor_cdefs.gc(cffi_levels[i_dimension][1], tensor_lib.free)

    memory_holder['vals'] = tensor_cdefs.gc(cffi_tensor.vals, tensor_lib.free)


def weakly_increasing(list: List[int]):
    return all(x <= y for x, y in zip(list, list[1:]))
