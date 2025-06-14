__all__ = [
    "allocate_taco_structure",
    "taco_structure_to_cffi",
    "take_ownership_of_arrays",
    "take_ownership_of_tensor_members",
    "take_ownership_of_tensor",
    "taco_type_header",
    "tensor_cdefs",
]

import platform
from itertools import pairwise
from weakref import WeakKeyDictionary

from cffi import FFI

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

taco_type_header = """
    typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;

    typedef struct {
      int32_t      order;         // tensor order (number of modes)
      int32_t*     dimensions;    // tensor dimensions
      int32_t      csize;         // component size
      int32_t*     mode_ordering; // mode storage ordering
      taco_mode_t* mode_types;    // mode storage types
      int32_t***   indices;       // tensor index data (per mode)
      double*      vals;          // tensor values
      int32_t      vals_size;     // values array size
    } taco_tensor_t;

    void free(void *ptr);
"""

# Define ffi for building the tensors.
tensor_cdefs = FFI()

# This `FFI` is `include`d in each kernel so objects created with its `new` can be passed to the kernels.
tensor_cdefs.cdef(taco_type_header)

# This library only has definitions, in order to `include` it elsewhere, `set_source` must be called with empty `source` first
tensor_cdefs.set_source("_main", "")

if platform.system() == "Windows":
    # On Windows, standard C functions like free are in msvcrt.dll
    tensor_lib = tensor_cdefs.dlopen("msvcrt.dll")
else:
    # On Linux and Mac, standard C functions like free are in the system C library
    tensor_lib = tensor_cdefs.dlopen(None)


def allocate_taco_structure(
    mode_types: tuple[int, ...],
    dimensions: tuple[int, ...],
    mode_ordering: tuple[int, ...],
):
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
        raise ValueError(
            f"Must all be the same length: mode_types = {mode_types}, dimensions = {dimensions}, "
            f"mode_ordering = {mode_ordering}"
        )

    for mode_type in mode_types:
        if mode_type not in (0, 1):
            raise ValueError(f"mode_types must only contain elements 0 or 1: {mode_types}")

    for dimension in dimensions:
        if dimension < 0:
            # cffi will reject integers too big to fit into an int32_t
            raise ValueError(f"All values in dimensions must be positive: {dimensions}")

    if set(mode_ordering) != set(range(len(mode_types))):
        raise ValueError(
            f"mode_ordering must contain each number in the set {{0, 1, ..., order - 1}} exactly once: "
            f"{mode_ordering}"
        )

    # This structure mimics the taco structure and holds the objects owning the memory of the arrays, ensuring that the
    # pointers stay valid as long as the cffi taco structure has not been garbage collected.
    memory_holder = {}

    cffi_tensor = tensor_cdefs.new("taco_tensor_t*")

    cffi_tensor.order = len(mode_types)

    cffi_dimensions = tensor_cdefs.new("int32_t[]", dimensions)
    memory_holder["dimensions"] = cffi_dimensions
    cffi_tensor.dimensions = cffi_dimensions

    cffi_mode_ordering = tensor_cdefs.new("int32_t[]", mode_ordering)
    memory_holder["mode_ordering"] = cffi_mode_ordering
    cffi_tensor.mode_ordering = cffi_mode_ordering

    cffi_mode_types = tensor_cdefs.new("taco_mode_t[]", mode_types)
    memory_holder["mode_types"] = cffi_mode_types
    cffi_tensor.mode_types = cffi_mode_types

    converted_levels = []
    memory_holder_levels = []
    memory_holder_levels_arrays = []
    for mode in mode_types:
        if mode == tensor_lib.taco_mode_dense:
            converted_arrays = []
        elif mode == tensor_lib.taco_mode_sparse:
            converted_arrays = [tensor_cdefs.NULL, tensor_cdefs.NULL]
        cffi_level = tensor_cdefs.new("int32_t*[]", converted_arrays)
        memory_holder_levels.append(cffi_level)
        memory_holder_levels_arrays.append(converted_arrays)
        converted_levels.append(cffi_level)
    cffi_levels = tensor_cdefs.new("int32_t**[]", converted_levels)
    memory_holder["indices"] = cffi_levels
    memory_holder["*indices"] = memory_holder_levels
    memory_holder["**indices"] = memory_holder_levels_arrays
    cffi_tensor.indices = tensor_cdefs.cast("int32_t***", cffi_levels)

    cffi_tensor.vals = tensor_cdefs.NULL

    cffi_tensor.vals_size = 0

    global_weakkeydict[cffi_tensor] = memory_holder

    return cffi_tensor


def taco_structure_to_cffi(
    indices: list[list[list[int]]],
    vals: list[float],
    *,
    mode_types: tuple[int, ...],
    dimensions: tuple[int, ...],
    mode_ordering: tuple[int, ...],
):
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
        raise ValueError(
            f"Length of indices ({len(indices)}) must be equal to the length of mode_types, dimensions, "
            f"and mode_ordering ({len(mode_types)})"
        )

    nnz = 1
    for i_level in range(cffi_tensor.order):
        if mode_types[i_level] == 0:
            if len(indices[i_level]) != 0:
                raise ValueError(
                    f"Level {i_level} is a dense mode and therefore expects indices[{i_level}] to be "
                    f"empty: {indices[i_level]}"
                )
            nnz *= dimensions[mode_ordering[i_level]]
        elif mode_types[i_level] == 1:
            if len(indices[i_level]) != 2:
                raise ValueError(
                    f"Level {i_level} is a compressed mode and therefore expects indices[{i_level}] to be "
                    f"length 2 not length {len(indices[i_level])}: {indices[i_level]}"
                )
            pos = indices[i_level][0]
            crd = indices[i_level][1]
            if len(pos) != nnz + 1:
                raise ValueError(
                    f"The pos array of level {i_level} (indices[{i_level}][0]) must have length {nnz}, "
                    f"the number of explicit indexes so far, not length {len(pos)}: {pos}"
                )
            if pos[0] != 0:
                raise ValueError(
                    f"The first element of the pos array of level {i_level} must be 0: {pos}"
                )
            if not weakly_increasing(pos):
                raise ValueError(
                    f"The pos array of level {i_level} (indices[{i_level}][0]) must be weakly "
                    f"monotonically increasing: {pos}"
                )
            if len(crd) != pos[-1]:
                raise ValueError(
                    f"The crd array of level {i_level} (indices[{i_level}][1]) must have length "
                    f"{pos[-1]}, the last element of this level's pos array, not length {len(crd)}: {crd}"
                )
            if not all(0 <= x < dimensions[mode_ordering[i_level]] for x in crd):
                raise ValueError(
                    f"All values in the crd array of level {i_level} (indices[{i_level}][1]) must be "
                    f"nonnegative and less than the size of this dimension: {crd}"
                )
            nnz = len(crd)

    if len(vals) != nnz:
        raise ValueError(
            f"Length of vals must be equal to the number of indexes implicitly defined by indices {nnz} "
            f"not {len(vals)}: {vals}"
        )

    # Get the partial constructed memory holder stored by allocate_taco_structure
    memory_holder = global_weakkeydict[cffi_tensor]

    cffi_indices = tensor_cdefs.cast("int32_t***", cffi_tensor.indices)
    for i_level, (mode, level) in enumerate(zip(mode_types, indices, strict=True)):
        if mode == tensor_lib.taco_mode_dense:
            pass
        elif mode == tensor_lib.taco_mode_sparse:
            for i_array, array in enumerate(level):
                cffi_array = tensor_cdefs.new("int32_t[]", array)
                memory_holder["**indices"][i_level][i_array] = cffi_array
                cffi_indices[i_level][i_array] = cffi_array

    cffi_vals = tensor_cdefs.new("double[]", vals)
    memory_holder["vals"] = cffi_vals
    cffi_tensor.vals = tensor_cdefs.cast("double*", cffi_vals)

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
        cffi_tensor: A cffi taco_tensor_t or taco_tensor_t*.
    """

    memory_holder = global_weakkeydict[cffi_tensor]

    order = cffi_tensor.order

    modes = cffi_tensor.mode_types[0:order]

    cffi_levels = tensor_cdefs.cast("int32_t***", cffi_tensor.indices)
    for i_dimension, mode in enumerate(modes):
        if mode == tensor_lib.taco_mode_dense:
            pass
        if mode == tensor_lib.taco_mode_sparse:
            memory_holder["**indices"][i_dimension][0] = tensor_cdefs.gc(
                cffi_levels[i_dimension][0], tensor_lib.free
            )
            memory_holder["**indices"][i_dimension][1] = tensor_cdefs.gc(
                cffi_levels[i_dimension][1], tensor_lib.free
            )

    memory_holder["vals"] = tensor_cdefs.gc(cffi_tensor.vals, tensor_lib.free)


def take_ownership_of_tensor_members(cffi_tensor) -> None:
    """Take ownership of taco tensor whose members were allocated elsewhere.

    This is not needed by pure Tensora. But some low-level programs might
    allocate members of a tensor and return it to Python. If the taco_tensor_t
    will be owned by Python, then this function can be used to take ownership of
    all the memory of all members of the taco_tensor_t so that it is freed when
    the cffi object is freed.

    Args:
        cffi_tensor: A cffi taco_tensor_t or taco_tensor_t*.
    """
    # This mayor may not have been called by take_ownership_of_tensor first, so
    # an entry in global_weakkeydict may or may not already exist.
    memory_holder = global_weakkeydict.get(cffi_tensor, {})

    # First, take ownership of everything that is owned after taco_structure_to_cffi
    order = cffi_tensor.order

    memory_holder["dimensions"] = tensor_cdefs.gc(cffi_tensor.dimensions, tensor_lib.free)

    memory_holder["mode_ordering"] = tensor_cdefs.gc(cffi_tensor.mode_ordering, tensor_lib.free)

    memory_holder["mode_types"] = tensor_cdefs.gc(cffi_tensor.mode_types, tensor_lib.free)

    memory_holder_levels = []
    memory_holder_levels_arrays = []
    for i_dimension, mode in enumerate(cffi_tensor.mode_types[0:order]):
        memory_holder_levels.append(
            tensor_cdefs.gc(cffi_tensor.indices[i_dimension], tensor_lib.free)
        )
        if mode == tensor_lib.taco_mode_dense:
            memory_holder_levels_arrays.append([])
        elif mode == tensor_lib.taco_mode_sparse:
            # It does not matter what the values are here. They will be overwritten in take_ownership_of_arrays.
            memory_holder_levels_arrays.append([tensor_cdefs.NULL, tensor_cdefs.NULL])
    memory_holder["indices"] = tensor_cdefs.gc(cffi_tensor.indices, tensor_lib.free)
    memory_holder["*indices"] = memory_holder_levels
    memory_holder["**indices"] = memory_holder_levels_arrays

    global_weakkeydict[cffi_tensor] = memory_holder

    # Second, defer to take_ownership_of_arrays to take ownership of everything else
    # This function assumes that the memory holder has already been assigned to
    # the weak key dictionary.
    take_ownership_of_arrays(cffi_tensor)


def take_ownership_of_tensor(cffi_tensor) -> None:
    """Take ownership of taco tensor that was allocated elsewhere entirely.

    This is not needed by pure Tensora. But some low-level programs might
    allocate a taco tensor and return it to Python. If the taco_tensor_t will be
    owned by Python, then this function can be used to take ownership of all the
    memory so that it is freed when the cffi object is freed.

    Args:
        cffi_tensor: A cffi taco_tensor_t*.
    """
    global_weakkeydict[cffi_tensor] = {"tensor": tensor_cdefs.gc(cffi_tensor, tensor_lib.free)}
    take_ownership_of_tensor_members(cffi_tensor)


def weakly_increasing(list: list[int]):
    return all(x <= y for x, y in pairwise(list))
