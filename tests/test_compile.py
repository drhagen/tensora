import tempfile

from cffi import FFI

from tensora import Tensor
from tensora.compile import tensor_cdefs, taco_define_header, taco_type_header, lock, \
    take_ownership_of_tensor_members, take_ownership_of_tensor

source = """
taco_tensor_t create_tensor() {
    int32_t order = 2;

    int32_t* dimensions = malloc(sizeof(int32_t) * order);
    dimensions[0] = 3;
    dimensions[1] = 4;

    int32_t* mode_ordering = malloc(sizeof(int32_t) * order);
    mode_ordering[0] = 0;
    mode_ordering[1] = 1;

    taco_mode_t* mode_types = malloc(sizeof(taco_mode_t) * order);
    mode_types[0] = taco_mode_dense;
    mode_types[1] = taco_mode_sparse;

    int32_t* pos = malloc(sizeof(*pos) * 4);
    pos[0] = 0;
    pos[1] = 3;
    pos[2] = 3;
    pos[3] = 5;
    int32_t* crd = malloc(sizeof(*crd) * 5);
    crd[0] = 0;
    crd[1] = 2;
    crd[2] = 3;
    crd[3] = 0;
    crd[4] = 3;

    int32_t** indices1 = malloc(sizeof(*indices1) * 2);
    indices1[0] = pos;
    indices1[1] = crd;

    int32_t*** indices = malloc(sizeof(*indices) * order);
    indices[0] = NULL;
    indices[1] = indices1;

    double* vals = malloc(sizeof(*vals) * 5);
    vals[0] = 6;
    vals[1] = 9;
    vals[2] = 8;
    vals[3] = 5;
    vals[4] = 7;

    taco_tensor_t tensor = {
        .order = order,
        .dimensions = dimensions,
        .csize = 8,
        .mode_ordering = mode_ordering,
        .mode_types = mode_types,
        .indices = (uint8_t***)indices,
        .vals = (uint8_t*)vals,
        .vals_size = 5
    };
    return tensor;
}

taco_tensor_t* create_pointer_to_tensor() {
    taco_tensor_t* tensor = malloc(sizeof(taco_tensor_t));
    *tensor = create_tensor();
    return tensor;
}
"""

ffi = FFI()
ffi.include(tensor_cdefs)
ffi.cdef("""
taco_tensor_t create_tensor();
taco_tensor_t* create_pointer_to_tensor();
""")
ffi.set_source('taco_kernel', taco_define_header + taco_type_header + source,
               extra_compile_args=['-Wno-unused-variable', '-Wno-unknown-pragmas'])

expected_tensor = Tensor.from_lol([[6, 0, 9, 8], [0, 0, 0, 0], [5, 0, 0, 7]], format='ds')

with tempfile.TemporaryDirectory() as temp_dir:
    # Lock because FFI.compile is not thread safe: https://foss.heptapod.net/pypy/cffi/-/issues/490
    with lock:
        # Create shared object in temporary directory
        lib_path = ffi.compile(tmpdir=temp_dir)

    # Load the shared object
    lib = ffi.dlopen(lib_path)


def test_take_ownership_of_tensor_on_returned_struct():
    cffi_tensor = lib.create_tensor()
    take_ownership_of_tensor_members(cffi_tensor)
    tensor = Tensor(cffi_tensor)
    assert tensor == expected_tensor


def test_take_ownership_of_tensor_on_returned_pointer_to_struct():
    cffi_tensor = lib.create_pointer_to_tensor()
    take_ownership_of_tensor(cffi_tensor)
    tensor = Tensor(cffi_tensor)
    assert tensor == expected_tensor
