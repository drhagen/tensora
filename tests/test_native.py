from tensora.native import KernelType, generate_c_code


def test_native_codegen():
    assignment = "f(i) = A0(i) + A1(i,j) * x(j) + A2(i,k,l) * x(k) * x(l)"
    formats = {"f": "d", "A0": "d", "A1": "ds", "A2": "dss", "x": "d"}
    kernel_type = KernelType.compute
    code = generate_c_code(assignment, formats, kernel_type)
    assert isinstance(code, str)
