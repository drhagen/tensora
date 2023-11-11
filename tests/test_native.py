from tensora.native import generate_code, KernelType


def test_native_codegen():
    assignment = "f(i) = A0(i) + A1(i,j) * x(j) + A2(i,k,l) * x(k) * x(l)"
    output_format = "d"
    input_formats = {"A0": "d", "A1": "ds", "A2": "dss", "x": "d"}
    kernel_type = KernelType.compute
    code = generate_code(assignment, output_format, input_formats, kernel_type)
    assert isinstance(code, str)
