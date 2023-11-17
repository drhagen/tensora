from tensora.native import KernelType, generate_c_code


def test_copy():
    assignment = "a(i) = b(i)"
    formats = {"a": "d", "b": "d"}
    kernel_type = KernelType.compute
    code = generate_c_code(assignment, formats, kernel_type)
    assert isinstance(code, str)


def test_csr_vector_product():
    assignment = "y(i) = A(i,j) * x(j)"
    formats = {"y": "d", "A": "ds", "x": "d"}
    kernel_type = KernelType.compute
    code = generate_c_code(assignment, formats, kernel_type)
    assert isinstance(code, str)


def test_csc_vector_product():
    assignment = "y(i) = A(i,j) * x(j)"
    formats = {"y": "d", "A": "d1s0", "x": "d"}
    kernel_type = KernelType.compute
    code = generate_c_code(assignment, formats, kernel_type)
    assert isinstance(code, str)


def test_csr_addition():
    assignment = "A(i,j) = B(i,j) + C(i,j)"
    formats = {"A": "ds", "B": "ds", "C": "ds"}
    kernel_type = KernelType.compute
    code = generate_c_code(assignment, formats, kernel_type)
    assert isinstance(code, str)


def test_rhs():
    assignment = "f(i) = A0(i) + A1(i,j) * x(j) + A2(i,k,l) * x(k) * x(l)"
    formats = {"f": "d", "A0": "d", "A1": "ds", "A2": "dss", "x": "d"}
    kernel_type = KernelType.compute
    code = generate_c_code(assignment, formats, kernel_type)
    assert isinstance(code, str)


def test_spmttkrp():
    assignment = "A(i,j) = B(i,k,l) * D(l,j) * C(k,j)"
    formats = {"A": "dd", "B": "sss", "D": "dd", "C": "dd"}
    kernel_type = KernelType.compute
    code = generate_c_code(assignment, formats, kernel_type)
    assert isinstance(code, str)
