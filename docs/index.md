# Tensora

Tensora is a tensor algebra library for Python. You can create `Tensor` objects in a variety of sparse and dense formats. You can do calculations with these tensors by passing the them to the `evaluate` function along with an expression (e.g. `y = evaluate('y(i) = A(i,j) * x(j)', A=A, x=x)`). The expression is parsed, a kernel is generated, the C code is compiled on the fly, the binary is invoked, and the result is packaged into a output `Tensor`.

Tensora also comes with the `tensora` command line tool that can be used to generate the kernel code for external use.

Tensora is based on the [Tensor Algebra Compiler](http://tensor-compiler.org/) (TACO) and can use either TACO or its own compiler to generate kernels.

## Installation

The recommended means of installation is with `pip` from PyPI.

```bash
pip install tensora
```

You must have a C compiler installed that is compatible with and available to [CFFI API mode](https://cffi.readthedocs.io/en/latest/overview.html#purely-for-performance-api-level-out-of-line). Tensora definitely works on Mac and Linux and definitely does not work on Windows. It will probably work on any system where [`ffi.dlopen(None)`](https://cffi.readthedocs.io/en/latest/cdef.html#ffi-dlopen-loading-libraries-in-abi-mode) works.

## Hello world

Here is an example of multiplying a sparse matrix in CSR format with a dense vector:

```python
from tensora import Tensor, evaluate

elements = {
    (1,0): 2.0,
    (0,1): -2.0,
    (1,2): 4.0,
}

A = Tensor.from_dok(elements, dimensions=(2,3), format='ds')
x = Tensor.from_lol([0, -1, 2])

y = evaluate('y(i) = A(i,j) * x(j)', 'd', A=A, x=x)

assert y == Tensor.from_lol([2,8])
```
