---
icon: material/sign-direction
---

# Getting started

Tensors are n-dimensional generalizations of matrices. Instead of being limited to 2 dimensions, tensors may have 3, 4, or more dimensions. They may also have 0 or 1 dimensions. The number of dimensions is the called the order. NumPy is the best known tensor library in Python; its central `ndarray` object is an example of a dense tensor.

Each dimension of a tensor has a size. This determines, conceptually, the number of elements in the tensor. "Conceptually" because the number of stored elements and the amount of memory required for the tensor may be smaller than that if the tensor is sparse.

Tensors also have a format. The format has a list of modes, which determines the internal layout of the tensor, and a mode ordering, which maps each dimension to each mode. Each mode can be either sparse or dense. An example of two different formats with the same internal layout would be CSR, which has format `ds` in Tensora, and CSC, which has format `d1s0`.

Here are a list of common formats:

| common name   | Tensora format |
|---------------|----------------|
| scalar        | `''`           |
| dense vector  | `'d'`          |
| sparse vector | `'s'`          |
| row-major     | `'dd'`         |
| column-major  | `'d1d0'`       |
| CSR           | `'ds'`         |
| CSC           | `'d1s0'`       |
| CSF           | `'sd'`         |
| DCSR          | `'ss'`         |

There are formats for higher order tensors, but they do not have common names. That is one of the goals of Tensora, to give access to the creation and use of new formats.

Tensors are [created](./creation.md) via one of several static methods on the `Tensor` class. The key [attributes](./tensors.md), `order`, `dimensions`, and `format`, are available on every `Tensor`. While basic [arithmetic](./tensors.md#arithmetic) (`+`, `-`, `*`, `@`) is available as well, it is generally better to use the `evaluate` function, which makes much more complex operations available and will fuse the loops of multiple arithmetic operators.
