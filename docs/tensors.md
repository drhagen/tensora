---
icon: material/cube-outline
---

# Tensors

The main type in Tensora is the `Tensor` class. `Tensor`s are immutable. New tensors may be constructed from operations on other `Tensor`s, but no property of a `Tensor` may change once it is constructed. This is different from NumPy arrays and Scipy matrices, which may be mutated in-place.

## Attributes

The order, dimensions, and format are the fundamental structural properties of a tensor. These are available as attributes of a `Tensor`.

### `tensor.order`

The order of a tensor is the number of dimensions it has. A scalar is a 0-order tensor, a vector is a 1-order tensor, a matrix is a 2-order tensor, and so on. Conceptually, the order may be any non-negative integer, but realistically, a large enough number of dimensions will cause a stack overflow or other resource error.

```python
from tensora import Tensor

tensor = Tensor.from_lol([[1,2,3], [4,5,6]])
assert tensor.order == 2
```

### `tensor.dimensions`

Each element of the `dimensions` tuple is the size of the corresponding dimension.

```python
from tensora import Tensor

tensor = Tensor.from_lol([[1,2,3], [4,5,6]])
assert tensor.dimensions == (2, 3)
```

### `tensor.format`

The type of `format` is a `tensora.Format` object, which has `modes` and `ordering` attributes. The `format.deparse()` method will give you a human-readable string.

```python
from tensora import Tensor

tensor = Tensor.from_lol([[1,2,3], [4,5,6]])
assert tensor.format.deparse() == 'dd'
```

## Arithmetic

The normal way to perform mathematical operations on `Tensor`s is to use the `evaluate` function. However, the `Tensor` class implements several of the standard arithmetic operations available in Python. Tensora makes some guesses on the format of the result. If more control is needed use `evaluate`.

### `tensor1 + tensor2` and `tensor1 - tensor2`

Addition and subtraction are element-wise operations. If both operands are `Tensor`s, they must have the same order and dimensions. The result will be a `Tensor` where each dimension is dense if either operand is dense at that dimension. If one of the operands is a Python scalar, it will be broadcast to the dimensions of the other operand. The result will be a `Tensor` with the same order, dimensions, and format as the other operand.

```python
from tensora import Tensor

tensor1 = Tensor.from_lol([[1,2,3], [4,5,6]])
tensor2 = Tensor.from_lol([[7,8,9], [10,11,12]])

assert tensor1 + tensor2 == Tensor.from_lol([[8,10,12], [14,16,18]])
assert tensor1 - tensor2 == Tensor.from_lol([[-6,-6,-6], [-6,-6,-6]])
```

### `tensor1 * tensor2`

Multiplication is and element-wise operation. If both operands are `Tensor`s, they must have the same order and dimensions. The result will be a `Tensor` where each dimension is sparse if either operand is sparse at that dimension. If one of the operands is a Python scalar, it will be broadcast to the dimensions of the other operand. The result will be a `Tensor` with the same order, dimensions, and format as the other operand.

```python
from tensora import Tensor

tensor1 = Tensor.from_lol([[1,2,3], [4,5,6]])
tensor2 = Tensor.from_lol([[7,8,9], [10,11,12]])

assert tensor1 * tensor2 == Tensor.from_lol([[7,16,27], [40,55,72]])
```

### `tensor1 @ tensor2`

Matrix multiplication is only permitted between vectors (order-1 tensors) and matrices (order-2 tensors). The dimensions of the operands must be compatible like normal and as in the table below. The result is a `Tensor` with the the expected dimensions. The format of the result is determined by the format of the operand dimensions that give the result dimension its size.

| `a`    | `b`    | `a @ b` | assignment                 |
|--------|--------|---------|----------------------------|
| (n,)   | (n,)   | ()      | `c() = a(i) * b(i)`        |
| (n,)   | (n, p) | (p,)    | `c(j) = a(i) * b(i,j)`     |
| (m, n) | (n,)   | (m,)    | `c(i) = a(i,j) * b(j)`     |
| (m, n) | (n, p) | (m, p)  | `c(i,j) = a(i,k) * b(k,j)` |

```python
from tensora import Tensor

A = Tensor.from_lol([[1,2,3], [4,5,6]])
x = Tensor.from_lol([1,2,3])

assert A @ x == Tensor.from_lol([14, 32])
```