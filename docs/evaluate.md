---
icon: material/function-variant
---

# Evaluate

```
evaluate(
    assignment: str,
    output_format: Format | str,
    *,
    **inputs: Tensor,
) -> Tensor
```

The main entry point for mathematical operations in Tensora is the `evaluate` function. It takes a tensor algebra assignment and a list of `Tensor` objects. It returns a new `Tensor`, having evaluated the expression according to the input `Tensor`s.

* `assignment` is parsable as an algebraic tensor assignment.

* `output_format` is the desired format of the output tensor.

* `inputs` is all the inputs to the expression. There must be one named argument for each variable name in `assignment`. The dimensions of the tensors in `inputs` must be consistent with `assignment` and with each other.

There is also `evaluate_tensora` and `evaluate_taco` that have identical interfaces, but use different tensor algebra compilers. `evaluate` is an alias for the default, which is currently `evaluate_tensora`. `evaluate_taco` is only available if the `tensora[taco]` extra is installed.

```python
from tensora import Tensor, evaluate

A = Tensor.from_lol([[1,2,3], [4,5,6]])
x = Tensor.from_lol([1,2,3])

y = evaluate('y(i) = A(i,j) * x(j)', 'd', A=A, x=x)
assert y == Tensor.from_lol([14, 32])
```

## Assignments

In a loose sense, the assignment strings use Einstein notation. The assignments are made of tensor names, index names, and operations. A tensor with its indexes is the target of the assignment on the left-hand side. Various tensors with their indexes are connected by elementary operations on the right-hand side.

### Output indexes

Indexes that appear on both sides match an output dimension to the input dimensions sharing that index.

```python
from tensora import Tensor, evaluate

a = Tensor.from_lol([1,2,3])
b = Tensor.from_lol([4,5,6])

c = evaluate('c(i) = a(i) * b(i)', 'd', a=a, b=b)
assert c == Tensor.from_lol([4, 10, 18])
```

### Contraction indexes

Indexes that appear only on the right-hand side are summed over, also known as a contraction.

```python
from tensora import Tensor, evaluate

A = Tensor.from_lol([[1,2,3], [4,5,6]])

a = evaluate('a(i) = A(i,j)', '', A=A)
assert a == Tensor.from_lol([6, 15])

b = evaluate('b(j) = A(i,j)', '', A=A)
assert b == Tensor.from_lol([5, 7, 9])
```

This commonly appears in the context of multiplication, in which it called an inner product.

```python
from tensora import Tensor, evaluate

a = Tensor.from_lol([1,2,3])
b = Tensor.from_lol([4,5,6])

c = evaluate('c() = a(i) * b(i)', '', a=a, b=b)
assert c == 32
```

### Broadcasting indexes

Indexes that appear only on the left-hand side would be interpreted as broadcasting the value of the right-hand side along that dimension.

This operation is not currently allowed by `evaluate` because that indicates that the expression should be broadcast along that target dimension, but there is currently no way to specify the size of that dimension. It is allowed by the `tensora` CLI, however.

```python
from tensora import Tensor, evaluate

a = Tensor.from_lol(1)

b = evaluate('b(i) = a()', 'd', a=a)
# BroadcastTargetIndexError: Expected index variable i on the target variable
# to be mentioned on the right-hand side, but it was not: b(i) = a(). Such
# broadcasting makes sense in a kernel and those kernels can be generated, but
# they cannot be used in `evaluate` or `tensor_method` because those functions
# get the output dimensions from the the dimensions of the input tensors.
```

### Reusing tensors

Tensor names may be repeated, possibly with different indexes. The tensor can and should only be provided once; it will be used for all occurrences of that tensor name in the assignment.

```python
from tensora import Tensor, evaluate

x = Tensor.from_lol([1,2,3])
V = Tensor.from_lol([[1,2,3], [4,5,6], [7,8,9]])

y = evaluate('y() = x(i) * V(i,j) * x(j)', '', x=x, V=V)
assert y == 228
```

### Diagonal indexes

Indexes may *not* be repeated within a tensor. Such syntax would represent a diagonal operation, which is currently not supported.

```python
from tensora import Tensor, evaluate

V = Tensor.from_lol([[1,2,3], [4,5,4], [3,2,1]])

v = evaluate('v(i) = V(i,i)', 'd', V=V)
# DiagonalAccessError: Diagonal access to a tensor (i.e. repeating the same
# index within a tensor) is not currently supported: V(i, i)
```
