---
icon: material/tree
---

# Creation

Creating a `Tensor` is best done via the `Tensor.from_*` methods. These methods convert a variety of data types into a `Tensor`. Most of the conversion methods optionally take both dimensions and format to determine the `dimensions` and `format` of the resulting tensor.

## `Tensor.from_lol`: list of lists

```
Tensor.from_lol(
    lol,
    *,
    dimensions: tuple[int, ...] = None,
    format: Format | str = None,
)
```

Convert a nested list of lists to a `Tensor`.

* `lol` is a list of lists, possibly deeply nested. That is, `lol` is a `float`, a `list[float]`, a `list[list[float]]`, etc. to an arbitrary depth of `list`s. The values are read in row-major format, meaning the top-level list is the first dimension and the deepest list (the one containing actual scalars) is the last dimension. All lists at the same level must have the same length. Note that these "lists" may be `Iterable`s. For those familiar, this is identical to the NumPy behavior when constructing an array from lists of lists via `numpy.array`.

* `dimensions` has a default value that is inferred from the structure of `lol`. If provided, it must be consistent with the structure of `lol`. Providing the dimensions is typically only useful when one or more non-final dimensions may have size zero. For example, `Tensor.from_lol([[], []])` has dimensions of `(2,0)`, while `Tensor.from_lol([[], []], dimensions=(2,0,3))` has dimensions of `(2,0,3)`.

* `format` has a default value of all dense dimensions.

```python
from tensora import Tensor

tensor = Tensor.from_lol([[1,2,3], [4,5,6]])

assert tensor.dimensions == (2, 3)
```

This is also the best way to create a scalar `Tensor` because passing a single number to this method means the list nesting is 0 levels deep and is therefore a 0-order tensor.

```python
from tensora import Tensor

tensor = Tensor.from_lol(2.5)

assert tensor.dimensions == ()
```

## `Tensor.from_dok`: dictionary of keys

```
Tensor.from_dok(
    dok: dict[tuple[int, ...], float],
    *,
    dimensions: tuple[int, ...] = None,
    format: Format | str = None,
)
```

Convert a dictionary of keys to a `Tensor`.

* `dok` is a Python dictionary where each key is the coordinate of one nonzero value and the value of the entry is the value of the tensor at that coordinate. All coordinates not mentioned are implicitly zero.

* `dimensions` has a default value that is the largest size in each dimension found among the coordinates.

* `format` has a default value of dense dimensions as long as the number of nonzeros is larger than the product of those dimensions and then sparse dimensions after that. The default value is subject to change with experience.

```python
from tensora import Tensor

tensor = Tensor.from_dok({
    (1,0): 2.0,
    (0,1): -2.0,
    (1,2): 4.0,
}, dimensions=(2,3), format='ds')

assert tensor == Tensor.from_lol([[0,-2,0], [2,0,4]])
```

## `Tensor.from_aos`: array of structs

```
Tensor.from_aos(
    aos: Iterable[tuple[int, ...]],
    values: Iterable[float],
    *,
    dimensions: tuple[int, ...] = None,
    format: Format | str = None,
)
```

Convert a list of coordinates and a corresponding list of values to a `Tensor`.

* `aos` is an iterable of the coordinates of the nonzero values.

* `values` must be the same length as `aos` and each value is the value at the corresponding coordinate.

* `dimensions` has the same default as `Tensor.from_dok`, the largest size in each dimension.

* `format`has the same default as `Tensor.from_dok`, dense for an many dimensions as needed to fit the nonzeros.

```python
from tensora import Tensor

tensor = Tensor.from_aos(
    [(1,0), (0,1), (1,2)],
    [2.0, -2.0, 4.0],
    dimensions=(2,3),
    format='ds',
)

assert tensor == Tensor.from_lol([[0,-2,0], [2,0,4]])
```

## `Tensor.from_soa`: struct of arrays

```
Tensor.from_soa(
    soa: tuple[Iterable[int], ...],
    values: Iterable[float],
    *,
    dimensions: tuple[int, ...] = None,
    format: Format | str = None,
)
```

Convert lists of indexes for each dimension and a corresponding list of values to a `Tensor`.

* `soa` is a tuple of iterables, where each iterable is all the indexes of the corresponding dimension. All iterables must be the same length.

* `values` must be the same length as the iterables in `soa` and each value is the nonzero value at the corresponding coordinate.

* `dimensions` has the same default as `Tensor.from_dok`, the largest size in each dimension.

* `format` has the same default as `Tensor.from_dok`, dense for an many dimensions as needed to fit the nonzeros.

```python
from tensora import Tensor

tensor = Tensor.from_soa(
    ([1,0,1], [0,1,2]),
    [2.0, -2.0, 4.0],
    dimensions=(2,3),
    format='ds',
)

assert tensor == Tensor.from_lol([[0,-2,0], [2,0,4]])
```

## `Tensor.from_numpy`: convert a NumPy array

```
Tensor.from_numpy(
    array: numpy.ndarray,
    *,
    format: Format | str = None,
)
```

Convert a NumPy array to a `Tensor`.

* `array` is any `numpy.ndarray`. The resulting `Tensor` will have the same order, dimensions, and values of this array.

* `format` has a default value of all dense dimensions.

```python
import numpy as np
from tensora import Tensor

array = np.array([[1,2,3], [4,5,6]])
tensor = Tensor.from_numpy(array)

assert tensor == Tensor.from_lol([[1,2,3], [4,5,6]])
```

## `Tensor.from_scipy_sparse`: convert a SciPy sparse matrix

```
Tensor.from_scipy_sparse(
    matrix: scipy.sparse.spmatrix,
    *,
    format: Format | str = None,
)
```

Convert a SciPy sparse matrix to a `Tensor`.

* `matrix` is any `scipy.sparse.spmatrix`. The resulting `Tensor` will have the same order, dimensions, and values of this matrix. The tensor will always have order 2.

* `format` has a default value of `ds` for `csr_matrix` and `d1s0` for `csc_matrix` and also `ds` for the other sparse matrix types, though that is subject to changes as Tensora adds new format mode types.

```python
import scipy.sparse as sp
from tensora import Tensor

matrix = sp.csr_matrix(([2.0, -2.0, 4.0], ([1,0,1], [0,1,2])), shape=(2,3))
tensor = Tensor.from_scipy_sparse(matrix)

assert tensor.format.deparse() == 'ds'
assert tensor == Tensor.from_lol([[0,-2,0], [2,0,4]])
```
