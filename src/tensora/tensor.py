__all__ = ['Tensor']

import itertools
from numbers import Real
from typing import List, Tuple, Dict, Iterable, Union, Any, Iterator, Optional

from .format import Mode, Format, parse_format
from .compile import taco_structure_to_cffi


class Tensor:
    """Tensor with arbitrarily dense or sparse dimensions.

    This is a thin Python wrapper around a taco structure stored in C and managed by cffi. An instance should be
    constructed via the `Tensor.from_*` static methods.
    """

    def __init__(self, cffi_tensor):
        self.cffi_tensor = cffi_tensor

    @staticmethod
    def from_lol(lol, *,
                 dimensions: Optional[Tuple[int, ...]] = None, format: Union[Format, str, None] = None) -> 'Tensor':
        if dimensions is None:
            dimensions = default_lol_dimensions(lol)

        order = len(dimensions)

        if format is None:
            format = Format((Mode.dense,) * order, tuple(range(order)))

        coordinates, values = lol_to_coordinates_and_values(lol)

        return Tensor.from_aos(coordinates, values, dimensions=dimensions, format=format)

    @staticmethod
    def from_dok(dictionary: Dict[Tuple[int, ...], float], *,
                 dimensions: Optional[Tuple[int, ...]] = None, format: Union[Format, str, None] = None) -> 'Tensor':
        return Tensor.from_aos(dictionary.keys(), dictionary.values(), dimensions=dimensions, format=format)

    @staticmethod
    def from_aos(coordinates: Iterable[Tuple[int, ...]], values: Iterable[float], *,
                 dimensions: Optional[Tuple[int, ...]] = None, format: Union[Format, str, None] = None) -> 'Tensor':
        # Lengths of modes, dimensions, and elements in coordinates must be equal. Lengths of coordinates and values
        # must be equal
        if dimensions is None:
            dimensions = default_aos_dimensions(coordinates)

        if format is None:
            coordinates = list(coordinates)
            format = default_format_given_nnz(dimensions, len(coordinates))
        elif isinstance(format, str):
            format = parse_format(format).or_die()

        # Reorder with first level first, etc.
        level_dimensions = tuple(dimensions[i] for i in format.ordering)
        level_coordinates = [tuple(coordinate[i] for i in format.ordering) for coordinate in coordinates]

        tree = coordinates_to_tree(level_coordinates, values)

        indexes, vals = tree_to_indices_and_values(tree, format.modes, level_dimensions)

        cffi_modes = tuple(x.c_int for x in format.modes)

        cffi_tensor = taco_structure_to_cffi(indexes, vals, mode_types=cffi_modes, dimensions=dimensions,
                                             mode_ordering=format.ordering)

        return Tensor(cffi_tensor)

    @staticmethod
    def from_soa(coordinates: Tuple[Iterable[int], ...], values: Iterable[float], *,
                 dimensions: Optional[Tuple[int, ...]] = None, format: Union[Format, str, None] = None) -> 'Tensor':
        # Lengths of coordinates, modes, and dimensions must be equal. Lengths of elements of coordinates and values
        # must be equal

        transposed_coordinates = [*zip(*coordinates)]

        return Tensor.from_aos(transposed_coordinates, values, dimensions=dimensions, format=format)

    @staticmethod
    def from_numpy(array, *, format: Union[Format, str, None] = None):
        import numpy

        order = array.ndim
        dimensions = array.shape

        if format is None:
            format = Format((Mode.dense,) * order, tuple(range(order)))

        if numpy.ndim(array) == 0:
            # from_lol does not understand that a scalar numpy array is a scalar
            array = float(array)

        return Tensor.from_lol(array, dimensions=dimensions, format=format)

    @staticmethod
    def from_scipy_sparse(matrix, *, format: Union[Format, str, None] = None):
        import scipy.sparse as scipy_sparse

        if format is None:
            if isinstance(matrix, scipy_sparse.csc_matrix):
                format = Format((Mode.dense, Mode.compressed), (1, 0))
            elif isinstance(matrix, scipy_sparse.csr_matrix):
                format = Format((Mode.dense, Mode.compressed), (0, 1))
            else:
                format = Format((Mode.dense, Mode.compressed), (0, 1))

        soa_matrix = matrix.tocoo()

        return Tensor.from_soa((soa_matrix.row, soa_matrix.col), soa_matrix.data,
                               dimensions=matrix.shape, format=format)

    @staticmethod
    def from_scalar(scalar: float) -> 'Tensor':
        return Tensor(taco_structure_to_cffi([], [scalar], mode_types=(), dimensions=(), mode_ordering=()))

    def to_format(self, format: Union[Format, str]):
        return Tensor.from_dok(self.to_dok(), dimensions=self.dimensions, format=format)

    @property
    def order(self) -> int:
        return self.cffi_tensor.order

    @property
    def dimensions(self) -> Tuple[int, ...]:
        return tuple(self.cffi_tensor.dimensions[0:self.order])

    @property
    def modes(self) -> Tuple[Mode, ...]:
        return tuple(Mode.from_c_int(value) for value in self.cffi_tensor.mode_types[0:self.order])

    @property
    def mode_ordering(self) -> Tuple[int, ...]:
        return tuple(self.cffi_tensor.mode_ordering[0:self.order])

    @property
    def taco_indices(self) -> List[List[List[int]]]:
        from .compile import tensor_cdefs

        order = self.order
        dimensions = self.dimensions
        modes = self.modes
        mode_ordering = self.mode_ordering
        cffi_indexes = tensor_cdefs.cast('int32_t***', self.cffi_tensor.indices)

        indices = []
        nnz = 1
        for i_dimension in range(order):
            if modes[i_dimension] == Mode.dense:
                indices.append([])
                nnz *= dimensions[mode_ordering[i_dimension]]
            elif modes[i_dimension] == Mode.compressed:
                pos = list(cffi_indexes[i_dimension][0][0:nnz + 1])
                crd = list(cffi_indexes[i_dimension][1][0:pos[-1]])
                indices.append([pos, crd])
                nnz = len(crd)

        return indices

    @property
    def taco_vals(self) -> List[float]:
        from .compile import tensor_cdefs

        order = self.order
        dimensions = self.dimensions
        modes = self.modes
        mode_ordering = self.mode_ordering
        cffi_indexes = tensor_cdefs.cast('int32_t***', self.cffi_tensor.indices)

        nnz = 1
        for i_dimension in range(order):
            if modes[i_dimension] == Mode.dense:
                nnz *= dimensions[mode_ordering[i_dimension]]
            elif modes[i_dimension] == Mode.compressed:
                nnz = cffi_indexes[i_dimension][0][nnz]

        cffi_vals = tensor_cdefs.cast('double*', self.cffi_tensor.vals)
        return list(cffi_vals[0:nnz])

    @property
    def format(self) -> Format:
        return Format(self.modes, self.mode_ordering)

    def items(self) -> Iterator[Tuple[Tuple[int, ...], float]]:
        from .compile import tensor_cdefs

        order = self.order
        modes = self.modes
        dimensions = self.dimensions
        mode_ordering = self.mode_ordering
        cffi_indexes = tensor_cdefs.cast('int32_t***', self.cffi_tensor.indices)
        cffi_values = tensor_cdefs.cast('double*', self.cffi_tensor.vals)
        level_dimensions = [dimensions[i] for i in mode_ordering]

        def recurse(i_level, prefix, position):
            if i_level < order:
                if modes[i_level] == Mode.dense:
                    for index in range(level_dimensions[i_level]):
                        next_position = level_dimensions[i_level] * position + index
                        yield from recurse(i_level + 1, prefix + (index,), next_position)
                elif modes[i_level] == Mode.compressed:
                    start = cffi_indexes[i_level][0][position]
                    end = cffi_indexes[i_level][0][position + 1]

                    for next_position in range(start, end):
                        index = cffi_indexes[i_level][1][next_position]
                        yield from recurse(i_level + 1, prefix + (index,), next_position)
            else:
                coordinate = tuple(prefix[mode_ordering[i]] for i in range(order))
                yield coordinate, cffi_values[position]

        yield from recurse(0, (), 0)

    def to_dok(self, *, explicit_zeros=False) -> Dict[Tuple[int, ...], float]:
        if explicit_zeros:
            return {key: value for key, value in self.items()}
        else:
            return {key: value for key, value in self.items() if value != 0.0}

    def __add__(self, other) -> 'Tensor':
        return evaluate_binary_operator(self, other, '+')

    def __radd__(self, other) -> 'Tensor':
        return evaluate_binary_operator(other, self, '+')

    def __sub__(self, other) -> 'Tensor':
        return evaluate_binary_operator(self, other, '-')

    def __rsub__(self, other) -> 'Tensor':
        return evaluate_binary_operator(other, self, '-')

    def __mul__(self, other) -> 'Tensor':
        return evaluate_binary_operator(self, other, '*')

    def __rmul__(self, other) -> 'Tensor':
        return evaluate_binary_operator(other, self, '*')

    def __matmul__(self, other):
        return evaluate_matrix_multiplication_operator(self, other)

    def __rmatmul__(self, other):
        return evaluate_matrix_multiplication_operator(other, self)

    def __float__(self):
        from .compile import tensor_cdefs

        if self.order != 0:
            raise ValueError(f'Can only convert Tensor of order 0 to float, not order {self.order}')

        cffi_vals = tensor_cdefs.cast('double*', self.cffi_tensor.vals)
        return cffi_vals[0]

    def __getstate__(self):
        return {
            'dimensions': self.dimensions,
            'mode_types': tuple(mode.c_int for mode in self.format.modes),
            'mode_ordering': self.format.ordering,
            'indices': self.taco_indices,
            'vals': self.taco_vals,
        }

    def __setstate__(self, state):
        self.cffi_tensor = taco_structure_to_cffi(
            indices=state['indices'],
            vals=state['vals'],
            mode_types=state['mode_types'],
            dimensions=state['dimensions'],
            mode_ordering=state['mode_ordering'],
        )

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.to_dok() == other.to_dok()
        else:
            return NotImplemented

    def __repr__(self):
        return f'Tensor.from_dok({str(self.to_dok())}, dimensions={self.dimensions}, format={self.format.deparse()!r})'


def lol_to_coordinates_and_values(data: Any, keep_zero: bool = False
                                  ) -> Tuple[Iterable[Tuple[int, ...]], Iterable[float]]:
    coordinates = []
    values = []

    def recurse(tree: List[Any], indexes: Tuple[int, ...]):
        if isinstance(tree, Real):
            # A leaf was reached
            if keep_zero or tree != 0.0:
                coordinates.append(indexes)
                values.append(tree)
        else:
            for i_element, element in enumerate(tree):
                recurse(element, indexes + (i_element,))

    recurse(data, ())

    return coordinates, values


def coordinates_to_tree(coordinates: Iterable[Tuple[int, ...]], values: Iterable[float]) -> Any:
    tree = None

    def recurse(node: Dict[int, Any], remaining_coordinates: Tuple[int, ...], payload: float):
        key = remaining_coordinates[0]
        if len(remaining_coordinates) == 1:
            node[key] = node.get(key, 0.0) + payload
        else:
            if key not in node:
                node[key] = {}
            recurse(node[key], remaining_coordinates[1:], payload)

    for coordinate, value in zip(coordinates, values):
        if len(coordinate) == 0:
            # Coordinates for order-0 tensors must be handled separately from others
            if tree is None:
                tree = 0.0
            tree += value
        else:
            if tree is None:
                tree = {}
            recurse(tree, coordinate, value)

    return tree


def tree_to_indices_and_values(tree: Any, modes: Tuple[Mode, ...], dimensions: Tuple[int, ...]
                               ) -> Tuple[List[List[List[int]]], List[float]]:
    order = len(modes)

    # Initialize indexes structure
    indexes = []
    values = []
    for mode, dimension in zip(modes, dimensions):
        if mode == Mode.dense:
            indexes.append([])
        elif mode == Mode.compressed:
            indexes.append([[0], []])

    def recurse(node, i_level):
        if modes[i_level] == Mode.dense:
            iter_next_level = range(dimensions[i_level])
        elif modes[i_level] == Mode.compressed:
            idx = sorted([key for key in node.keys()])
            indexes[i_level][0].append(indexes[i_level][0][-1] + len(idx))
            indexes[i_level][1].extend(idx)

            iter_next_level = idx
        else:
            raise NotImplementedError()

        if i_level == order - 1:
            # Final level is reached; append values
            for key in iter_next_level:
                values.append(node.get(key, 0.0))
        else:
            # Still descending tree; recurse
            for key in iter_next_level:
                next_tree = node.get(key, {})
                recurse(next_tree, i_level + 1)

    if len(dimensions) == 0:
        if tree is None:
            values = [0.0]
        else:
            values = [tree]
    else:
        if tree is None:
            recurse({}, 0)
        else:
            recurse(tree, 0)

    return indexes, values


def evaluate_binary_operator(left: Union[Tensor, Real], right: Union[Tensor, Real], operator: str) -> Tensor:
    from .function import evaluate

    def indexes_string(tensor):
        return ','.join(f'i{i}' for i in range(tensor.order))

    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.dimensions != right.dimensions:
            raise ValueError(f'Cannot apply operator {operator} between tensor with dimensions {left.dimensions} and '
                             f'tensor with dimensions {right.dimensions}')

        if operator == '*':
            # Output has density of least dense tensor
            output_format = ''.join('d' if mode1 == Mode.dense and mode2 == Mode.dense else 's'
                                    for mode1, mode2 in zip(left.format.modes, right.format.modes))
        elif operator in ('+', '-'):
            # Output has density of most dense tensor
            output_format = ''.join('d' if mode1 == Mode.dense or mode2 == Mode.dense else 's'
                                    for mode1, mode2 in zip(left.format.modes, right.format.modes))
        else:
            raise NotImplementedError()

        indexes = indexes_string(left)
        return evaluate(f'output({indexes}) = left({indexes}) {operator} right({indexes})',
                        output_format, left=left, right=right)

    elif isinstance(left, Tensor) and isinstance(right, Real):
        if operator == '*':
            # Output has density of tensor
            output_format = left.format.deparse()
        elif operator in ('+', '-'):
            # Output is full dense
            output_format = 'd' * left.order
        else:
            raise NotImplementedError()

        indexes = indexes_string(left)
        return evaluate(f'output({indexes}) = left({indexes}) {operator} right',
                        output_format, left=left, right=Tensor.from_scalar(float(right)))

    elif isinstance(left, Real) and isinstance(right, Tensor):
        if operator == '*':
            # Output has density of tensor
            output_format = right.format.deparse()
        elif operator in ('+', '-'):
            # Output is full dense
            output_format = 'd' * right.order
        else:
            raise NotImplementedError()

        indexes = indexes_string(right)
        return evaluate(f'output({indexes}) = left {operator} right({indexes})',
                        output_format, left=Tensor.from_scalar(float(left)), right=right)

    else:
        return NotImplemented


def evaluate_matrix_multiplication_operator(left: Tensor, right: Tensor):
    from .function import evaluate

    if isinstance(left, Tensor) and isinstance(right, Tensor):
        if left.order == 1 and right.order == 1:
            scalar_tensor = evaluate('output = left(i) * right(i)', '', left=left, right=right)
            return float(scalar_tensor)
        elif left.order == 2 and right.order == 1:
            # Output format is the uncontracted dimension of the matrix
            output_format = left.format.modes[left.format.ordering[0]].character
            return evaluate('output(i) = left(i,j) * right(j)', output_format, left=left, right=right)
        elif left.order == 1 and right.order == 2:
            # Output format is the uncontracted dimension of the matrix
            output_format = right.format.modes[right.format.ordering[1]].character
            return evaluate('output(j) = left(i) * right(i,j)', output_format, left=left, right=right)
        elif left.order == 2 and right.order == 2:
            # Output format are the uncontracted dimensions of the matrices
            left_output_format = left.format.modes[left.format.ordering[0]].character
            right_output_format = right.format.modes[right.format.ordering[1]].character
            output_format = left_output_format + right_output_format
            return evaluate('output(i,k) = left(i,j) * right(j,k)', output_format, left=left, right=right)
        else:
            raise ValueError(f'Matrix multiply is only defined between tensors of orders 1 and 2, not orders '
                             f'{left.order} and {right.order}')
    else:
        return NotImplemented


def default_lol_dimensions(lol) -> Tuple[int, ...]:
    """Extract dimensions from dense list-of-lists.

    Given nested lists of lists representing a dense tensor in row-major format, discover the dimensions of the tensor
    as implied by the lengths of the lists. The length of the top-level list is the size of the first dimension, the
    length of the first element of that list is the size of the second dimension, and so on until a scalar is
    encountered. For example, `infer_lol_dimensions([[1,2,3],[4,5,6]])` returns `(2,3)`.

    A scalar value is allowed, which translates correctly to dimensions equal to `()`.

    Not all tensors can have their dimensions inferred from a dense LOL. A tensor with dimensions `(0,2)` will never be
    inferred because its dense LOL representation is `[]`, which would actually be inferred as having dimensions `(0,)`.
    """
    dimensions = []
    subdata = lol
    while True:
        if isinstance(subdata, list):
            dimensions.append(len(subdata))
            if len(subdata) > 0:
                subdata = subdata[0]
            else:
                break
        else:
            break

    return tuple(dimensions)


def default_aos_dimensions(coordinates: Iterable[Tuple[int, ...]]) -> Tuple[int, ...]:
    order = None
    maximums = []
    for coordinate in coordinates:
        if order is None:
            order = len(coordinate)
            maximums = list(coordinate)
        else:
            if len(coordinate) != order:
                raise ValueError(f'All coordinates must be the same length; the first coordinate has length'
                                 f'{order}, but this coordinate is not that length: {coordinate}')

            for i, (dimension, index) in enumerate(zip(maximums, coordinate)):
                if index > dimension:
                    maximums[i] = coordinate[i]

    # i+ 1 is the length of a dimension whose largest index is i
    return tuple(i + 1 for i in maximums)


def default_format_given_nnz(dimensions: Tuple[int, ...], nnz: int) -> Format:
    # The default format is to use dense dimensions as long as the number of nonzeros is larger than the product
    # of those dimensions.
    needed_dense = 0
    required_threshold = 1
    for needed_dense, dimension in enumerate(dimensions):
        required_threshold *= dimension
        if nnz < required_threshold:
            break

    return Format((Mode.dense,) * needed_dense + (Mode.compressed,) * (len(dimensions) - needed_dense),
                  tuple(range(len(dimensions))))


def taco_indexes_from_aos_coordinates(coordinates: Iterable[Tuple[int, ...]], values: Iterable[float], *,
                                      modes: Tuple[Mode, ...], dimensions=Tuple[int, ...]):  # pragma: no cover
    # This is an experimental alternative to coordinates_to_tree and tree_to_indices_and_values. It is not currently
    # used anywhere.

    class CartesianAppend(Iterable):
        # Does a cartesian product, but assumes
        def __init__(self, base: Iterable[List[Any]], append: Iterable[Any]):
            self.base = base
            self.append = append

        def __iter__(self):
            for prefix, suffix in itertools.product(self.base, self.append):
                if isinstance(prefix, tuple):
                    yield prefix + (suffix,)
                else:
                    yield prefix + [suffix]

    # Sort coordinates
    sorted_coordinates_and_values = sorted(zip(coordinates, values), key=lambda x: x[0])

    # Remove duplicates
    deduplicated_coordinates = []
    deduplicated_values = []
    previous_coordinate = None
    for coordinate, value in sorted_coordinates_and_values:
        if coordinate == previous_coordinate:
            deduplicated_values[-1] += value
        else:
            deduplicated_coordinates.append(coordinate)
            deduplicated_values.append(value)

    # To SoA coordinates for easier handling
    soa_coordinates = [*zip(*deduplicated_coordinates)]

    # Build taco levels from sorted, deduplicated, SoA coordinates
    levels = []
    previous_prefixes = [()]
    for i_level, (mode, dimension) in enumerate(zip(modes, dimensions)):
        if mode == Mode.dense:
            levels.append([dimension])
            previous_prefixes = CartesianAppend(previous_prefixes, range(dimension))
        elif mode == Mode.compressed:
            # There needs to be one entry in pos per previous prefix indicating the position in idx when this prefix
            # is encountered
            previous_prefix_iterator = iter(previous_prefixes)
            current_prefix = None

            pos = []
            idx = []
            unique_coordinates = []
            previous_coordinate = None
            for coordinate in zip(*soa_coordinates[0:i_level + 1]):
                if coordinate != previous_coordinate:
                    while coordinate[0:-1] != current_prefix:
                        # The prefix has changed. Mark in pos the position of this prefix. Some prefixes may be
                        # empty (e.g. the prefix came from a dense range); in such a case, we need to emit the
                        # current position multiple times until the current prefix is found.
                        current_prefix = next(previous_prefix_iterator)
                        pos.append(len(idx))

                    # Store coordinate of this level in idx
                    idx.append(coordinate[-1])

                    unique_coordinates.append(coordinate)
                    previous_coordinate = coordinate

            levels.append([pos, idx])
            previous_prefixes = unique_coordinates
        else:
            raise RuntimeError(f'Unknown mode: {mode}')

    return levels
