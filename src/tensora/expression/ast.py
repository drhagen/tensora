from __future__ import annotations

__all__ = [
    "Expression",
    "Literal",
    "Integer",
    "Float",
    "Tensor",
    "Add",
    "Subtract",
    "Multiply",
    "Assignment",
]

from abc import abstractmethod
from dataclasses import dataclass


class Expression:
    __slots__ = ()

    @abstractmethod
    def variables(self) -> dict[str, list[Tensor]]:
        raise NotImplementedError()

    @abstractmethod
    def deparse(self) -> str:
        """Convert the assignment back into a string."""
        raise NotImplementedError()

    @abstractmethod
    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        """Map of index name to tensors and dimensions it exists in.

        Returns:
            A mapping where each key is the string name of an index and each value is a sets of
            pairs. In each pair, the first element is the name of a tensor and the second element
            is the dimension in which that index appears.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.deparse()


class Literal(Expression):
    __slots__ = ()

    def variables(self) -> dict[str, list[Tensor]]:
        return {}

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return {}


@dataclass(frozen=True, slots=True)
class Integer(Literal):
    value: int

    def deparse(self):
        return str(self.value)


@dataclass(frozen=True, slots=True)
class Float(Literal):
    value: float

    def deparse(self):
        return str(self.value)


@dataclass(frozen=True, slots=True)
class Tensor(Expression):
    name: str
    indexes: tuple[str, ...]

    @property
    def order(self):
        return len(self.indexes)

    def variables(self) -> dict[str, list[Tensor]]:
        return {self.name: [self]}

    def deparse(self):
        return self.name + "(" + ",".join(self.indexes) + ")"

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        participants = {}
        for i, index_name in enumerate(self.indexes):
            participants[index_name] = participants.get(index_name, set()) | {(self.name, i)}
        return participants


def merge_index_participants(left: Expression, right: Expression):
    left_indexes = left.index_participants()
    right_indexes = right.index_participants()
    return {
        index_name: left_indexes.get(index_name, set()) | right_indexes.get(index_name, set())
        for index_name in {*left_indexes.keys(), *right_indexes.keys()}
    }


@dataclass(frozen=True, slots=True)
class Add(Expression):
    left: Expression
    right: Expression

    def variables(self) -> dict[str, list[Tensor]]:
        variables_mapping = self.left.variables().copy()
        for name, variables in self.right.variables().items():
            if name in variables_mapping:
                variables_mapping[name] = [*variables_mapping[name], *variables]
            else:
                variables_mapping[name] = variables
        return variables_mapping

    def deparse(self):
        left_string = self.left.deparse()

        right_string = self.right.deparse()
        if isinstance(self.right, (Add, Subtract)):
            # Preserve AST even though addition is associative.
            right_string = f"({right_string})"

        return left_string + " + " + right_string

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True, slots=True)
class Subtract(Expression):
    left: Expression
    right: Expression

    def variables(self) -> dict[str, list[Tensor]]:
        variables_mapping = self.left.variables().copy()
        for name, variables in self.right.variables().items():
            if name in variables_mapping:
                variables_mapping[name] = [*variables_mapping[name], *variables]
            else:
                variables_mapping[name] = variables
        return variables_mapping

    def deparse(self):
        left_string = self.left.deparse()

        right_string = self.right.deparse()
        if isinstance(self.right, (Add, Subtract)):
            right_string = f"({right_string})"

        return left_string + " - " + right_string

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True, slots=True)
class Multiply(Expression):
    left: Expression
    right: Expression

    def variables(self) -> dict[str, list[Tensor]]:
        variables_mapping = self.left.variables().copy()
        for name, variables in self.right.variables().items():
            if name in variables_mapping:
                variables_mapping[name] = [*variables_mapping[name], *variables]
            else:
                variables_mapping[name] = variables
        return variables_mapping

    def deparse(self):
        left_string = self.left.deparse()
        if isinstance(self.left, (Add, Subtract)):
            left_string = f"({left_string})"

        right_string = self.right.deparse()
        # Preserve AST even though multiplication is associative.
        if isinstance(self.right, (Add, Subtract, Multiply)):
            right_string = f"({right_string})"

        return f"{left_string} * {right_string}"

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True)
class Assignment:
    target: Tensor
    expression: Expression

    def __post_init__(self):
        from ._exceptions import (
            InconsistentDimensionsError,
            MutatingAssignmentError,
            NameConflictError,
        )

        target_name = self.target.name

        # Mutable containers
        variable_orders: dict[str, int] = {target_name: self.target.order}
        index_names = set(self.target.indexes)

        # Validate expression
        variables_mapping = self.expression.variables()
        for name, variables in variables_mapping.items():
            if name == target_name:
                raise MutatingAssignmentError(self)

            for variable in variables:
                index_names.update(variable.indexes)

            (first, *rest) = variables
            for variable in rest:
                if first.order != variable.order:
                    raise InconsistentDimensionsError(self, first, variable)

            variable_orders[name] = first.order

        # Detect name conflicts
        conflicted_names = index_names.intersection(variable_orders.keys())
        if len(conflicted_names) > 0:
            raise NameConflictError(conflicted_names.pop(), self)

        self._variable_orders: dict[str, int]
        object.__setattr__(self, "_variable_orders", variable_orders)

    def deparse(self) -> str:
        return self.target.deparse() + " = " + self.expression.deparse()

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.target, self.expression)

    def variable_orders(self) -> dict[str, int]:
        """Number of dimensions of each variable.

        Returns:
            A mapping where each key is the string name of a variable and each value is the number
            of dimensions that variable has.
        """

        return self._variable_orders

    def __str__(self) -> str:
        return self.deparse()
