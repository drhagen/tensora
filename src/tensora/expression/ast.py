from __future__ import annotations

__all__ = [
    "Expression",
    "Literal",
    "Integer",
    "Float",
    "Variable",
    "Scalar",
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
    def variables(self) -> dict[str, list[Variable]]:
        raise NotImplementedError()

    @abstractmethod
    def deparse(self) -> str:
        """Convert the assignment back into a string."""
        pass

    @abstractmethod
    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        """Map of index name to tensors and dimensions it exists in.

        Returns:
            A mapping where each key is the string name of an index and each value is a sets of
            pairs. In each pair, the first element is the name of a tensor and the second element
            is the dimension in which that index appears.
        """
        pass

    def __str__(self):
        return self.deparse()


class Literal(Expression):
    __slots__ = ()

    def variables(self) -> dict[str, list[Variable]]:
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


class Variable(Expression):
    __slots__ = ()

    name: str
    indexes: list[str]
    order: int | None

    def variables(self) -> dict[str, list[Variable]]:
        return {self.name: [self]}

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        participants = {}
        for i, index_name in enumerate(self.indexes):
            participants[index_name] = participants.get(index_name, set()) | {(self.name, i)}
        return participants


@dataclass(frozen=True, slots=True)
class Scalar(Variable):
    name: str

    @property
    def order(self):
        return None

    @property
    def indexes(self):
        return []

    def deparse(self):
        return self.name


@dataclass(frozen=True, slots=True)
class Tensor(Variable):
    name: str
    indexes: list[str]

    @property
    def order(self):
        return len(self.indexes)

    def deparse(self):
        return self.name + "(" + ",".join(self.indexes) + ")"


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

    def variables(self) -> dict[str, list[Variable]]:
        variables_mapping = self.left.variables().copy()
        for name, variables in self.right.variables().items():
            if name in variables_mapping:
                variables_mapping[name] = [*variables_mapping[name], *variables]
            else:
                variables_mapping[name] = variables
        return variables_mapping

    def deparse(self):
        return self.left.deparse() + " + " + self.right.deparse()

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True, slots=True)
class Subtract(Expression):
    left: Expression
    right: Expression

    def variables(self) -> dict[str, list[Variable]]:
        variables_mapping = self.left.variables().copy()
        for name, variables in self.right.variables().items():
            if name in variables_mapping:
                variables_mapping[name] = [*variables_mapping[name], *variables]
            else:
                variables_mapping[name] = variables
        return variables_mapping

    def deparse(self):
        return self.left.deparse() + " - " + self.right.deparse()

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True, slots=True)
class Multiply(Expression):
    left: Expression
    right: Expression

    def variables(self) -> dict[str, list[Variable]]:
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
        if isinstance(self.right, (Add, Subtract)):
            right_string = f"({right_string})"

        return f"{left_string} * {right_string}"

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True)
class Assignment:
    target: Variable
    expression: Expression

    def __post_init__(self):
        from .exceptions import InconsistentVariableSizeError, MutatingAssignmentError

        target_name = self.target.name

        variable_orders: dict[str, int | None] = {}
        variables_mapping = self.expression.variables()
        for name, (first, *rest) in variables_mapping.items():
            if name == target_name:
                raise MutatingAssignmentError(self)

            for variable in rest:
                if first.order != variable.order:
                    raise InconsistentVariableSizeError(self, first, variable)

            variable_orders[name] = first.order

        self._parameter_orders: dict[str, int | None]
        object.__setattr__(self, "_parameter_orders", variable_orders)

    def deparse(self) -> str:
        return self.target.deparse() + " = " + self.expression.deparse()

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.target, self.expression)

    def variable_orders(self) -> dict[str, int | None]:
        """Number of dimensions of each variable.

        This is the same as `parameter_orders` except that it also includes the target variable.

        Returns:
            A mapping where each key is the string name of a variable and each value is the number
            of dimensions that variable has in the right hand side.
        """

        return {
            self.target.name: self.target.order if isinstance(self.target, Tensor) else None,
            **self._parameter_orders,
        }

    def __str__(self) -> str:
        return self.deparse()
