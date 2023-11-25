__all__ = [
    "Node",
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


class Node:
    __slots__ = ()

    @abstractmethod
    def deparse(self) -> str:
        """Convert the assignment back into a string."""
        pass

    @abstractmethod
    def variable_orders(self) -> dict[str, int]:
        """Number of dimensions of each variable.

        Returns:
            A mapping where each key is the string name of a variable and each value is the number of dimensions that
            variable has in the expression.
        """
        pass

    @abstractmethod
    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        """Map of index name to tensors and dimensions it exists in.

        Returns:
            A mapping where each key is the string name of an index and each value is a sets of pairs. In each pair, the
            first element is the name of a tensor and the second element is the dimension in which that index appears.
        """
        pass

    def __str__(self):
        return self.deparse()


class Expression(Node):
    __slots__ = ()


class Literal(Expression):
    __slots__ = ()

    def variable_orders(self) -> dict[str, int]:
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

    @property
    def order(self):
        return len(self.indexes)

    def variable_orders(self) -> dict[str, int]:
        return {self.name: self.order}

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        participants = {}
        for i, index_name in enumerate(self.indexes):
            participants[index_name] = participants.get(index_name, set()) | {(self.name, i)}
        return participants


@dataclass(frozen=True, slots=True)
class Scalar(Variable):
    name: str

    @property
    def indexes(self):
        return []

    def deparse(self):
        return self.name


@dataclass(frozen=True, slots=True)
class Tensor(Variable):
    name: str
    indexes: list[str]

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

    def deparse(self):
        return self.left.deparse() + " + " + self.right.deparse()

    def variable_orders(self) -> dict[str, int]:
        return {**self.left.variable_orders(), **self.right.variable_orders()}

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True, slots=True)
class Subtract(Expression):
    left: Expression
    right: Expression

    def deparse(self):
        return self.left.deparse() + " - " + self.right.deparse()

    def variable_orders(self) -> dict[str, int]:
        return {**self.left.variable_orders(), **self.right.variable_orders()}

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True, slots=True)
class Multiply(Expression):
    left: Expression
    right: Expression

    def deparse(self):
        if isinstance(self.left, (Add, Subtract)):
            left_string = "(" + self.left.deparse() + ")"
        else:
            left_string = self.left.deparse()

        if isinstance(self.right, (Add, Subtract)):
            right_string = "(" + self.right.deparse() + ")"
        else:
            right_string = self.right.deparse()

        return left_string + " * " + right_string

    def variable_orders(self) -> dict[str, int]:
        return {**self.left.variable_orders(), **self.right.variable_orders()}

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.left, self.right)


@dataclass(frozen=True, slots=True)
class Assignment(Node):
    target: Variable
    expression: Expression

    def deparse(self) -> str:
        return self.target.deparse() + " = " + self.expression.deparse()

    def variable_orders(self) -> dict[str, int]:
        return {**self.target.variable_orders(), **self.expression.variable_orders()}

    def index_participants(self) -> dict[str, set[tuple[str, int]]]:
        return merge_index_participants(self.target, self.expression)
