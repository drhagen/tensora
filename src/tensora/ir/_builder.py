from __future__ import annotations

__all__ = ["SourceBuilder"]

from abc import abstractmethod
from contextlib import contextmanager
from typing import Mapping

from .._stable_set import StableSet
from .ast import (
    Block,
    Branch,
    Declaration,
    Expression,
    FunctionDefinition,
    Loop,
    Statement,
    Variable,
)
from .types import Type


class Builder:
    def __init__(self):
        self.lines: list[Statement] = []

    @abstractmethod
    def finalize(self):
        raise NotImplementedError()


class BlockBuilder(Builder):
    def __init__(self, comment: str | None = None):
        super().__init__()
        self._comment = comment

    def finalize(self):
        return Block(self.lines, self._comment)


class BranchBuilder(Builder):
    def __init__(self, condition: Expression):
        super().__init__()
        self._condition = condition

    def finalize(self):
        return Branch(self._condition, Block(self.lines), Block([]))


class LoopBuilder(Builder):
    def __init__(self, condition: Expression):
        super().__init__()
        self._condition = condition

    def finalize(self):
        return Loop(self._condition, Block(self.lines))


class FunctionDefinitionBuilder(Builder):
    def __init__(self, name: str, parameters: Mapping[str, Type], return_type: Type):
        super().__init__()
        self._name = name
        self._parameters = parameters
        self._return_type = return_type

    def finalize(self):
        return FunctionDefinition(
            Variable(self._name),
            [Declaration(Variable(name), type) for name, type in self._parameters.items()],
            self._return_type,
            Block(self.lines),
        )


class SourceBuilder:
    def __init__(self, comment: str | None = None):
        self._dependencies: StableSet[str] = StableSet()
        self._stack: list[Builder] = [BlockBuilder(comment)]

    def append(self, statement: Statement | SourceBuilder):
        match statement:
            case Statement():
                self._stack[-1].lines.append(statement)
            case SourceBuilder():
                for dependency in statement._dependencies:
                    self.add_dependency(dependency)
                statement = statement.finalize()
                if statement.comment is not None:
                    self._stack[-1].lines.append(statement)
                else:
                    # This is not simplified by peephole, which only simplifies empty blocks;
                    # it does not inline not blocks with no comments. This is because blocks
                    # with no comments still get newlines between them. This means that
                    # appending a SourceBuilder appends the lines, but appending a block
                    # appends the block itself.
                    self._stack[-1].lines.extend(statement.statements)

    def add_dependency(self, name: str):
        self._dependencies.add(name)

    @contextmanager
    def block(self, comment: str | None = None):
        self._stack.append(BlockBuilder(comment))
        yield None
        self.append(self._stack.pop().finalize())

    @contextmanager
    def branch(self, condition: Expression):
        self._stack.append(BranchBuilder(condition))
        yield None
        self.append(self._stack.pop().finalize())

    @contextmanager
    def loop(self, condition: Expression):
        self._stack.append(LoopBuilder(condition))
        yield None
        self.append(self._stack.pop().finalize())

    @contextmanager
    def function_definition(self, name: str, parameters: Mapping[str, Type], return_type: Type):
        self._stack.append(FunctionDefinitionBuilder(name, parameters, return_type))
        yield None
        self.append(self._stack.pop().finalize())

    def finalize(self) -> Block:
        assert len(self._stack) == 1
        return self._stack[0].finalize()
