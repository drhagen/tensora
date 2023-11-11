from __future__ import annotations

__all__ = ['SourceBuilder']

from abc import abstractmethod
from contextlib import contextmanager
from typing import List, Dict, Optional, Union

from .ast import Statement, FunctionDefinition, Variable, Declaration, Block, Expression, Branch, Loop
from .types import Type


class Builder:
    def __init__(self):
        self.lines = []

    @abstractmethod
    def finalize(self):
        raise NotImplementedError()


class BlockBuilder(Builder):
    def __init__(self, comment: Optional[str] = None):
        super().__init__()
        self.comment = comment

    def finalize(self):
        return Block(self.lines, self.comment)


class BranchBuilder(Builder):
    def __init__(self, condition: Expression):
        super().__init__()
        self.condition = condition

    def finalize(self):
        return Branch(self.condition, Block(self.lines), Block([]))


class LoopBuilder(Builder):
    def __init__(self, condition: Expression):
        super().__init__()
        self.condition = condition

    def finalize(self):
        return Loop(self.condition, Block(self.lines))


class FunctionDefinitionBuilder(Builder):
    def __init__(self, name: str, parameters: Dict[str, Type], return_type: Type):
        super().__init__()
        self.name = name
        self.parameters = parameters
        self.return_type = return_type

    def finalize(self):
        return FunctionDefinition(
            Variable(self.name),
            [Declaration(Variable(name), type) for name, type in self.parameters.items()],
            self.return_type,
            Block(self.lines),
        )


class SourceBuilder:
    def __init__(self, comment: Optional[str] = None):
        self.dependencies: Dict[str, ()] = {}
        self.stack: List[Builder] = [BlockBuilder(comment)]

    def append(self, statement: Union[Statement, SourceBuilder]):
        if isinstance(statement, SourceBuilder):
            for dependency in statement.dependencies:
                self.add_dependency(dependency)
            statement = statement.finalize()
            if statement.comment is not None:
                self.stack[-1].lines.append(statement)
            else:
                self.stack[-1].lines.extend(statement.statements)
        else:
            self.stack[-1].lines.append(statement)

    def add_dependency(self, name: str):
        self.dependencies[name] = ()

    @contextmanager
    def block(self, comment: Optional[str] = None):
        self.stack.append(BlockBuilder(comment))
        yield None
        self.append(self.stack.pop().finalize())

    @contextmanager
    def branch(self, condition: Expression):
        self.stack.append(BranchBuilder(condition))
        yield None
        self.append(self.stack.pop().finalize())

    @contextmanager
    def loop(self, condition: Expression):
        self.stack.append(LoopBuilder(condition))
        yield None
        self.append(self.stack.pop().finalize())

    @contextmanager
    def function_definition(self, name: str, parameters: Dict[str, Type], return_type: Type):
        self.stack.append(FunctionDefinitionBuilder(name, parameters, return_type))
        yield None
        self.append(self.stack.pop().finalize())

    def finalize(self) -> Block:
        assert len(self.stack) == 1
        return self.stack[0].finalize()
