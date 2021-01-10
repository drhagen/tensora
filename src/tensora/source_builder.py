from __future__ import annotations

__all__ = ['SourceBuilder']

from typing import List


class SourceIndented:
    def __init__(self, source: SourceBuilder, n: int):
        self.source = source
        self.n = n

    def __enter__(self):
        self.source.indent += self.n

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.source.indent -= self.n


class SourceBuilder:
    def __init__(self):
        self.lines: List[str] = []
        self.indent = 0

    def append(self, line: str):
        self.lines.append(' ' * self.indent + line)

    def extend(self, lines: List[str]):
        for line in lines:
            self.append(line)

    def include(self, source: SourceBuilder):
        for line in source.lines:
            self.append(line)

    def indented(self, n: int = 2):
        return SourceIndented(self, n)

    def source(self):
        return '\n'.join(self.lines)
