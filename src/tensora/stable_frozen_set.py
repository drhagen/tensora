from __future__ import annotations

__all__ = ["StableFrozenSet"]

from typing import AbstractSet, Hashable, Iterator, TypeVar

T = TypeVar("T", bound=Hashable, covariant=True)


class StableFrozenSet(AbstractSet[T]):
    def __init__(self, *items: T):
        used = set()
        unique_items = []
        for item in items:
            if item not in used:
                unique_items.append(item)
                used.add(item)
        self._items = tuple(unique_items)
        self._set = frozenset(items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, x: T) -> bool:
        return x in self._set

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __reversed__(self):
        return StableFrozenSet(*reversed(self._items))

    def __or__(self, other: StableFrozenSet[T]) -> StableFrozenSet[T]:
        return StableFrozenSet(*self._items, *other._items)

    def __eq__(self, other):
        if isinstance(other, StableFrozenSet):
            return self._set == other._set
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self._set)
