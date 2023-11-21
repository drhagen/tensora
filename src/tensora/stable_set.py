from __future__ import annotations

__all__ = ["StableSet", "StableFrozenSet"]

from typing import AbstractSet, Hashable, Iterator, MutableSet, TypeVar

Element = TypeVar("Element", bound=Hashable, covariant=True)


class StableSet(MutableSet[Element]):
    def __init__(self, *items: Element):
        # Rely on stable dictionary
        self._items = {item: None for item in items}

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, x: Element) -> bool:
        return x in self._items

    def __iter__(self) -> Iterator[Element]:
        return iter(self._items)

    def __reversed__(self):
        return StableSet(*reversed(self._items))

    def __or__(self, other: StableSet[Element]) -> StableSet[Element]:
        return StableSet(*self._items, *other._items)

    def __eq__(self, other: StableSet[Element]):
        if isinstance(other, StableSet):
            return self._items == other._items
        else:
            return NotImplemented

    def __repr__(self) -> str:
        return f"StableSet({', '.join(repr(item) for item in self._items)})"

    def add(self, element: Element, /) -> None:
        self._items[element] = None

    def discard(self, element: Element, /) -> None:
        del self[element]


class StableFrozenSet(AbstractSet[Element]):
    def __init__(self, *items: Element):
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

    def __contains__(self, x: Element) -> bool:
        return x in self._set

    def __iter__(self) -> Iterator[Element]:
        return iter(self._items)

    def __reversed__(self):
        return StableFrozenSet(*reversed(self._items))

    def __or__(self, other: StableFrozenSet[Element]) -> StableFrozenSet[Element]:
        return StableFrozenSet(*self._items, *other._items)

    def __eq__(self, other):
        if isinstance(other, StableFrozenSet):
            return self._set == other._set
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self._set)

    def __repr__(self) -> str:
        return f"StableFrozenSet({', '.join(repr(item) for item in self._items)})"
