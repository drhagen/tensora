__all__ = ["format_order"]

from ._format import Format


def format_order(format: Format | None) -> int | None:
    return len(format.modes) if format is not None else None
