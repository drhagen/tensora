from dataclasses import dataclass


@dataclass(frozen=True)
class Id:
    name: str
    instance: int
