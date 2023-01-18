import pytest

from tensora.desugar.desugar_expression import desugar_assignment
from tensora.expression import ast as sugar
from tensora.desugar import ast as desugar
from tensora.desugar.id import Id


@pytest.mark.parametrize(
    ["expression", "expected"],
    [
        [
            sugar.Assignment(sugar.Tensor("a", ["i"]), sugar.Add(sugar.Tensor("b", ["i"]), sugar.Tensor("c", ["i"]))),
            desugar.Assignment(
                desugar.Tensor(Id("a", 0), ["i"]),
                desugar.Add(desugar.Tensor(Id("b", 1), ["i"]), desugar.Tensor(Id("c", 2), ["i"])),
            ),
        ],
        [
            sugar.Assignment(
                sugar.Tensor("A", ["i", "j"]),
                sugar.Multiply(sugar.Tensor("B", ["i", "k"]), sugar.Tensor("C", ["k", "i"])),
            ),
            desugar.Assignment(
                desugar.Tensor(Id("A", 0), ["i", "j"]),
                desugar.Contract(
                    "k",
                    desugar.Multiply(desugar.Tensor(Id("B", 1), ["i", "k"]), desugar.Tensor(Id("C", 2), ["k", "i"])),
                ),
            ),
        ],
        [
            sugar.Assignment(
                sugar.Tensor("A", ["i", "j"]),
                sugar.Add(
                    sugar.Tensor("K", ["i", "j"]),
                    sugar.Multiply(sugar.Tensor("B", ["i", "k"]), sugar.Tensor("C", ["k", "i"])),
                ),
            ),
            desugar.Assignment(
                desugar.Tensor(Id("A", 0), ["i", "j"]),
                desugar.Add(
                    desugar.Tensor(Id("K", 1), ["i", "j"]),
                    desugar.Contract(
                        "k",
                        desugar.Multiply(
                            desugar.Tensor(Id("B", 2), ["i", "k"]), desugar.Tensor(Id("C", 3), ["k", "i"])
                        ),
                    ),
                ),
            ),
        ],
        [
            sugar.Assignment(sugar.Tensor("a", ["i"]), sugar.Tensor("b", ["i", "j"])),
            desugar.Assignment(
                desugar.Tensor(Id("a", 0), ["i"]),
                desugar.Contract("j", desugar.Tensor(Id("b", 1), ["i", "j"])),
            ),
        ],
    ],
)
def test_desugar(expression, expected):
    actual = desugar_assignment(expression)
    assert actual == expected
