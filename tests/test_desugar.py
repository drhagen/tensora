import pytest

from tensora.desugar import ast as desugar
from tensora.desugar import desugar_assignment
from tensora.expression import ast as sugar


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            sugar.Assignment(
                sugar.Tensor("a", ("i",)),
                sugar.Add(sugar.Tensor("b", ("i",)), sugar.Tensor("c", ("i",))),
            ),
            desugar.Assignment(
                desugar.Tensor(0, "a", ("i",)),
                desugar.Add(desugar.Tensor(1, "b", ("i",)), desugar.Tensor(2, "c", ("i",))),
            ),
        ),
        (
            sugar.Assignment(
                sugar.Tensor("A", ("i", "j")),
                sugar.Multiply(sugar.Tensor("B", ("i", "k")), sugar.Tensor("C", ("k", "i"))),
            ),
            desugar.Assignment(
                desugar.Tensor(0, "A", ("i", "j")),
                desugar.Contract(
                    "k",
                    desugar.Multiply(
                        desugar.Tensor(1, "B", ("i", "k")),
                        desugar.Tensor(2, "C", ("k", "i")),
                    ),
                ),
            ),
        ),
        (
            sugar.Assignment(
                sugar.Tensor("A", ("i", "j")),
                sugar.Add(
                    sugar.Tensor("K", ("i", "j")),
                    sugar.Multiply(sugar.Tensor("B", ("i", "k")), sugar.Tensor("C", ("k", "i"))),
                ),
            ),
            desugar.Assignment(
                desugar.Tensor(0, "A", ("i", "j")),
                desugar.Add(
                    desugar.Tensor(1, "K", ("i", "j")),
                    desugar.Contract(
                        "k",
                        desugar.Multiply(
                            desugar.Tensor(2, "B", ("i", "k")),
                            desugar.Tensor(3, "C", ("k", "i")),
                        ),
                    ),
                ),
            ),
        ),
        (
            sugar.Assignment(sugar.Tensor("a", ("i",)), sugar.Tensor("b", ("i", "j"))),
            desugar.Assignment(
                desugar.Tensor(0, "a", ("i",)),
                desugar.Contract("j", desugar.Tensor(1, "b", ("i", "j"))),
            ),
        ),
    ],
)
def test_desugar(expression, expected):
    actual = desugar_assignment(expression)
    assert actual == expected
