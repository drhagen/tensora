from tensora import Tensor, evaluate


def test_sum_non_adjacent():
    b = Tensor.from_lol([1, 2, 3])
    c = Tensor.from_lol([[1, 3, 5], [2, 4, 6]])
    d = Tensor.from_lol([7, 8, 9])
    actual = evaluate("a(i) = b(i) + c(j,i) + d(i)", "d", b=b, c=c, d=d)
    expected = Tensor.from_lol([11, 17, 23])
    assert actual == expected
