import pytest


def generate_pairs():
    # e.g. test all combinations of small positives up to 5
    for a in range(1, 6):
        for b in range(1, 6):
            yield a, b


@pytest.mark.parametrize("a,b", generate_pairs())
def test_addition(a: int, b: int):
    assert a > 0
    assert b > 0
    assert a + b > a
    assert a + b > b
