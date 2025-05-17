"""Test math functions."""


def generate_pairs() -> tuple[int, int]:
    """Test all combinations of small positives up to 5."""
    for a in range(1, 6):
        for b in range(1, 6):
            yield a, b
