import pytest

from lexical_benchmark.processing import stratify


@pytest.fixture
def stratifier_obj() -> stratify.TextBlockStratifier:
    """Initialise the stratifier object using default values and random text blocks."""
    st = stratify.TextBlockStratifier(chunk_number=20, seed=42)
    for _ in range(4):
        st.add_block(stratify._random_text_chunk(rndr=st.random_state))  # noqa: SLF001

    return st


def test_sampling(stratifier_obj) -> None:
    """Test sampling into a chunked list of blocks."""
    # Initialise coords
    stratifier_obj._init_block_coords()  # noqa: SLF001
    a = stratifier_obj._get_1d_sample_coords()  # noqa: SLF001

    for idx, (i, j) in enumerate(a):
        assert idx == i, f"In coords : ({i=}, {j=}) i should be gradual and equal to {idx}."
    assert len(set(a)) == len(a), "All items in A should be unique."

    b = stratifier_obj._get_1d_sample_coords(exclude=a)  # noqa: SLF001
    assert set(a) not in set(b), f"{a=} should not be in {b=}."
    for idx, (i, j) in enumerate(b):
        assert idx == i, f"In coords : ({i=}, {j=}) i should be gradual and equal to {idx}."
    assert len(set(b)) == len(b), "All items in B should be unique."

    c = stratifier_obj._get_1d_sample_coords(exclude=[*a, *b])  # noqa: SLF001
    assert set(a) not in set(c), f"{a=} should not be in {c=}."
    assert set(b) not in set(c), f"{b=} should not be in {c=}."
    for idx, (i, j) in enumerate(c):
        assert idx == i, f"In coords : ({i=}, {j=}) i should be gradual and equal to {idx}."
    assert len(set(c)) == len(c), "All items in C should be unique."
