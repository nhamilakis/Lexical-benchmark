import math

from lexical_benchmark.processing import stratify


def test_sampling() -> None:
    """Test sampling into a chunked list of blocks."""
    # Initialize seed & chunk number
    stratifier_obj = stratify.TextBlockStratifier(chunk_number=60, seed=42)
    for _ in range(6):
        stratifier_obj.add_block(stratify._random_text_chunk(rndr=stratifier_obj.random_state))  # noqa: SLF001

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


def _check_in(size_targets: tuple[int, ...], coord_blocks, x: int, y: int) -> None:
    """A helper function to help with checking 0 index expansion."""
    for size in size_targets:
        for item in coord_blocks[x][y]:
            assert item in coord_blocks[size][y], f"Items from [1][0] should be present in [{size}][{y}]"


def test_stratify_mapping() -> None:
    """Test building of stratify chunk mappings (without dev)."""
    # Initialize seed & chunk number
    stratifier_obj = stratify.TextBlockStratifier(chunk_number=60, seed=42)
    for _ in range(6):
        stratifier_obj.add_block(stratify._random_text_chunk(rndr=stratifier_obj.random_state))  # noqa: SLF001
    stratif_map = stratifier_obj.build_stratifier(size_targets=(1, 2, 3, 4, 5, 6))

    # Check 0 index expansion
    _check_in((1, 2, 3, 4, 5, 6), stratif_map.coord_blocks, 1, 0)
    _check_in((2, 3, 4, 5, 6), stratif_map.coord_blocks, 2, 0)
    _check_in((3, 4, 5, 6), stratif_map.coord_blocks, 3, 0)
    _check_in((4, 5, 6), stratif_map.coord_blocks, 4, 0)
    _check_in((5, 6), stratif_map.coord_blocks, 5, 0)


def test_stratify_mapping_with_dev_set() -> None:
    """Test building of stratify chunk mappings (with dev set)."""
    # Initialize seed & chunk number
    stratifier_obj = stratify.TextBlockStratifier(chunk_number=60, seed=42, dev_percent=0.09)
    for _ in range(6):
        stratifier_obj.add_block(stratify._random_text_chunk(rndr=stratifier_obj.random_state))  # noqa: SLF001

    stratif_map = stratifier_obj.build_stratifier(size_targets=(1, 2, 3, 4, 5, 6))

    # Check 0 index expansion
    _check_in((1, 2, 3, 4, 5, 6), stratif_map.coord_blocks, 1, 0)
    _check_in((2, 3, 4, 5, 6), stratif_map.coord_blocks, 2, 0)
    _check_in((3, 4, 5, 6), stratif_map.coord_blocks, 3, 0)
    _check_in((4, 5, 6), stratif_map.coord_blocks, 4, 0)
    _check_in((5, 6), stratif_map.coord_blocks, 5, 0)

    dev_set = set(stratif_map.dev_coords)
    assert all(
        item not in stratif_map.coord_blocks[x][y]
        for x in stratif_map.coord_blocks
        for y in stratif_map.coord_blocks[x]
        for item in dev_set
    ), "Item from dev set should not be in any train item (dev[x] found in coords[x][y])"


def test_dataset_splitting() -> None:
    """Test spliting functionality of the stratifying class."""
    # Initialize seed & chunk number
    stratifier_obj = stratify.TextBlockStratifier(chunk_number=60, seed=42, dev_percent=0.09)
    for _ in range(6):
        stratifier_obj.add_block(stratify._random_text_chunk(rndr=stratifier_obj.random_state))  # noqa: SLF001

    block_stack = stratifier_obj.get_splits_stack()
    assert len(block_stack.blocks) == 6, "There should be 6 categories"
    expected_chunks = math.ceil(60 + (60 * 0.09))

    for i in block_stack.blocks:
        assert len(block_stack.blocks[i].chunks) == expected_chunks, (
            f"Chunks should be split in exactly {expected_chunks} parts !!"
        )
