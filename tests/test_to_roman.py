import pytest

from lexical_benchmark.text_lib.text_cleaners import to_roman


@pytest.mark.anyio
def test_roman_numerals() -> None:
    """Tests roman numeral conversion for various cases.

    Tests basic numerals, subtractive notation and edge cases.

    Raises:
        AssertionError: If conversion is incorrect

    """
    # Test cases as tuples of (expected_result, roman_numeral)
    test_cases: list[tuple[int, str]] = [
        (1, "I"),
        (4, "IV"),
        (5, "V"),
        (9, "IX"),
        (10, "X"),
        (40, "XL"),
        (50, "L"),
        (90, "XC"),
        (100, "C"),
        (400, "CD"),
        (500, "D"),
        (900, "CM"),
        (1000, "M"),
        # Complex numbers
        (2023, "MMXXIII"),
        (3999, "MMMCMXCIX"),
    ]

    for number, expected_roman in test_cases:
        assert to_roman(number) == expected_roman, f"Failed to convert {number} => {expected_roman}"
