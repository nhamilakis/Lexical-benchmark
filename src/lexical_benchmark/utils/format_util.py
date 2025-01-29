def str_to_bool(value):
    """Convert string representation of boolean to actual boolean value."""
    if isinstance(value, bool):
        return value

    value = str(value).lower().strip()

    true_values = {"true", "1", "yes", "y", "on"}
    false_values = {"false", "0", "no", "n", "off"}

    if value in true_values:
        return True
    elif value in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert '{value}' to boolean")

def format_text_with_boundaries(text: str, add_space: True) -> str:
    """
    Format text by:
    1. Removing punctuation
    2. Adding spaces between characters
    3. Adding boundary markers between words

    Args:
        text: Input text string
    Returns:
        Formatted text with character spacing and word boundaries
    """
    # Remove punctuation and split into words
    words = text.split()
    if add_space:
        # Format each word by adding spaces between characters
        spaced_words = [" ".join(word.lower()) for word in words]
        # Join words with boundary marker
        return " | ".join(spaced_words) + " |"
    else:
        spaced_words = ["".join(word.lower()) for word in words]
        # Join words with boundary marker
        return "|".join(spaced_words) + "|"
