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

