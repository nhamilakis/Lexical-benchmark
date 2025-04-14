from dateutil.relativedelta import relativedelta

from .extract import extract, extract_with_tags

__all__ = ["extract", "extract_with_tags"]


def child_age_parsor(age_string: str | None) -> relativedelta | None:
    """Parse age of child."""
    if not age_string:
        return None

    # initialise variables
    years, months, days = (None, None, None)

    # Parse years
    if ";" in age_string:
        years, _, rest = age_string.rpartition(";")
    else:
        rest = age_string

    # Parse months
    if "." in rest:
        months, _, days = rest.rpartition(".")
    else:
        months = rest

    try:
        years = int(years) if years else 0
        months = int(months) if months else 0
        days = int(days) if days else 0
    except ValueError as err:
        raise ValueError(f"Failed to parse age [{age_string}]<{type(age_string)}> !") from err

    return relativedelta(years=years, months=months, days=days)


def normalised_child_age(age: str | relativedelta | None) -> float | None:
    """Normalises the age of a child into a number of months."""
    match age:
        case str():
            as_relative: relativedelta = child_age_parsor(age)
        case relativedelta():
            as_relative: relativedelta = age
        case None:
            return None
        case _:
            raise ValueError(f"Type {type(age)} not an expected input type.")

    months_from_years = as_relative.years * 12

    # Get months directly from relativedelta
    months_direct = as_relative.months

    # Convert days to fractional months (approximate)
    months_from_days = as_relative.days / 30.44

    # Sum all components
    return months_from_years + months_direct + months_from_days
