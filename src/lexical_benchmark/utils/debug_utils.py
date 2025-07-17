import atexit

from rich.pretty import pprint

from lexical_benchmark import settings

_FIRST: bool = False


def debug_print(*args, header: str | None = None, **kwargs) -> None:
    """Print debug information."""
    global _FIRST  # noqa: PLW0603

    if settings.DEBUG:
        if _FIRST:
            pprint("=================== DEBUG INFO ===================")
            atexit.register(lambda: pprint("=================== END DEBUG RUN ==================="))
            _FIRST = False

        if header:
            pprint(f"{'=' * 5} {header.capitalize()} {'=' * 5}")
        else:
            pprint(f"{'=' * 15}")
        pprint(*args, **kwargs)
        pprint("=" * 15)
