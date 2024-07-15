import sys
import warnings

try:
    import pymorphy2  # type:ignore[import-untyped]
    from polyglot.downloader import downloader  # type:ignore[import-untyped]
    from polyglot.text import Word  # type:ignore[import-untyped]
except ImportError:
    warnings.warn(
        "Missing 'polyglot' optional dependencies, reinstall the module using 'lm_benchmark[polyglot]'",
        stacklevel=2,
    )
    sys.exit(1)


def main() -> None:
    """Does something on morphology."""
    print(downloader.supported_languages_table("morph2"))

    words = ["preprocessing", "processor", "invaluable", "thankful", "crossed"]
    for w in words:
        word = Word(w, language="en")
        print(f"{word:<20}{word.morphemes}")

    morph = pymorphy2.MorphAnalyzer()
    word = "unhappinessly"
    parsed_word = morph.parse(word)[0]
    print(parsed_word.normal_form)  # base form
    print(parsed_word.tag)
