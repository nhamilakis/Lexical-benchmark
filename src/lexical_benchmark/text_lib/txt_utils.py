from pathlib import Path


def sentence_formatting(
    src: Path,
    *,
    target: Path | None = None,
    punctuation: str = ".!?",
    remove_blank: bool = True,
    keep_punctuation: bool = True,
) -> None:
    """Function allowing to re-organise a file to contain one sentence per line."""
    raw_lines = src.safe_readlines()

    if remove_blank:
        # Filter blank lines with less than 1 character
        raw_lines = [line.strip() for line in raw_lines if len(line.strip()) > 1]

    # Join lines with spaces instead of newlines to form a single line
    raw_text = " ".join(raw_lines)
    raw_text = raw_text.replace("\n", " ")

    # Replace all punctuations with new-line
    for c in punctuation:
        replacement = f"{c}\n" if keep_punctuation else "\n"
        raw_text = raw_text.replace(c, replacement)

    # Write into target or into source if no target given
    if target:
        target.safe_write_text(raw_text)
    else:
        src.safe_write_text(raw_text)


def _group_is_in_(*words: str, target: str) -> bool:
    """Check if any of the given words is in the target string."""
    return any(w in target for w in words)


class KeywordToGenre:
    """Convert a keyword to a book genre (based on word tag analysis)."""

    science_words: tuple[str, ...] = (
        "science",
        "space",
        "university",
    )
    religion_words = (
        "religion",
        "bible",
        "christianity",
        "testament",
        "christian",
        "church",
        "churches",
        "catholic",
        "apocalyptic",
        "theology",
    )
    mystery_words: tuple[str, ...] = ("mystery", "detective")
    fiction_words: tuple[str, ...] = (
        "fiction",
        "fictitious",
        "fairy",
        "magic",
    )
    political_philosophy: tuple[str, ...] = (
        "social",
        "socialism",
        "anarchism",
        "practical",
        "war",
        "philosophy",
        "political",
    )
    juvenile_words: tuple[str, ...] = (
        "juvenile",
        "children",
        "tales",
        "character",
        "humor",
        "humorous",
    )
    geography_words: tuple[str, ...] = (
        "french",
        "african",
        "western",
        "italian",
        "mexico",
        "russian",
        "german",
        "scandinavian",
        "iceland",
        "planet",
        "historyfrance",
        "canadian",
        "canada",
        "america",
        "britain",
        "west",
        "north",
        "eastern",
        "hemisphere",
        "york",
        "ireland",
        "europe",
        "british",
        "germany",
        "revolution",
        "slavik",
        "georgia",
        "oceania",
        "asia",
        "region",
        "period",
        "wessex",
        "geography",
        "travel",
        "military",
    )

    def group1(self, tags: str) -> str | None:
        """Check if tag is in group 1."""
        if _group_is_in_(*self.science_words, target=tags):
            return "science"
        if _group_is_in_(*self.religion_words, target=tags):
            return "religion"
        if _group_is_in_(*self.mystery_words, target=tags):
            return "mystery"
        if _group_is_in_(*self.fiction_words, target=tags):
            return "fiction"
        if _group_is_in_(*self.political_philosophy, target=tags):
            return "political & philosophy"

        return None

    def group2(self, tags: str) -> str | None:
        """Check if tag is in group 2."""
        if _group_is_in_(*self.juvenile_words, target=tags):
            return "Juvenile books"
        if _group_is_in_(*self.geography_words, target=tags):
            return "geography, nature & history"
        if _group_is_in_("essays", "short", target=tags):
            return "essays"

        return None

    def group3(self, tags: str) -> tuple[bool, str]:
        """Check if tag is in group 3."""
        if _group_is_in_("poetry", target=tags):
            return "poetry"
        if _group_is_in_("biography", target=tags):
            return "biography"
        if _group_is_in_("fantasy", target=tags):
            return "fantasy"
        if _group_is_in_("psycholog", target=tags):
            return "psychology"
        if _group_is_in_("zoology", target=tags):
            return "zoology"

        return None

    def __call__(self, tags: str) -> str:
        """Convert a list of keyword into a genre."""
        genre = self.group1(tags)
        if genre:
            return genre
        genre = self.group2(tags)
        if genre:
            return genre
        genre = self.group3(tags)
        if genre:
            return genre
        return "unknown"
