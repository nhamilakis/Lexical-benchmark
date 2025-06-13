from pathlib import Path


def trim_sentence_list(text: list[str], max_token_count: int) -> list[str]:
    """Trim text to keep sentences within a target word count.

    Removes sentences from the end of the text list until the word count
    is less than or equal to the target count. Never splits sentences.
    """
    if max_token_count < 0:
        raise ValueError("max_token_count cannot be negative !!")
    if not text:
        return []

    total_words = 0
    trimmed_text = []

    for sentence in text:
        word_count = len(sentence.split())
        new_total = total_words + word_count

        if new_total <= max_token_count:
            trimmed_text.append(sentence)
            total_words = new_total
        else:
            break

    return trimmed_text


def word_count(lines: list[str], *, skip_stupid: bool = True) -> int:
    """Count number of words."""
    total = 0
    for ln in lines:
        if len(ln) <= 3 and skip_stupid:
            continue
        total += len(ln.split())
    return total


def type_count(lines: list[str], *, skip_stupid: bool = True) -> int:
    """Count token-types in a set."""
    words = []
    for ln in lines:
        if len(ln) <= 3 and skip_stupid:
            continue
        words.extend(ln.split())
    return len(set(words))


def tokenizer(line: str) -> list[str]:
    """Tokenizing function."""
    return line.split()


def line_tokenizer(lines: list[str]) -> list[str]:
    """Line tokenizing function."""
    words = []
    for ln in lines:
        words.extend(ln.split())
    return words


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


def split_lines_by_tokens(
    lines: list[str], train_ratio: float = 0.8, *, shuffle: bool = False, random_seed: int | None = None
) -> tuple[str, str]:
    """Split given lines into dev, train subsets.

    Splits lines to approximate the desired token ratio without splitting
    individual lines. Always ensures train ratio doesn't exceed the target.

    Raises:
        ValueError: If train_ratio is not between 0 and 1
        ValueError: If lines list is empty

    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not lines:
        raise ValueError("lines list cannot be empty")

    lines_copy = lines.copy()
    if shuffle:
        import random

        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(lines_copy)
    total_tokens = word_count(lines_copy)
    target_train_tokens = int(total_tokens * train_ratio)

    # Greedily select lines for training set
    train_lines: list[str] = []
    train_tokens = 0
    for line in lines_copy:
        line_tokens = word_count([line])
        if train_tokens + line_tokens <= target_train_tokens:
            train_lines.append(line)
            train_tokens += line_tokens
        else:
            break

    # Remaining lines go to dev set
    dev_lines = lines_copy[len(train_lines) :]
    return dev_lines, train_lines


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
        return "misc"
