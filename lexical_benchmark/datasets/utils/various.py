import pandas as pd
import spacy


def spacy_model(model_name: str) -> spacy.Language:
    """Safely load spacy Language Model."""
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli.download import download

        download(model_name)

        return spacy.load(model_name)


def word_to_pos(word: str, pos_model: spacy.Language) -> str | None:
    """Infer Part of Speech from a given word."""
    doc = pos_model(word)
    first_token = next(iter(doc), None)
    if first_token:
        return first_token.pos_
    return None


def segment_synonym(df: pd.DataFrame, header: str) -> pd.DataFrame:
    """Seperate lines for synonyms."""
    df = df.assign(Column_Split=df[header].str.split("/")).explode("Column_Split")
    return df.drop(header, axis=1).rename(columns={"Column_Split": header})


def remove_exp(df: pd.DataFrame, header: str) -> pd.DataFrame:
    """Remove expressions with more than one word."""
    return df[~df[header].str.contains(r"\s", regex=True)]


def merge_word(df: pd.DataFrame, header: str) -> pd.DataFrame:
    """Merge same word in different semantic senses."""
    merged_df = df.groupby(header).first().reset_index()
    # Aggregate other columns
    for col in df.columns:
        if col != header:
            if df[col].dtype == "object":
                merged_df[col] = df.groupby(header)[col].first().reset_index()[col]
            else:
                merged_df[col] = df.groupby(header)[col].sum().reset_index()[col]
    return merged_df


def to_roman(value: int) -> str:
    """Convert an int into a roman numeral."""
    roman_map = {
        1: "I",
        4: "IV",
        5: "V",
        9: "IX",
        10: "X",
        40: "XL",
        50: "L",
        90: "XC",
        100: "C",
        400: "CD",
        500: "D",
        900: "CM",
        1000: "M",
    }
    result = ""
    remainder = value

    if value > 3_999_999:
        raise ValueError(f"Roman numerals cannot exceed 3,999,999, given {value} !!!")

    for i in sorted(roman_map.keys(), reverse=True):
        if remainder > 0:
            multiplier = i
            roman_digit = roman_map[i]

            times = remainder // multiplier
            remainder = remainder % multiplier
            result += roman_digit * times

    return result
