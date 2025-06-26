import collections
from pathlib import Path

try:
    import polars as pl  # type: ignore[import-not-found, import-untyped]
except ImportError:
    pl = None  # type: ignore[assignment]

from lexical_benchmark import settings
from lexical_benchmark.datasets import childes, stella, wordstats
from lexical_benchmark.datasets import utils as dataset_utils


def _prepare_childes_adult() -> None:
    """Prepare word frequencies from CHILDES/EN/Adult."""
    if pl is None:
        raise OSError(f"This function ({_prepare_childes_adult}) requires polars to be installed !")

    dataset = childes.CHILDESDataset()
    target = wordstats.WordStatsDataset(lang="EN")
    na_freq = dataset.wf.processed("Eng-NA", "adult").read_csv(has_header=True, use_pandas=False)
    na_freq = na_freq.rename({"freq": "count"})

    uk_freq = dataset.wf.processed("Eng-UK", "adult").read_csv(has_header=True, use_pandas=False)
    uk_freq = uk_freq.rename({"freq": "count"})

    combined_df = pl.concat([uk_freq, na_freq])
    combined_df = combined_df.group_by("word").agg(pl.col("count").sum()).sort("word")

    # Write to disk
    target.word_frequencies.childes_adult.mk_parent()
    combined_df.write_csv(target.word_frequencies.childes_adult)


def _prepare_stela() -> None:
    """Prepare word frequencies from STELA/EN/by_month/60/00."""
    if pl is None:
        raise OSError(f"This function({_prepare_stela}) requires polars to be installed !")

    stella_dataset = stella.STELATranscriptDataset(root_dir=settings.PATH.stela2)
    target = wordstats.WordStatsDataset(lang="EN")
    transcriptions = stella_dataset.by_month / "EN/60/00/transcription.txt"
    word_counts = collections.Counter(transcriptions.read_tokenized())
    wf = pl.DataFrame({"word": list(word_counts.keys()), "count": list(word_counts.values())})

    # Write to disk
    target.word_frequencies.stela_by_month_60_00.mk_parent()
    wf.write_csv(target.word_frequencies.stela_by_month_60_00, include_header=True)


def _prepare_childrealistic() -> None:
    """Prepare word frequencies from Childlike/EN/by_month/60/00."""
    if pl is None:
        raise OSError(f"This function({_prepare_childrealistic}) requires polars to be installed !")

    target = wordstats.WordStatsDataset(lang="EN")
    txt = settings.PATH.child_realistic / "by_month/EN/60/00" / "transcription.txt"
    if not txt.is_file():
        raise ValueError("Dataset Childlike is missing by_month information !!")
    word_counts = collections.Counter(txt.read_tokenized())
    wf = pl.DataFrame({"word": list(word_counts.keys()), "count": list(word_counts.values())})

    # Write to disk
    target.word_frequencies.child_realistic_by_month_60_00.mk_parent()
    wf.write_csv(target.word_frequencies.child_realistic_by_month_60_00, include_header=True)


def _prepare_cdi_childes() -> None:
    """Prepare word frequencies of CDI words with CHILDES/Adult frequencies."""
    if pl is None:
        raise OSError(f"This function ({_prepare_cdi_childes}) requires polars to be installed !")

    df = (settings.PATH.wordbank_cdi / "en-na/ws_cdi_produce.csv").read_csv(has_header=True, use_pandas=False)
    df = df.with_columns(pl.col("word").str.to_lowercase())
    target = wordstats.WordStatsDataset(lang="EN")

    if not target.word_frequencies.childes_adult.is_file():
        raise ValueError("Requires CHILDES/EN/Adult data to be computed !!")

    childes_wf = target.word_frequencies.childes_adult.read_csv(has_header=True)
    cdi_childes_wf = df.join(childes_wf.select(["word", "count"]), on="word", how="left").with_columns(
        pl.col("count").fill_null(0)
    )

    # Write to disk
    target.word_frequencies.cdi_childes.mk_parent()
    cdi_childes_wf.write_csv(target.word_frequencies.cdi_childes, include_header=True)


def _prepare_cdi_childrealistic() -> None:
    """Prepare word frequencies of CDI words with CHILDRealistic frequencies."""
    if pl is None:
        raise OSError(f"This function ({_prepare_cdi_childrealistic}) requires polars to be installed !")

    df = (settings.PATH.wordbank_cdi / "en-na/ws_cdi_produce.csv").read_csv(has_header=True, use_pandas=False)
    df = df.with_columns(pl.col("word").str.to_lowercase())

    target = wordstats.WordStatsDataset(lang="EN")
    if not target.word_frequencies.child_realistic_by_month_60_00.is_file():
        raise ValueError("Requires CHILDRealistic/by_month/EN data to be computed !!")

    childrealistc_wf = target.word_frequencies.child_realistic_by_month_60_00.read_csv(has_header=True)
    cdi_childes_wf = df.join(childrealistc_wf.select(["word", "count"]), on="word", how="left").with_columns(
        pl.col("count").fill_null(0)
    )

    # write to disk
    target.word_frequencies.cdi_childrealistic.mk_parent()
    cdi_childes_wf.write_csv(target.word_frequencies.cdi_childrealistic, include_header=True)


def build_word_pos_maps(pos_model: str, *, require_gpu: bool, batch_size: int) -> dict[str, str]:
    """Build a Mapping of all words in the dataset."""
    dataset = wordstats.WordStatsDataset(lang="EN")
    stela_wf = dataset.word_frequencies.stela_by_month_60_00.read_csv(use_pandas=False)
    childes_wf = dataset.word_frequencies.childes_adult.read_csv(use_pandas=False)
    child_realistic_wf = dataset.word_frequencies.child_realistic_by_month_60_00.read_csv(use_pandas=False)
    cdi_child_realistic_wf = dataset.word_frequencies.cdi_childrealistic.read_csv(use_pandas=False)
    cdi_childes_wf = dataset.word_frequencies.cdi_childes.read_csv(use_pandas=False)

    # List of unique words
    words = list(
        {
            *list(stela_wf["word"]),
            *list(childes_wf["word"]),
            *list(child_realistic_wf["word"]),
            *list(cdi_child_realistic_wf["word"]),
            *list(cdi_childes_wf["word"]),
        }
    )
    pos_model = dataset_utils.various.spacy_model(pos_model, require_gpu=require_gpu)
    pos_lst = dataset_utils.batch_word_to_pos(words, pos_model, batch_size=batch_size)
    return {f"{w}": p for w, p in zip(words, pos_lst, strict=True)}


def attach_pos(csv_file: Path, word_pos_mapping: dict[str, str]) -> None:
    """Attach to a word-count csv file the POS tags."""
    wf = csv_file.read_csv(use_pandas=False)
    pos_wf = wf.with_columns(pl.col("word").map_elements(lambda x: word_pos_mapping.get(x)).alias("POS"))
    pos_wf.write_csv(csv_file, include_header=True)


def prepare_word_stats_word_counts() -> None:
    """Prepare all the word-count files for the wordstats dataset."""
    # Gather stela stats
    _prepare_stela()
    # Gather CHILDES stats
    _prepare_childes_adult()
    # Gather ChildRealistic
    _prepare_childrealistic()

    # Gather CDI with F from CHILDES/ChildRealistic
    _prepare_cdi_childes()
    _prepare_cdi_childrealistic()
