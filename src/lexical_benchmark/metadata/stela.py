import dataclasses
import logging
import pprint
import string
import typing as t
from pathlib import Path
from urllib.parse import urlparse

import polars as pl

from lexical_benchmark import datasets, web_scrappers
from lexical_benchmark.dataloaders import by_size, hour_txt
from lexical_benchmark.processing import word_stats
from lexical_benchmark.text_lib import lexicon, txt_utils

from .core import MetaBuilder, MetadataDir

L = logging.getLogger(__name__)


class _LineLenghtStruct(t.TypedDict):
    """Struct for gathering of line-length stats."""

    lang: str
    split: str
    chunk: str
    book: str
    line: int
    length: int


class _BySizeStats(t.TypedDict):
    """Struct gathering by_size split statistics."""

    lang: str
    size: str
    chunk: str
    token_count: int
    type_count: int
    token_rejection_rate: float
    type_rejection_rate: float
    type_token_ratio: float
    normalisation: str = "2kTokens"


class _BookGenrePathStruct(t.TypedDict):
    """Struct containing book paths & genres."""

    path: Path
    genre: str


@dataclasses.dataclass
class STELAMetaBuilder(MetaBuilder):
    """STELA Metadata Extractor."""

    dataset_cfg: datasets.STELADatasetConfig
    meta_dir: "STELAMetaDir"

    def build_all(self) -> None:
        """Build all stats."""
        _ = self.book_stats(save=True, force=True)
        _ = self.book_stat_resume(save=True, force=True)
        _ = self.extract_url_sources(save=True, force=True)
        _ = self.scrap_extra_metadata(save=True, force=True)
        _ = self.gather_line_length_stats(save=True, force=True)

    def _add_better_genres(self, book_stats_df: pl.DataFrame) -> pl.DataFrame:
        """Fix genre repartition using data extracted online.

        We try and split the big Literature & Undefined genre block by adding some
        genres from online sources.


        Returns:
          The same Dataframe with the genre column split into more categories

        """
        book_stats_df = book_stats_df.rename({"genre": "genre1"})

        if not self.meta_dir.external_book_metadata.is_file():
            raise FileNotFoundError(f"Expected {self.meta_dir.external_book_metadata} to exist.")
        scrapped_data = self.meta_dir.external_book_metadata.read_json()
        info2genre = txt_utils.KeywordToGenre()

        def extract_genre2(book_id: str) -> str | None:
            """Helper function to extract genres."""
            item = scrapped_data.get(book_id, None)
            if item is None:
                return None
            keyphrases = []
            loc_class = item.get("loc_class")
            if loc_class:
                keyphrases.append(loc_class)

            subjects = item.get("subjects", [])
            if subjects:
                keyphrases.extend(subjects)

            keyphrase = " ".join(keyphrases)
            keyphrase = "".join([c.lower() for c in keyphrase if c.lower() in string.ascii_letters])
            return info2genre(keyphrase)

        def choose_genre(genre1: str, genre2: str) -> str:
            if genre1 in ("Literature", "Undefined"):
                if genre2 in ("science", "essays"):
                    return "Science, Craft & Essay".lower()
                return genre2.lower()

            if genre2 in ("science", "essays"):
                return "Science, Craft & Essay".lower()
            return genre1.lower()

        # Apply function to book_stats
        book_stats_df = book_stats_df.with_columns(pl.col("book_id").map_elements(extract_genre2).alias("genre2"))

        # determine final genre
        df_with_final_genre = book_stats_df.with_columns(
            pl.struct(["genre1", "genre2"])
            .map_elements(lambda x: choose_genre(x["genre1"], x["genre2"]))
            .alias("genre")
        )

        return df_with_final_genre.drop(["genre1", "genre2"])

    def book_stats(self, *, genre2: bool = True, save: bool = True, force: bool = False) -> pl.DataFrame:
        """Build the book stats CSV.

        The stats contain:
            - book_id
            - chunk_id
            - count (Number of tokens (words))
            - types (Number of types (distinct words))
            - genre (Book genre)

        Procedure:
            - Iterate over cleaned books and count TOKENS & TYPES
            - Extrapolate the genre from the asscociations.csv
        """
        # If not forcing do not rebuild the dataframe
        if self.meta_dir.book_stats.is_file() and not force:
            return pl.read_csv(self.meta_dir.book_stats, separator=";")

        iter_items = hour_txt.StelaHourTxtItemsLoader.iter_items()

        results = []
        for item in iter_items:
            for book_path in item.book_path_list:
                words = book_path.read_tokenized()
                results.append(
                    {
                        "book_id": book_path.stem,
                        "chunk_id": item.chunk_id,
                        "count": len(words),
                        "types": len(set(words)),
                    }
                )
        book_stats = pl.DataFrame(results)

        associations = pl.read_csv(self.meta_dir.asscociations, separator=";")
        # Keep only the book and genre columns from the second DataFrame
        df_genres = associations.select(["book", "genre", "text_source", "book_title"]).unique()
        # Join the DataFrames on book_id = book
        result_df = book_stats.join(
            df_genres,
            left_on="book_id",
            right_on="book",
            how="left",  # Use left join to keep all rows from the first DataFrame
        )

        if genre2:
            result_df = self._add_better_genres(result_df)

        if save:
            result_df.write_csv(self.meta_dir.book_stats, separator=";", include_header=True)
        return result_df

    def book_stat_resume(self, *, save: bool = True, force: bool = False) -> pl.DataFrame:
        """Build the resume of the book_stats csv file."""
        # If not forcing do not rebuild the dataframe
        if self.meta_dir.book_stats_resume.is_file() and not force:
            return pl.read_csv(self.meta_dir.book_stats_resume, separator=";")

        # Load from book_stats
        book_stats = pl.read_csv(self.meta_dir.book_stats, separator=";")
        total_books = book_stats["book_id"].n_unique()
        unique_books = book_stats.unique(subset=["book_id"], keep="first")

        # TODO: need to add two more columns (text_source & book_title)
        resume = (
            unique_books.group_by("genre")
            .agg(
                [
                    pl.count("book_id").alias("num_books"),
                    pl.sum("count").alias("total_count"),
                    pl.sum("types").alias("total_types"),
                    pl.mean("count").round(3).alias("avg_count"),
                    pl.mean("types").round(3).alias("avg_types"),
                ]
            )
            .sort("num_books", descending=True)
            .with_columns((pl.col("num_books") / total_books * 100).round(3).alias("percent_books"))
        )

        if save:
            resume.write_csv(self.meta_dir.book_stats_resume, separator=";", include_header=True)
        return resume

    def extract_url_sources(self, *, save: bool = True, force: bool = False) -> list[str]:
        """Extract the domain names for all external book sources."""
        if self.meta_dir.book_source_domains.is_file() and not force:
            return self.meta_dir.book_source_domains.safe_readlines()

        df = pl.read_csv(self.meta_dir.asscociations, separator=";")
        soures_lst = df["text_source"].to_list()
        # keep only the domain name
        soures_lst = {urlparse(url).netloc for url in soures_lst}

        # save to disk
        if save:
            self.meta_dir.book_source_domains.safe_write_text("\n".join(soures_lst))

        return list(soures_lst)

    def scrap_extra_metadata(
        self, *, save: bool = True, force: bool = False, keep_cache: bool = True, cache_freq: int = 10
    ) -> dict:
        """Scrap the web to fetch extra book metadata.

        Using the text_source url from matched2.csv we are able to scrap extra information for
        a lot of the books (archive.org & gutenberg.org).

        Result:
            Results are saved as json :
                book_id -> {
                    source: url used
                    loc_class: genre classification according to USA Library archivists
                    subjects: other genres extracted from keywords (topic, subject descriptions etc...)
                    original_publication: Date of book publication
                    release_date: Date where the book was released on the public domain
                    unknown_source: Flag signifying the source (website does not have a scrapper)
                    failed: Flag signifying some error produced incomplete or missing data.
                }
        """
        if self.meta_dir.external_book_metadata.is_file() and not force:
            return self.meta_dir.external_book_metadata.read_json()

        cache_file = self.meta_dir.external_book_metadata.parent / ".scrapper.cache.json"
        results = {}

        # Resume from cache
        if cache_file.is_file():
            results = cache_file.read_json()

        def cache() -> None:
            """Cache temp results."""
            try:
                cache_file.write_json(results)
            except TypeError as e:
                pprint.pprint(results)
                raise e from e

        df = pl.read_csv(self.meta_dir.asscociations, separator=";")
        idx = 0
        for row in df.iter_rows(named=True):
            if row["book"] not in results:
                url = row["text_source"]
                item = web_scrappers.BookMetadata.fetch(url).to_dict()
                print(f"Adding items {idx}:{type(item)}")
                results[row["book"]] = item
                idx += 1

            # Cache every cache_freq items
            if idx % cache_freq == 0 and keep_cache:
                L.debug(f"Caching progress % {idx}")
                cache()

        if save:
            self.meta_dir.external_book_metadata.write_json(results)

        cache_file.unlink(missing_ok=True)
        return results

    def gather_line_length_stats(self, *, save: bool = True, force: bool = False) -> pl.DataFrame:
        """Gather statistics on line length across all stella books.

        Information:
            For each book in the 50h/** list of chunks (as it allows to not have duplicates),
            we count the size of each line of text.

            Resulting dataframe:

            LANG<str> | SPLIT<str> | CHUNK<str> | BOOK<str> | LINE<int> | LENGTH<int>

            This dataframe allows us to build two informations :

                - line length resume (min, max, average) lenght of each line in number of words.

                - line length frequency distribution, to see the average line size

            Lines in books have been re-balanced to contain single sentences (as per given punctuation).
        """
        if self.meta_dir.line_length_stats.is_file() and not force:
            return pl.read_csv(self.meta_dir.line_length_stats, separator=";")

        attrs = {"langs": (self.meta_dir.lang,), "hours": ("50h",)}
        iter_items = hour_txt.StelaHourTxtItemsLoader.iter_items(**attrs)
        stats = []

        for item in iter_items:
            for book_path in item.book_path_list:
                book_name = book_path.stem
                for idx, line in enumerate(book_path.safe_readlines()):
                    length = len(line.split())
                    if length > 1:
                        stats.append(
                            _LineLenghtStruct(
                                lang=self.meta_dir.lang,
                                split=item.hour_split,
                                chunk=item.chunk,
                                book=book_name,
                                line=idx,
                                length=length,
                            )
                        )

        df: pl.DataFrame = pl.DataFrame(stats)
        if save:
            df.write_csv(self.meta_dir.line_length_stats, separator=";", include_header=True)
        return df

    def build_by_size_stats(self, *, save: bool = True, force: bool = False) -> pl.DataFrame:
        """Gather all relevant statistics for the by_size datasetsplit of STELA."""
        if self.meta_dir.by_size_stats.is_file() and not force:
            return pl.read_csv(self.meta_dir.by_size_stats, separator=";")
        items = by_size.BySizeItemsLoader.iter_items(dataset_name="stela")

        word_stats_factory = word_stats.WordRejectionRates(
            filter_fn=lexicon.DictionairyWordCleaner(lang=self.meta_dir.lang).check,
            tokenizer=txt_utils.line_tokenizer,
        )
        results = []

        for chunk in items:
            lines = chunk.train_file.safe_readlines()
            wrd_stats = word_stats_factory.normalise_clean_chunk(lines)

            stats: _BySizeStats = {
                "lang": chunk.lang,
                "size": chunk.split,
                "chunk": chunk.chunk,
                "token_count": txt_utils.word_count(lines),
                "type_count": txt_utils.type_count(lines),
                "token_rejection_rate": wrd_stats.mean_token_rejection_rate(),
                "type_rejection_rate": wrd_stats.mean_type_rejection_rate(),
                "type_token_ratio": wrd_stats.mean_type_token_ratio(),
            }
            results.append(stats)

        df = pl.DataFrame(results)
        if save:
            df.write_csv(self.meta_dir.by_size_stats, separator=";", include_header=True)
        return df


@dataclasses.dataclass
class STELAMetaDir(MetadataDir):
    """STELA metadata Handler."""

    dataset_name: datasets.DATASET_NAMES = "stela"

    @property
    def asscociations(self) -> Path:
        """Path to a CSV containing wav - chunk - text associations."""
        return self.root_dir / "associations.csv"

    @property
    def book_stats(self) -> Path:
        """Path to CSV containing word counts per book."""
        return self.root_dir / "book_stats.csv"

    @property
    def book_stats_resume(self) -> Path:
        """Path to CSV containing word counts per book."""
        return self.root_dir / "book_stats_resume.csv"

    @property
    def book_source_domains(self) -> Path:
        """Path to txt containing all the sources for the audio book transcriptions."""
        return self.root_dir / "web_sources.txt"

    @property
    def external_book_metadata(self) -> Path:
        """Path to JSON containing external book metadata."""
        return self.root_dir / "scraped_book_data.json"

    @property
    def line_length_stats(self) -> Path:
        """Path to CSV containing stats on line length."""
        return self.root_dir / "line_length.csv"

    @property
    def manual_genre_list(self) -> Path:
        """Path to file containing manual genre classification of books."""
        return self.root_dir / "manual_genres.toml"

    @property
    def stratification_sanity_check(self) -> Path:
        """Path to stratification meta stats data."""
        return self.root_dir / "stratify-sanity-check.csv"

    @property
    def by_size_stats(self) -> Path:
        """Statististics of the by_size split of STELA."""
        return self.root_dir / "by_size_stats.csv"

    def line_length_by_count(self) -> pl.DataFrame:
        """Line length stats, grouped by count on unique books.

        This dataframe allows us to plot a frequency distribution of
        the line lenghts (in number of TOKENS(words)).

        Requires:
            - line_length_stats (line_length.csv)
        """
        df = pl.read_csv(self.line_length_stats, separator=";")
        return df.group_by("length").agg(pl.count().alias("count")).sort("length")

    def line_length_stats_resume(self) -> pl.DataFrame:
        """Line length resume statistics (min, max, average).

        Requires:
            - line_length_stats (line_length.csv)
        """
        df = pl.read_csv(self.line_length_stats, separator=";")
        return (
            df.group_by("lang")
            .agg(
                [
                    pl.min("length").alias("min_length"),
                    pl.mean("length").alias("avg_length"),
                    pl.max("length").alias("max_length"),
                    pl.count().alias("total_lines"),
                ]
            )
            .sort("lang")
        )

    def extract_genre_keywords(self) -> list[str]:
        """Extract genres from book_data.json to do frequency analysis.

        Requires:
            - external_book_metadata (book_data.json)
        """
        book_data = self.external_book_metadata.read_json()
        keyphrases = []
        for value in book_data.values():
            loc_class = value.get("loc_class")
            if loc_class:
                keyphrases.append(loc_class)

            subjects = value.get("subjects", [])
            if subjects:
                keyphrases.extend(subjects)

        keywords = []
        for phrase in keyphrases:
            word_list = phrase.lower().split()
            for word in word_list:
                clean_word = "".join([c for c in word if c in string.ascii_letters])
                if clean_word:
                    keywords.append(clean_word)

        return keywords

    def by_hour2by_genre(self) -> t.Iterable[_BookGenrePathStruct]:
        """Make the filesmap to build to get the by_genre.

        Requires:
            - book_stats
        """
        df_books = pl.read_csv(self.book_stats, separator=";")
        df_books = df_books.unique(subset=["book_id"])

        for row in df_books.iter_rows(named=True):
            hour, chunk = row["chunk_id"].split("_")
            item = hour_txt.StelaHourTxtItemsLoader.load(lang=self.lang, hour=hour, chunk=chunk)
            book_path = item.get_book_path(row["book_id"])
            yield _BookGenrePathStruct(path=book_path, genre=row["genre"])

    def by_size_stats_resume(self) -> pl.DataFrame:
        """Resume of the by_size statistics."""
        df = pl.read_csv(self.by_size_stats, separator=";")

        def sum_types(lang, size) -> int:
            size = f"{size:02}"
            words = []
            for item in by_size.BySizeItemsLoader.iter_items(dataset_name="stela", lang=(lang,), splits=(size,)):
                words.extend(item.train_file.read_tokenized())
            return len(set(words))

        df_resume = df.group_by(["lang", "size"]).agg(
            pl.col("token_count").sum(),
            pl.col("token_rejection_rate").mean(),
            pl.col("type_rejection_rate").mean(),
            pl.col("type_token_ratio").mean(),
        )

        # Extract type_count from dataset (requires original word-list to be computed)
        return df_resume.with_columns(
            pl.struct(["lang", "size"])
            .map_elements(lambda x: sum_types(x["lang"], x["size"]), return_dtype=int)
            .alias("type_count")
        )

    @property
    def builder(self) -> STELAMetaBuilder:
        """Load the metadata builder object."""
        return STELAMetaBuilder(dataset_cfg=self.dataset_cfg, meta_dir=self)
