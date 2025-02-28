from dataclasses import dataclass
from pathlib import Path

import polars as pl

from lexical_benchmark import datasets
from lexical_benchmark.dataloaders import hour_txt

from .core import MetaBuilder, MetadataDir


@dataclass
class STELAMetaBuilder(MetaBuilder):
    """STELA Metadata Extractor."""

    dataset_cfg: datasets.STELADatasetConfig
    meta_dir: "STELAMetaDir"

    def build_all(self, *, force: bool = False) -> None:
        """Build all stats."""
        _ = self.book_stats(save=True, force=force)

    def book_stats(self, *, save: bool = True, force: bool = False) -> pl.DataFrame:
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
            return pl.read_csv(self.meta_dir, separator=";")

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

        if save:
            result_df.write_csv(self.meta_dir.book_stats, separator=";", include_header=True)
        return result_df

    def get_book_genres(self, lang: str) -> pl.DataFrame:
        """Load book genres from original InfTrain dataset."""
        book_metadata_file = self.dataset_cfg.original_root / "metadata" / "book_metadata.csv"
        if not book_metadata_file.is_file():
            raise ValueError(f"File {book_metadata_file} not found")
        book_meta = pl.read_csv(book_metadata_file)

        book_meta = book_meta.filter(pl.col("language") == lang)
        """
        new_book_stats = book_stats.join(
            book_meta_en.select(["book_id", "book_title", "genre", "text_source"]),
            on="book_id",
            how="left",
        )
        """
        return book_meta[["book_id", "book_title", "genre"]]

    def book_stat_resume(self, *, save: bool = True, force: bool = False) -> pl.DataFrame:
        """Build the resume of the book_stats csv file."""
        # If not forcing do not rebuild the dataframe
        if self.meta_dir.book_stats_resume.is_file() and not force:
            return pl.read_csv(self.meta_dir.book_stats_resume, separator=";")

        # Load from book_stats
        book_stats = pl.read_csv(self.meta_dir.book_stats, separator=";")
        total_books = book_stats["book_id"].n_unique()
        unique_books = book_stats.unique(subset=["book_id"], keep="first")

        # TODO need to add two more columns (text_source & book_title)
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


@dataclass
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
    def builder(self) -> STELAMetaBuilder:
        """Load the metadata builder object."""
        return STELAMetaBuilder(dataset_cfg=self.dataset_cfg, meta_dir=self)
