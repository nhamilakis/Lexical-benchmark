import dataclasses
import logging
from pathlib import Path

import polars as pl

from lexical_benchmark import datasets, text_lib
from lexical_benchmark.dataloaders import childes as childes_dataloader
from lexical_benchmark.text_lib.parsing import cha as cha_parser

from .core import MetaBuilder, MetadataDir

L = logging.getLogger(__name__)


class CHILDESMetaBuilder(MetaBuilder):
    """Child build class for metadata."""

    dataset_cfg: datasets.CHILDESDatasetConfig
    meta_dir: "CHILDESMetaDir"

    def build_all(self) -> None:
        """Build all stats."""

    def build_child_speech_quantities(self, *, save: bool = True, force: bool = False) -> pl.DataFrame:
        """Build csv containing speech quantities."""
        if self.meta_dir.speech_quantities.is_file() and not force:
            return pl.read_csv(self.meta_dir.speech_quantities, separator=";")

        def word_count(item_id: str, lang_accent: str) -> int:
            """Count words of an item."""
            item: childes_dataloader.CHILDESTextLoader = childes_dataloader.CHILDESTextLoader.load(
                lang_accent=lang_accent, item_id=item_id, speech_type="child"
            )
            return text_lib.word_count(item.load_speech())

        # Load pre-existing metadata
        df_child_meta = pl.read_csv(
            self.meta_dir.child_meta,
            separator=";",
            schema=pl.Schema(
                {
                    "file_id": pl.String(),
                    "lang": pl.String(),
                    "child_gender": pl.String(),
                    "child_age": pl.String(),
                    "lang_accent": pl.String(),
                }
            ),
        )
        # Fixed miss-labeled multi-lingual items
        df_corrected = df_child_meta.with_columns(
            lang=pl.when(pl.col("file_id") == "Nicolopoulou_MP94_MP94_25noe_1")
            .then(pl.lit("eng"))
            .when(pl.col("file_id") == "MacWhinney_020617b")
            .then(pl.lit("eng"))
            .otherwise(pl.col("lang"))
        )

        df_processed = (
            df_corrected.with_columns(
                word_count=pl.struct(["file_id", "lang_accent"]).map_elements(
                    lambda x: word_count(x["file_id"], x["lang_accent"]), return_dtype=pl.Int32
                )
            )
            .filter((pl.col("word_count") > 0) & (pl.col("child_age") != "-"))
            .with_columns(
                age_months=pl.col("child_age")
                .map_elements(cha_parser.normalised_child_age, return_dtype=pl.Float64)
                .cast(pl.Int32)
            )
            .group_by(["age_months", "lang"])
            .agg(total_words=pl.col("word_count").sum())
            .sort(["lang", "age_months"])
        )
        if save:
            df_processed.write_csv(self.meta_dir.speech_quantities, separator=";", include_header=True)
        return df_processed

    def build_child_meta(self, *, save: bool = True, force: bool = False) -> pl.DataFrame:
        """Build csv containing child metadata."""
        if self.meta_dir.child_meta.is_file() and not force:
            return pl.read_csv(self.meta_dir.child_meta, separator=";")

        items = []
        for lang_accent in self.dataset_cfg.LANG_ACCENT[self.meta_dir.lang]:
            path = self.dataset_cfg.meta_dir / f"child_metadata_{lang_accent}.csv"
            if not path.is_file():
                L.info(f"Missing child-meta for {lang_accent}")
                continue
            df = pl.read_csv(path).with_columns(pl.lit(lang_accent).alias("lang_accent"))
            items.append(df)
            path.unlink()

        if not items:
            raise ValueError("Sources dataframes are mising, requires pre-process pipeline.")

        merged_df: pl.DataFrame = pl.concat(items)
        if save:
            merged_df.write_csv(self.meta_dir.child_meta, separator=";", include_header=True)
        return merged_df


@dataclasses.dataclass
class CHILDESMetaDir(MetadataDir):
    """CHILDES metadata Handler."""

    dataset_name: datasets.DATASET_NAMES = "childes"

    @property
    def builder(self) -> CHILDESMetaBuilder:
        """Load the metadata builder object."""
        return CHILDESMetaBuilder(dataset_cfg=self.dataset_cfg, meta_dir=self)

    @property
    def child_meta(self) -> Path:
        """Children metadata."""
        return self.root_dir / "child_metadata.csv"

    @property
    def speech_quantities(self) -> Path:
        """CSV containing speech quantities per age."""
        return self.root_dir / "speech_quantity.csv"
