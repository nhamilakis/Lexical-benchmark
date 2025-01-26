"""Tools to collect and use transcriptions from the Geenration dataset."""
from pathlib import Path

import pandas as pd
from tqdm import tqdm


class GenerationMerger:
    """Merge model generations and re-distribute by actual months."""

    def __init__(self, hour_per_year: int, gen_dir: Path, lang: str = "EN", filename: str = "gen.csv") -> None:
        self.hour_per_year = hour_per_year
        self.gen_dir = gen_dir
        self.lang = lang
        self.filename = filename

    def concat_files(self) -> pd.DataFrame:
        gen_all = pd.DataFrame()
        info_dict = {}
        for dataset in self.gen_dir.iterdir():
            if dataset.is_dir():
                for month in (dataset / "by_month" / self.lang).iterdir():
                    for chunk in month.iterdir():
                        for model in chunk.iterdir():
                            # load the target file to make sure we have the corresponding file
                            gen_path_year = model / f"{self.hour_per_year}_hour_per_year.csv"
                            gen_path = model / self.filename

                            if gen_path_year.exists():
                                gen_path = gen_path_year
                            elif gen_path.exists():
                                gen_path = gen_path  # Use regular gen_path
                            else:
                                print(f"No generated file in {model}. Skip")
                                continue

                            gen = pd.read_csv(gen_path).loc[:, "month":]
                            info_dict = {"dataset": dataset.name, "chunk": chunk.name, "model_type": model.name}
                            gen = gen.assign(**info_dict)
                            gen_all = pd.concat([gen_all, gen])

                            if month.name == '24':    # note here we hard_coded the generatin!
                                print("Duplicating the generation from month 24")
                                gen = pd.read_csv(gen_path).loc[:, "month":]
                                # remove the code that is higher than 24 (included)
                                print(f"before removing the additional rows {gen.shape[0]}")
                                gen = gen[gen['month']<24]
                                print(f"after removing the additional rows {gen.shape[0]}")

                                info_dict = {"dataset": dataset.name, "chunk": "01", "model_type": model.name}
                                gen = gen.assign(**info_dict)
                                gen_all = pd.concat([gen_all, gen])

        if not gen_all.empty:
            gen_all.to_csv(self.gen_dir / self.filename)
            print(f"Saving the concatenated generation to {self.gen_dir / self.filename}")
        return gen_all, info_dict


    def save_grouped_files(self, df: pd.DataFrame, info_dict: dict) -> None:
        """Save the monthly gen."""
        col_header = list(info_dict.keys()) + ["month"]
        for group, gen_group in df.groupby(col_header):
            file_dir = (
                self.gen_dir
                / group[0]
                / f"{self.hour_per_year}_hour_per_year"
                / self.lang
                / f"{group[3]:02d}"
                / group[1]
                / group[2]
            )
            file_dir.mkdir(parents=True, exist_ok=True)
            # pop the info headers
            gen_group = gen_group.drop(col_header, axis=1)
            gen_group.to_csv(file_dir / self.filename)
            print(f"Saving the monthly generation to {file_dir / self.filename}")

    def process(self) -> None:
        """Concatenate and redistribute by months."""
        gen_all, info_dict = self.concat_files()
        self.save_grouped_files(gen_all, info_dict)


class ModelMonth:
    """Annotate model info."""

    def __init__(self, child_df: Path, out_dir: Path, target_months: list[int] = [6, 12, 18, 24, 30, 36]) -> None:
        self.child_df = child_df
        self.out_dir = out_dir
        self.out_dir.parent.mkdir(parents=True, exist_ok=True)
        self.target_months = target_months
        self.month_dict = self._get_prop()

    def _get_prop(self) -> dict:
        """Get prop of different pseudo months."""
        month_dict = {}
        for n, target in enumerate(self.target_months):
            if n == 0:
                for i in range(1, target + 1):
                    month_dict[i] = [{target: 1}]
            else:
                prev_target = self.target_months[n - 1]
                for i in range(prev_target, target + 1):
                    month_diff_prop = 1 - (i - prev_target) / (target - prev_target)
                    month_dict[i] = [{prev_target: month_diff_prop}, {target: 1 - month_diff_prop}]
        return month_dict

    @staticmethod
    def append_model(
        df: pd.DataFrame,
        proportion_list: list[dict[int, float]],
        word_count_column: str = "sent_len",
        target_column: str = "model",
    ) -> pd.DataFrame:
        """Map model months with human month."""
        if word_count_column not in df.columns:
            raise ValueError(f"Column '{word_count_column}' not found")

        result_df = df.copy()
        result_df = result_df.sort_values(by=word_count_column)
        total_words = result_df[word_count_column].sum()
        cumsum = result_df[word_count_column].cumsum()
        cumsum_proportions = cumsum / total_words

        result_df[target_column] = None
        thresholds = []
        values = []
        cumulative_prop = 0

        for prop_dict in proportion_list:
            for value, proportion in prop_dict.items():
                cumulative_prop += proportion
                thresholds.append(cumulative_prop)
                values.append(value)

        current_threshold_idx = 0
        for idx, row_proportion in enumerate(cumsum_proportions):
            while current_threshold_idx < len(thresholds) - 1 and row_proportion > thresholds[current_threshold_idx]:
                current_threshold_idx += 1
            result_df.iloc[idx, result_df.columns.get_loc(target_column)] = values[current_threshold_idx]

        return result_df

    def process(self) -> None:
        data = pd.read_csv(self.child_df)
        data_all = pd.DataFrame()

        for month, data_group in tqdm(data.groupby("month")):
            month_prop = self.month_dict[month]
            df = self.append_model(data_group, month_prop)
            data_all = pd.concat([data_all, df])

        data_all.to_csv(self.out_dir)
        print(f"Saving the gen file to {self.out_dir}")


class CHILDESMonth:
    """Format human reference data into the target csv."""

    def __init__(
        self,
        root_dir: Path,
        out_dir: Path,
        speaker: str = "child",
        accent_lst: list = ["Eng-NA", "Eng-UK"],
        max_months: int = 36,
    ) -> None:
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.out_dir.parent.mkdir(parents=True, exist_ok=True)
        self.max_months = max_months
        self.speaker = speaker
        self.accent_lst = accent_lst
        self.df = pd.DataFrame()

    @staticmethod
    def convert_to_months(age_string: str) -> float:
        """Convert annotated age info into months."""
        try:
            years_str, remainder = age_string.split(";")
            months_str = remainder.split(".")[0]
            total_months = (int(years_str) * 12) + int(months_str)
            return round(total_months, 0)
        except:
            return 0

    @staticmethod
    def get_len(text: str) -> int:
        return len(text.split())

    def concat_file(self, meta_df: pd.DataFrame, text_dir: Path) -> pd.DataFrame:
        """Concatenate file in different months."""
        for n in range(meta_df.shape[0]):
            file = meta_df["file_id"].iloc[n]
            try:
                with open(text_dir / f"{file}.txt", "r") as f:
                    text = [line.strip() for line in f.readlines() if line.strip()]
                if text:
                    text_df = pd.DataFrame(text, columns=["text"])
                    text_df["month"] = meta_df["month"].iloc[n]
                    text_df["file_id"] = file
                    text_df["sent_len"] = text_df["text"].apply(self.get_len)
                    self.df = pd.concat([self.df, text_df])

            except FileNotFoundError:
                continue
        return self.df

    def process(self) -> None:
        """Concatenate all the data."""
        for accent in self.accent_lst:
            meta_df = pd.read_csv(self.root_dir / "metadata" / f"metadata_{accent}.csv")
            meta_df["month"] = meta_df["child_age"].apply(self.convert_to_months)
            sel_df = meta_df[(meta_df["month"] < self.max_months + 1) & (meta_df["month"] > 0)]

            text_dir = self.root_dir / self.speaker / accent
            self.concat_file(sel_df, text_dir)

        self.df = self.df.sort_values("month", ascending=True)
        # Move column to second-to-last position
        cols = self.df.columns.tolist()
        cols.remove("text")
        cols.insert(len(cols) - 1, "text")
        self.df = self.df[cols]
        self.df.to_csv(self.out_dir)
        print(f"Saving the gen file to {self.out_dir}")
