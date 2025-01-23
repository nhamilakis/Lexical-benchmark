"""Tools to collect and use transcriptions from the Geenration dataset."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from lexical_benchmark import settings
from lexical_benchmark.settings import PATH, month2chunk


class CHILDESMonth:
    """format human reference data into the target csv""" 
    def __init__(self, root_dir: Path, out_dir: Path, speaker:str='child',accent_lst:list = ["Eng-NA", "Eng-UK"], max_months: int = 36) -> None:
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_months = max_months
        self.speaker = speaker
        self.accent_lst = accent_lst
        self.df = pd.DataFrame()

    @staticmethod
    def convert_to_months(age_string: str) -> float:
        """Convert annotated age info into months"""
        try:
            years_str, remainder = age_string.split(";")
            months_str = remainder.split(".")[0]
            total_months = (int(years_str) * 12) + int(months_str)
            return round(total_months, 0)
        except:
            return 0

    @staticmethod
    def get_len(text: str):
        return len(text.split())

    def concat_file(self, meta_df: pd.DataFrame, text_dir: Path) -> pd.DataFrame:
        """concatenate fiel in different months"""
        for n in range(meta_df.shape[0]):
            file = meta_df["file_id"].iloc[n]
            try:
                with open(text_dir / f"{file}.txt", "r") as f:
                    text = [line.strip() for line in f.readlines() if line.strip()]
                if text:
                    text_df = pd.DataFrame(text, columns=["text"])
                    text_df["month"] = meta_df["month"].iloc[n]
                    text_df["file_id"] = file
                    text_df["sent_len"] = text_df["text"].progress_apply(self.get_len)
                    self.df = pd.concat([self.df, text_df])

            except FileNotFoundError:
                continue
        return self.df

    def process(self):
        """concatenate all the data."""
        for accent in self.accent_lst:
            meta_df = pd.read_csv(self.root_dir / "metadata" / f"metadata_{accent}.csv")
            meta_df["month"] = meta_df["child_age"].progress_apply(self.convert_to_months)
            sel_df = meta_df[(meta_df["month"] < self.max_months + 1) & (meta_df["month"] > 0)]

            text_dir = self.root_dir / self.speaker / accent
            self.concat_file(sel_df, text_dir)

        self.df = self.df.sort_values("month", ascending=True)
        # create the directory if not exsiting

        self.df.to_csv(self.out_dir / "CHILDES.csv")
        print(f"Saving the gen file to {self.out_dir}/CHILDES.csv")





class ModelMonth:
    """annotate model info based on different estimations"""

   def __init__(self, out_dir: Path, target_months: list[int] = [6, 12, 18, 24, 30, 36]):
       self.out_dir = out_dir
       self.target_months = target_months
       self.month_dict = self._get_prop()
       
   def _get_prop(self) -> dict:
       month_dict = {}
       for n, target in enumerate(self.target_months):
           if n == 0:
               for i in range(1, target + 1):
                   month_dict[i] = [{target: 1}]
           else:
               prev_target = self.target_months[n - 1]
               for i in range(prev_target, target + 1):
                   month_diff_prop = 1 - (i - prev_target) / (target - prev_target)
                   month_dict[i] = [{prev_target: month_diff_prop}, 
                                  {target: 1 - month_diff_prop}]
       return month_dict
   
   @staticmethod
   def append_model(df: pd.DataFrame,
                   proportion_list: list[dict[int, float]],
                   word_count_column: str = "sent_len",
                   target_column: str = "model") -> pd.DataFrame:
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
           while (current_threshold_idx < len(thresholds) - 1 and 
                  row_proportion > thresholds[current_threshold_idx]):
               current_threshold_idx += 1
           result_df.iloc[idx, result_df.columns.get_loc(target_column)] = values[current_threshold_idx]
           
       return result_df

   def process(self):
       data = pd.read_csv(self.out_dir / "CHILDES.csv")
       data_all = pd.DataFrame()
       
       for month, data_group in data.groupby("month"):
           month_prop = self.month_dict[month]
           df = self.append_model(data_group, month_prop)
           data_all = pd.concat([data_all, df])
           
       data_all.to_csv(self.out_dir / "CHILDES_model.csv")

# Usage
processor = MonthlyDataProcessor(out_dir=settings.PATH.DATA_DIR / "gen")
processor.process()