"""Prepare the age-relevant generation file"""

from pathlib import Path

import pandas as pd

from lexical_benchmark import settings

root_dir: Path = settings.PATH.dataset_root / "CHILDES"
accent_lst = ["Eng-NA", "Eng-UK"]
out_dir: Path = settings.PATH.DATA_DIR / "gen"


def convert_to_months(age_string: str) -> float:
    """Convert age from 'years;months.days' format to total months.

    Args:
        age_string: Age in format 'y;mm.dd' (e.g., '1;09.04' for 1 year, 9 months, 4 days)

    Returns:
        Total age in months (rounded to 2 decimal places)

    """
    try:
        # Parse components
        years_str, remainder = age_string.split(";")
        months_str = remainder.split(".")[0]

        # Convert to numbers
        years = int(years_str)
        months = int(months_str)

        # Calculate total months
        total_months = (years * 12) + months
        return round(total_months, 0)
    except:
        return 0


def get_len(text: str):
    return len(text.split())


def concat_file(df, meta_df, text_dir):
    n = 0
    while n < meta_df.shape[0]:
        file = meta_df["file_id"].to_list()[n]
        with open(text_dir / f"{file}.txt", "r") as f:
            text = [line.strip() for line in f.readlines() if line.strip()]
        # only concatenate the non-empty list
        if len(text) > 0:
            # append the month
            text_df = pd.DataFrame([text]).T
            text_df.columns = ["text"]
            text_df.insert(0, "month", meta_df["month"].to_list()[n])
            text_df.insert(1, "file_id", file)
            text_df["sent_len"] = text_df["text"].apply(get_len)
            df = pd.concat([df, text_df])
        n += 1
    return df


# filter the matadata until 36 months
# age; filename; content
df = pd.DataFrame()
for accent in accent_lst:
    meta_df = pd.read_csv(root_dir / "metadata" / f"metadata_{accent}.csv")
    # convert into months
    meta_df["month"] = meta_df["child_age"].apply(convert_to_months)

    # filter all the months
    sel_df = meta_df[(meta_df["month"] < 37) & (meta_df["month"] > 0)]
    print(sel_df.shape[0])
    # loop over the dir
    text_dir = root_dir / "child" / accent
    df = concat_file(df, sel_df, text_dir)


# sort the df by month
df = df.sort_values("month", ascending=True)
df.to_csv(out_dir / "CHILDES.csv")
print(f"Saving the gen file to {out_dir}/CHILDES.csv")
