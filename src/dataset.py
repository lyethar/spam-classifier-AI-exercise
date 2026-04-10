"""Load SMS spam data from UCI (tab-separated) or Kaggle-style CSV (v1, v2)."""

import pandas as pd


def load_spam_dataset(path: str) -> pd.DataFrame:
    with open(path, encoding="latin-1") as f:
        first = f.readline()

    header = first.lstrip("\ufeff").strip().lower()
    if header.startswith("v1,"):
        df = pd.read_csv(path, encoding="latin-1")
        if "v1" not in df.columns or "v2" not in df.columns:
            raise ValueError(
                "CSV header looks like Kaggle (v1) but columns v1/v2 are missing."
            )
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            names=["label", "message"],
            encoding="latin-1",
            header=None,
        )

    df["label"] = df["label"].astype(str).str.strip()
    df["message"] = df["message"].astype(str)
    return df
