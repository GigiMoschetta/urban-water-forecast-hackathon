from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int


OFFICIAL_FOLDS = [
    Fold("fold_1", 202001, 202112, 202201, 202212),
    Fold("fold_2", 202001, 202212, 202301, 202312),
    Fold("fold_3", 202001, 202312, 202401, 202412),
    Fold("fold_4", 202001, 202412, 202501, 202512),
]


def split_by_fold(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[(df["period_ym"] >= fold.train_start) & (df["period_ym"] <= fold.train_end)].copy()
    valid = df[(df["period_ym"] >= fold.valid_start) & (df["period_ym"] <= fold.valid_end)].copy()
    return train, valid
