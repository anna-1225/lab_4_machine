
from pathlib import Path
import pandas as pd

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "processed.cleveland.data"
)

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=COLUMNS,
    sep=",",
    na_values=["?"],
)

df = df.apply(pd.to_numeric, errors="raise")

print("Размер датасета:")
print(df.shape)

print("\nПервые пять строк:")
print(df.head())

print("\nТипы столбцов:")
print(df.dtypes)

print("\nКоличество пропусков:")
print(df.isna().sum())

print("\nКоличество дубликатов:")
print(df.duplicated().sum())

print("\nРаспределение исходной переменной num:")
print(df["num"].value_counts().sort_index())

if df["num"].isna().any():
    raise ValueError("Целевая переменная содержит пропуски")

df["target"] = (df["num"] > 0).astype(int)

X = df.drop(columns=["num", "target"])
y = df["target"]

print("\nРаспределение бинарной целевой переменной:")
print(y.value_counts().sort_index())

print("\nРазмер матрицы признаков X:")
print(X.shape)

print("\nРазмер целевой переменной y:")
print(y.shape)

print("\nОписательная статистика:")
print(X.describe().round(2))
