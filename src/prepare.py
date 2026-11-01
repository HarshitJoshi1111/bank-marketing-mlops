"""Stage 2: validate the raw data and turn it into a clean modelling table.

- checks the schema (columns, types, allowed values) so a changed source file fails loudly
- drops leaking columns (see params.yaml)
- pdays = 999 means "not contacted in an earlier campaign": replaced by a 0/1 flag
- target y (yes/no) -> 1/0
"""
import pandas as pd

from src.params import ROOT, load_params

RAW = ROOT / "data" / "raw" / "bank-additional-full.csv"
OUT = ROOT / "data" / "prepared" / "bank.csv"

NUMERIC = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORIES = {
    "job": {"admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed",
            "services", "student", "technician", "unemployed", "unknown"},
    "marital": {"divorced", "married", "single", "unknown"},
    "education": {"basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course",
                  "university.degree", "unknown"},
    "default": {"no", "yes", "unknown"},
    "housing": {"no", "yes", "unknown"},
    "loan": {"no", "yes", "unknown"},
    "contact": {"cellular", "telephone"},
    "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"},
    "day_of_week": {"mon", "tue", "wed", "thu", "fri"},
    "poutcome": {"failure", "nonexistent", "success"},
    "y": {"no", "yes"},
}
NOT_CONTACTED = 999


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError with every problem found, not just the first one."""
    problems = []
    missing = (set(NUMERIC) | set(CATEGORIES)) - set(df.columns)
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
    for col in NUMERIC:
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} is not numeric")
    for col, allowed in CATEGORIES.items():
        if col in df:
            unexpected = set(df[col].dropna().unique()) - allowed
            if unexpected:
                problems.append(f"{col}: unexpected values {sorted(unexpected)[:5]}")
    nulls = df.isna().sum()
    if nulls.any():
        problems.append(f"missing values: {nulls[nulls > 0].to_dict()}")
    if "age" in df and not df["age"].between(15, 100).all():
        problems.append("age outside 15-100")
    if problems:
        raise ValueError("data validation failed:\n  " + "\n  ".join(problems))


def prepare(df: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    df = df.drop(columns=drop_columns).drop_duplicates()
    df["previously_contacted"] = (df["pdays"] != NOT_CONTACTED).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != NOT_CONTACTED, -1)
    df["y"] = (df["y"] == "yes").astype(int)
    return df.reset_index(drop=True)


def main() -> None:
    params = load_params()["prepare"]
    raw = pd.read_csv(RAW, sep=";")
    validate(raw)
    df = prepare(raw, params["drop_columns"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(raw)} rows -> {len(df)} after removing duplicates, {df.shape[1] - 1} features, "
          f"positive rate {df['y'].mean():.3f}")


if __name__ == "__main__":
    main()
