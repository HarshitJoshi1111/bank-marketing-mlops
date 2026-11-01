"""Stage 3: stratified train/test split (the positive class is only ~11 %)."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.params import ROOT, load_params

IN = ROOT / "data" / "prepared" / "bank.csv"
OUT = ROOT / "data" / "split"


def main() -> None:
    params = load_params()["split"]
    df = pd.read_csv(IN)
    train, test = train_test_split(df, test_size=params["test_size"], random_state=params["seed"], stratify=df["y"])
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    test.to_csv(OUT / "test.csv", index=False)
    print(f"train {len(train)} rows (positive {train['y'].mean():.3f}), test {len(test)} rows (positive {test['y'].mean():.3f})")


if __name__ == "__main__":
    main()
