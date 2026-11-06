"""Stage 1: download the UCI Bank Marketing dataset and extract the CSV.

The UCI archive contains two more zip files; the "additional" version has 41,188 phone calls with
20 features, including five economic indicators (employment rate, consumer price index, ...).
"""
import io
import zipfile

import requests

from src.params import ROOT, load_params

OUT = ROOT / "data" / "raw" / "bank-additional-full.csv"


def extract(archive: bytes, member: str) -> bytes:
    """Find `member` in the outer zip or in any zip nested inside it."""
    with zipfile.ZipFile(io.BytesIO(archive)) as outer:
        if member in outer.namelist():
            return outer.read(member)
        for name in outer.namelist():
            if name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    if member in inner.namelist():
                        return inner.read(member)
    raise FileNotFoundError(f"{member} not found in the archive")


def main() -> None:
    params = load_params()["data"]
    response = requests.get(params["url"], timeout=60)
    response.raise_for_status()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(extract(response.content, params["file"]))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
