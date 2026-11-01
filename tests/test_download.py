import io
import zipfile

import pytest

from src.download import extract


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buffer.getvalue()


def test_extract_finds_file_in_nested_zip():
    inner = zip_bytes({"bank-additional/bank-additional-full.csv": b"a;b\n1;2\n"})
    outer = zip_bytes({"bank.zip": zip_bytes({"bank.csv": b"x"}), "bank-additional.zip": inner})
    assert extract(outer, "bank-additional/bank-additional-full.csv") == b"a;b\n1;2\n"


def test_extract_missing_file():
    with pytest.raises(FileNotFoundError):
        extract(zip_bytes({"other.csv": b""}), "bank-additional/bank-additional-full.csv")
