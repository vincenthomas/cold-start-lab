"""Download MovieLens-1M into data/.

MovieLens data may not be redistributed, so this repo ships the script,
not the data. Source: https://grouplens.org/datasets/movielens/1m/
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL) as r:
        buf = io.BytesIO(r.read())
    with zipfile.ZipFile(buf) as z:
        for name in z.namelist():
            if name.endswith(".dat"):
                target = DATA_DIR / Path(name).name
                target.write_bytes(z.read(name))
                print(f"wrote {target}")
    print("done.")


if __name__ == "__main__":
    main()
