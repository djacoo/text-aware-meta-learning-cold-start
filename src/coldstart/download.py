import urllib.request
from pathlib import Path

# MovieLens-1M as extended by the MeLU authors: plots, directors, actors and cold-start splits
MELU_URL = "https://raw.githubusercontent.com/hoyeoplee/MeLU/master/movielens/ml-1m/"
STATE_FILES = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]
FILES = [
    "users.dat",
    "movies_extrainfos.dat",
    "m_rate.txt",
    "m_genre.txt",
    "m_director.txt",
    "m_actor.txt",
    "m_gender.txt",
    "m_age.txt",
    "m_occupation.txt",
    "m_zipcode.txt",
    *[f"{state}{suffix}.json" for state in STATE_FILES for suffix in ("", "_y")],
]


def download_movielens_1m(root="data/ml-1m"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = root / name
        # checked file by file, so an interrupted download picks up where it stopped
        if not path.exists():
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(MELU_URL + name, part)
            part.rename(path)


if __name__ == "__main__":
    download_movielens_1m()
