import argparse
import csv
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

# Movies & TV from Amazon Reviews 2023 (Hou et al., 2024), ratings of the 5-core version
URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
RATINGS = "benchmark/5core/rating_only/Movies_and_TV.csv"
META = "raw/meta_categories/meta_Movies_and_TV.jsonl"
STATES = ("warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state")
# Amazon labels mix genres with studios, formats and deals, so they are mapped to MovieLens genres
GENRES = {
    "Action": r"\baction\b",
    "Adventure": r"adventure",
    "Animation": r"animation|\banime\b",
    "Biography": r"biograph",
    "Comedy": r"comed",
    "Crime": r"crime",
    "Documentary": r"documentar",
    "Drama": r"drama",
    "Family": r"\bkids\b|family|children",
    "Fantasy": r"fantasy",
    "Film-Noir": r"noir",
    "History": r"histor",
    "Horror": r"horror",
    "Music": r"\bmusic\b|concert",
    "Musical": r"musical",
    "Mystery": r"mystery",
    "Romance": r"romance",
    "Sci-Fi": r"science fiction|sci-fi",
    "Sport": r"\bsports?\b",
    "Thriller": r"thriller|suspense",
    "War": r"\bwar\b|military",
    "Western": r"western",
}


def download_amazon(root):
    raw = Path(root) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in (RATINGS, META):
        path = raw / Path(name).name
        if not path.exists():
            part = path.with_suffix(".part")
            urllib.request.urlretrieve(URL + name, part)
            part.rename(path)
    return raw / Path(RATINGS).name, raw / Path(META).name


def as_list(value):
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(",")
    return [v.strip() for v in value if v.strip()]


def clean(text):
    return " ".join(re.sub(r"<[^>]+>", " ", text).split())


def title_key(title):
    # editions of the same film differ only in brackets, e.g. "[Blu-ray]" or "(Widescreen)"
    title = re.sub(r"[\[(][^\])]*[\])]", " ", title.lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", title).split())


def genres(labels):
    labels = " | ".join(labels).lower()
    return [genre for genre, pattern in GENRES.items() if re.search(pattern, labels)]


def item_info(meta):
    details = meta.get("details") or {}
    people = {
        "directors": as_list(details.get("Directors") or details.get("Director"))[:3],
        "actors": as_list(details.get("Starring") or details.get("Actors"))[:5],
    }
    labels = as_list(meta.get("categories")) + as_list(details.get("Genre"))
    return {
        "title": clean(meta.get("title") or ""),
        "rate": clean(str(details.get("MPAA rating") or "")),
        "genres": genres(labels),
        **people,
        "description": clean(" ".join(meta.get("description") or [])),
    }


def read_items(meta_path, counts):
    items = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            meta = json.loads(line)
            asin = meta.get("parent_asin")
            if asin not in counts:
                continue
            info = item_info(meta)
            if info["title"] and info["description"]:
                items[asin] = info
    return items


def build_amazon(root="data/amazon", n_items=5000, new_items=0.4, new_users=0.2, seed=0):
    root = Path(root)
    ratings_path, meta_path = download_amazon(root)

    # first pass: how often and since when each item was rated
    counts, first = Counter(), {}
    with open(ratings_path) as f:
        for user, asin, _, time in csv.reader(f):
            if user == "user_id":
                continue
            counts[asin] += 1
            first[asin] = min(first.get(asin, int(time)), int(time))

    # one product per film, the most rated edition, then the most rated films
    items = read_items(meta_path, counts)
    best = {}
    for asin in sorted(items, key=lambda a: (-counts[a], a)):
        best.setdefault(title_key(items[asin]["title"]), asin)
    kept = sorted(best.values(), key=lambda a: (-counts[a], a))[:n_items]
    item_id = {asin: i for i, asin in enumerate(kept)}

    # new items are the last ones to appear, as MeLU used the release year; Amazon users rate
    # few items, so 40% instead of MeLU's 20% leaves enough users in the new item states
    times = sorted(first[a] for a in kept)
    cutoff = times[int(len(times) * (1 - new_items))]
    new_item = {item_id[a]: first[a] >= cutoff for a in kept}

    # second pass: ratings of the kept items, a repeated rating keeps the latest
    latest = {}
    with open(ratings_path) as f:
        for user, asin, rating, time in csv.reader(f):
            if asin in item_id:
                key = (user, item_id[asin])
                if key not in latest or int(time) >= latest[key][1]:
                    latest[key] = (float(rating), int(time))

    # users with 13 to 100 ratings as in MeLU, a random share of them are the new users
    per_user = Counter(user for user, _ in latest)
    users = sorted(user for user, n in per_user.items() if 13 <= n <= 100)
    user_id = {user: i for i, user in enumerate(users)}
    new_user = set(random.Random(seed).sample(range(len(users)), int(len(users) * new_users)))

    states = {name: ({}, {}) for name in STATES}
    for (user, item), (rating, _) in sorted(latest.items(), key=lambda kv: kv[1][1]):
        if user not in user_id:
            continue
        u = user_id[user]
        name = STATES[(u in new_user) + 2 * new_item[item]]
        states[name][0].setdefault(str(u), []).append(item)
        states[name][1].setdefault(str(u), []).append(rating)
    for name, (movies, y) in states.items():
        (root / f"{name}.json").write_text(json.dumps(movies))
        (root / f"{name}_y.json").write_text(json.dumps(y))

    info = [items[a] | {"asin": a, "first_rating": first[a]} for a in kept]
    (root / "items.json").write_text(json.dumps({"items": info, "n_users": len(users)}))


def describe(item):
    parts = [f"{item['title']}."]
    if item["genres"]:
        parts.append(f"{', '.join(item['genres'])}.")
    if item["directors"]:
        parts.append(f"Directed by {', '.join(item['directors'])}.")
    if item["actors"]:
        parts.append(f"Starring {', '.join(item['actors'])}.")
    parts.append(item["description"])
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/amazon")
    parser.add_argument("--items", type=int, default=5000)
    args = parser.parse_args()
    build_amazon(args.root, args.items)


if __name__ == "__main__":
    main()
