import json

from coldstart.amazon import build_amazon, genres, title_key
from coldstart.preprocess import load_data, load_tasks


def write_raw(root):
    raw = root / "raw"
    raw.mkdir(parents=True)
    films = [f"F{i}" for i in range(30)]
    rows = ["user_id,parent_asin,rating,timestamp"]
    for u in range(30):
        for i, asin in enumerate(films):
            # item i first appears at time i, so the last items are the new ones
            rows.append(f"U{u},{asin},{1 + (u + i) % 5}.0,{i * 100 + u}")
    # a second edition of F0 with fewer ratings, and an item without description
    rows += [f"U{u},F0BR,5.0,{5000 + u}" for u in range(3)]
    rows += [f"U{u},NODESC,4.0,{6000 + u}" for u in range(30)]
    (raw / "Movies_and_TV.csv").write_text("\n".join(rows))

    metas = [
        {
            "parent_asin": asin,
            "title": f"Film {i}" if asin != "F3" else "Film 3 [Blu-ray]",
            "description": [f"<b>Story</b> number {i}."],
            "categories": ["Movies & TV", "Drama"],
            "details": {"Directors": "Ann Lee, Bo Ray", "Starring": ["Cy Dee"], "MPAA rating": "R"},
        }
        for i, asin in enumerate(films)
    ]
    metas.append({"parent_asin": "F0BR", "title": "Film 0 (Blu-ray)", "description": ["Same."]})
    metas.append({"parent_asin": "NODESC", "title": "No text", "description": []})
    (raw / "meta_Movies_and_TV.jsonl").write_text("\n".join(json.dumps(m) for m in metas))


def test_title_key():
    assert title_key("The Matrix [Blu-ray]") == title_key("The Matrix (Widescreen Edition)")
    assert title_key("The Matrix") != title_key("The Matrix Reloaded")


def test_genres():
    labels = ["Movies & TV", "Science Fiction & Fantasy", "Warner Home Video", "Kids & Family"]
    assert genres(labels) == ["Family", "Fantasy", "Sci-Fi"]
    assert genres(["Musicals & Performing Arts", "Military & War"]) == ["Musical", "War"]


def test_build_and_load_amazon(tmp_path):
    write_raw(tmp_path)
    build_amazon(tmp_path, n_items=100)
    items = json.loads((tmp_path / "items.json").read_text())["items"]
    asins = [item["asin"] for item in items]
    # the Blu-ray copy of Film 0 and the item without description are gone
    assert sorted(asins) == sorted(f"F{i}" for i in range(30))
    assert items[0]["description"] == "Story number 0."
    assert items[0]["directors"] == ["Ann Lee", "Bo Ray"]
    assert items[0]["genres"] == ["Drama"]

    states = {}
    for name in ("warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"):
        states[name] = json.loads((tmp_path / f"{name}.json").read_text())
    new_items = {i for movies in states["item_cold_state"].values() for i in movies}
    assert {asins[i] for i in new_items} == {f"F{i}" for i in range(18, 30)}
    assert len(states["user_cold_state"]) == 6
    assert not set(states["warm_state"]) & set(states["user_cold_state"])

    data = load_data(tmp_path)
    assert data.user_features == {}
    assert data.item_features["genre"].shape[0] == 30
    assert data.descriptions[0].startswith("Film 0. Drama. Directed by Ann Lee, Bo Ray.")
    tasks = load_tasks(data, "warm_train", seed=0)
    assert tasks and all(len(task.query_items) == 10 for task in tasks)
