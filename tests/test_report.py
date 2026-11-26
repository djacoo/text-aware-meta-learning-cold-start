import json

from coldstart.report import main

STATES = ("warm_test", "user_cold", "item_cold", "user_item_cold")


def test_table(tmp_path):
    for name, mae in (("a", 0.8), ("b", 0.7)):
        result = {"model": {s: {"mae": [mae, mae + 0.02]} for s in STATES}}
        result["user mean"] = {s: {"mae": [0.9, 0.9]} for s in STATES}
        (tmp_path / f"{name}.json").write_text(json.dumps(result))
    out = tmp_path / "table.tex"
    rows = [
        f"A={tmp_path / 'a.json'}",
        f"B={tmp_path / 'b.json'}",
        f"Mean={tmp_path / 'a.json'}:user mean",
    ]
    main(["table", str(out), *rows])
    lines = out.read_text().splitlines()
    # two lines of header, then one row per method, the lowest MAE of each column in bold
    assert lines[5].startswith("B & \\textbf{0.710}")
    assert lines[6].startswith("Mean & 0.900")


def test_curves(tmp_path):
    rows = {
        k: {"random": {"mae": 1.0 / int(k)}, "hybrid": {"mae": 0.9 / int(k)}} for k in ("1", "3")
    }
    (tmp_path / "al.json").write_text(json.dumps({"user_cold": rows, "item_cold": rows}))
    main(["curves", str(tmp_path / "al.json"), str(tmp_path / "al.pdf")])
    assert (tmp_path / "al.pdf").stat().st_size > 0
