import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STATES = {
    "warm_test": "Warm",
    "user_cold": "User cold",
    "item_cold": "Item cold",
    "user_item_cold": "User-item cold",
}


def table(rows, metric="mae"):
    # rows are (label, json file of coldstart.evaluate, method), mean and std over seeds
    values = []
    for _, path, method in rows:
        with open(path) as f:
            result = json.load(f)[method]
        values.append([(np.mean(result[s][metric]), np.std(result[s][metric])) for s in STATES])
    means = np.array([[mean for mean, _ in row] for row in values])
    # the best value of each column in bold, lowest MAE or highest nDCG
    best = means.argmin(axis=0) if metric == "mae" else means.argmax(axis=0)
    lines = ["\\begin{tabular}{l" + "c" * len(STATES) + "}", "\\hline"]
    lines.append("Method & " + " & ".join(STATES.values()) + " \\\\")
    lines.append("\\hline")
    for i, ((label, _, _), row) in enumerate(zip(rows, values, strict=True)):
        cells = []
        for j, (mean, std) in enumerate(row):
            text = f"{mean:.3f}" if best[j] != i else f"\\textbf{{{mean:.3f}}}"
            cells.append(f"{text}{{\\scriptsize$\\pm${std:.3f}}}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def curves(path, out, metric="mae"):
    # one panel per state, error of each strategy against the number of rated items
    with open(path) as f:
        result = json.load(f)
    fig, axes = plt.subplots(1, len(result), figsize=(3 * len(result), 2.6))
    for ax, (state, by_k) in zip(np.atleast_1d(axes), result.items(), strict=True):
        ks = sorted(by_k, key=int)
        for name in by_k[ks[0]]:
            values = [by_k[k][name][metric] for k in ks]
            ax.plot([int(k) for k in ks], values, marker="o", label=name)
        ax.set_title(STATES.get(state, state))
        ax.set_xlabel("k")
        ax.set_xticks([int(k) for k in ks])
    np.atleast_1d(axes)[0].set_ylabel(metric.upper())
    np.atleast_1d(axes)[0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("table", help="LaTeX table from coldstart.evaluate --json files")
    t.add_argument("out")
    t.add_argument("rows", nargs="+", help='"label=file.json" or "label=file.json:method"')
    t.add_argument("--metric", default="mae")
    c = commands.add_parser("curves", help="figure from coldstart.run_active_learning --json")
    c.add_argument("results")
    c.add_argument("out")
    c.add_argument("--metric", default="mae")
    args = parser.parse_args(argv)

    if args.command == "table":
        rows = []
        for row in args.rows:
            label, source = row.split("=", 1)
            path, _, method = source.partition(":")
            rows.append((label, path, method or "model"))
        with open(args.out, "w") as f:
            f.write(table(rows, args.metric))
    else:
        curves(args.results, args.out, args.metric)


if __name__ == "__main__":
    main()
