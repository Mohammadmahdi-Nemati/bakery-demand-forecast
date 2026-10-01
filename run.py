"""Startet die komplette Auswertung und schreibt die Ergebnisse nach reports/.

Aufruf:
    python run.py --data-dir ../bakery-sales-db/data
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bakery_forecast import pipeline  # noqa: E402

REPORTS = Path(__file__).parent / "reports"
plt.rcParams.update({"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "path.simplify": True})

POLICY_SHORT = {
    "Heutige Planung (Ø 4 Wochen + 12 %)": "Heute",
    "Heutige Planung + Feiertagsregel": "Heute +\nFeiertagsregel",
    "Modell + Newsvendor-Menge": "Modell",
    "Modell + Newsvendor + Feiertagsregel": "Modell +\nFeiertagsregel",
}


def plot_costs(policies, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [POLICY_SHORT[n] for n in policies.index]
    waste = policies["abfall_kosten_eur"].to_numpy() / 1000
    lost = policies["entgangene_marge_eur"].to_numpy() / 1000
    ax.bar(names, waste, color="#c2410c", label="Abfall (Herstellkosten)")
    ax.bar(names, lost, bottom=waste, color="#94a3b8", label="Entgangene Marge (Ausverkauf)")
    for i, total in enumerate(waste + lost):
        ax.text(i, total + 0.4, f"{total:.1f} T€".replace(".", ","), ha="center", fontsize=9)
    ax.set_ylabel("Tausend €")
    ax.set_title("Kosten der Fehlplanung, 8 Testwochen, 3 Filialen", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_ylim(0, (waste + lost).max() * 1.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_holidays(sims, path):
    """Abfallquote an normalen Tagen und an Feiertagen: heutige Planung vs. Modell + Feiertagsregel."""
    pick = {"Heutige Planung": sims["Heutige Planung (Ø 4 Wochen + 12 %)"],
            "Modell + Feiertagsregel": sims["Modell + Newsvendor + Feiertagsregel"]}
    groups = ["Normale Tage", "Feiertage"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.36
    for j, (name, sim) in enumerate(pick.items()):
        vals = []
        for is_hol in (0, 1):
            part = sim[sim["is_holiday"] == is_hol]
            vals.append(100 * part["waste"].sum() / part["produce"].sum())
        xs = [i + (j - 0.5) * width for i in range(len(groups))]
        bars = ax.bar(xs, vals, width, color=["#c2410c", "#2563eb"][j], label=name)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.5, f"{v:.1f} %".replace(".", ","), ha="center", fontsize=9)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel("Abfallquote (%)")
    ax.set_title("Abfall an Feiertagen: Die Planung kennt sie nicht", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_markdown(r, path):
    def table(df):
        df = df.reset_index().rename(columns={"index": "Planung"})
        head = "| " + " | ".join(map(str, df.columns)) + " |"
        sep = "|" + "---|" * len(df.columns)
        def fmt(v, col):
            if not isinstance(v, float):
                return str(v)
            if "%" in col:
                return f"{v:.1f}".replace(".", ",")
            return f"{v:,.0f}".replace(",", ".")
        rows = ["| " + " | ".join(fmt(v, c) for v, c in zip(row, df.columns)) + " |"
                for row in df.itertuples(index=False)]
        return "\n".join([head, sep, *rows])

    text = f"""# Ergebnisse (automatisch erzeugt von run.py)

## Genauigkeit der Nachfrageprognose (Testzeitraum, gegen echte Nachfrage)

{table(r.accuracy.set_index("Methode"))}

## Planungsregeln im Vergleich (alle Tage)

{table(r.policies)}

## Nur Feiertage

{table(r.policies_holiday)}

## Ausverkaufs-Korrektur

- Ausverkaufstage gesamt: {r.imputation['ausverkaufstage']}
- Absatz unterschätzt die echte Nachfrage an diesen Tagen um {str(r.imputation['absatz_unterschaetzt_um_%']).replace('.', ',')} %
- Nach Korrektur nur noch um {str(r.imputation['korrektur_unterschaetzt_um_%']).replace('.', ',')} %
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../bakery-sales-db/data",
                        help="Ordner mit den CSV-Dateien aus bakery-sales-db")
    args = parser.parse_args()

    r = pipeline.run(args.data_dir)
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    plot_costs(r.policies, REPORTS / "figures" / "kosten.svg")
    plot_holidays(r.sims, REPORTS / "figures" / "feiertage.svg")
    write_markdown(r, REPORTS / "results.md")
    print((REPORTS / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
