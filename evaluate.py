#!/usr/bin/env python3
"""
Reproduce the accuracy and preservation results of the YAD-AfroXLMR paper
from the released prediction files. CPU only; runs in under a minute.

    python evaluate.py                 # writes CSVs to ./results
    python evaluate.py --out my_dir

Outputs:
    final_results.csv        CER, WER, DER, WDER, BLEU, ChrF per system
    data_vs_architecture.csv data gain (B0->B3), architecture spread (A/B3/C), ratio
    preservation.csv         base-letter preservation against the input
    mark_counts.csv          every combining mark in the gold and each output
    mt5_changed_words.csv    spans where mT5 changed the words of the reference
"""

import argparse
import os
import unicodedata

from yad_eval import (LABELS, MARKS, SYSTEMS, changed_spans, mark_counts,
                      preservation, read_lines, save_csv, score_folded)

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "unyo.txt")    # undiacritized input
GOLD = os.path.join(ROOT, "data", "dcyo.txt")   # diacritized reference
PRED_DIR = os.path.join(ROOT, "predictions")
N_LINES = 3330


def load():
    src = [l.strip() for l in read_lines(SRC)]
    gold = read_lines(GOLD)
    preds = {s: read_lines(os.path.join(PRED_DIR, f"{s}.txt")) for s in SYSTEMS}
    assert len(src) == len(gold) == N_LINES, f"expected {N_LINES} lines in source and gold"
    for s, p in preds.items():
        assert len(p) == N_LINES, f"{s}: {len(p)} lines, expected {N_LINES}"
    return src, gold, preds


def results_table(gold, preds):
    final, rows = {}, []
    print(f"{'System':<22}{'CER':>8}{'WER':>8}{'DER':>8}{'WDER':>8}{'BLEU':>8}{'ChrF':>8}")
    print("-" * 70)
    for s in SYSTEMS:
        r = score_folded(gold, preds[s])
        final[s] = r
        rows.append({"system": s, **{k: round(v, 4) for k, v in r.items()}})
        print(f"{LABELS[s]:<22}" + "".join(f"{r[k]:>8.4f}" for k in r))
    bad = [s for s, r in final.items() if r["CER"] >= r["WER"]]
    print("\nCER < WER on every row." if not bad else f"\nCER >= WER for: {bad}  (check these)")
    return final, rows


def data_vs_architecture(final):
    gain = final["B0"]["DER"] - final["B3"]["DER"]
    arch = {k: final[k]["DER"] for k in ("A", "B3", "C")}
    spread = max(arch.values()) - min(arch.values())
    best = min(arch, key=arch.get)
    print(f"Data gain (B0 -> B3):          {100 * gain:.2f} DER points")
    print(f"Architecture spread (A/B3/C):  {100 * spread:.2f} DER points")
    print(f"Ratio:                         {gain / spread:.1f}x")
    print(f"Best configuration ({best}) vs mT5: DER {100 * arch[best]:.2f}% vs "
          f"{100 * final['mT5']['DER']:.2f}% (gap {100 * (arch[best] - final['mT5']['DER']):.2f} points)")
    return [{"data gain (pts)": round(100 * gain, 2), "architecture spread (pts)": round(100 * spread, 2),
             "ratio": round(gain / spread, 1), "best config": best,
             "best config DER (%)": round(100 * arch[best], 2),
             "mT5 DER (%)": round(100 * final["mT5"]["DER"], 2)}]


def preservation_table(src, preds):
    rows = []
    print(f"{'System':<6}{'unchanged':>11}{'words changed':>15}{'words altered':>22}{'non-Yoruba':>12}")
    for s in SYSTEMS:
        r = preservation(src, preds[s])
        rows.append({"system": s, **r})
        print(f"{s:<6}{r['lines unchanged']:>11,}{r['lines with words changed']:>15,}"
              f"{r['words altered']:>12,} ({r['words altered %']:.2f}%){r['non-Yoruba marks']:>12}")
    print(f"(lines: {N_LINES:,}; input words: {rows[0]['words total']:,})")
    return rows


def marks_table(gold, preds):
    rows = []
    for name, lines in [("gold", gold)] + [(s, preds[s]) for s in SYSTEMS]:
        c = mark_counts(lines)
        foreign = {ch: n for ch, n in c.items() if ch not in MARKS}
        detail = ", ".join(f"U+{ord(ch):04X} {unicodedata.name(ch, '?').title()} x{n}"
                           for ch, n in foreign.items())
        rows.append({"file": name, "U+0300": c.get("̀", 0), "U+0301": c.get("́", 0),
                     "U+0323": c.get("̣", 0), "U+0329": c.get("̩", 0),
                     "non-Yoruba marks": sum(foreign.values()), "non-Yoruba detail": detail})
    print(f"{'File':<6}{'grave':>8}{'acute':>8}{'U+0323':>8}{'U+0329':>8}{'other':>7}  detail")
    for r in rows:
        print(f"{r['file']:<6}{r['U+0300']:>8}{r['U+0301']:>8}{r['U+0323']:>8}{r['U+0329']:>8}"
              f"{r['non-Yoruba marks']:>7}  {r['non-Yoruba detail']}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"), help="folder for the CSV files")
    out = ap.parse_args().out

    src, gold, preds = load()

    print("== Results (U+0329 folded into U+0323) ==\n")
    final, rows = results_table(gold, preds)
    save_csv(rows, os.path.join(out, "final_results.csv"))

    print("\n== Data versus architecture ==\n")
    save_csv(data_vs_architecture(final), os.path.join(out, "data_vs_architecture.csv"))

    print("\n== Base-letter preservation (against the input) ==\n")
    save_csv(preservation_table(src, preds), os.path.join(out, "preservation.csv"))

    print("\n== Combining marks per file ==\n")
    save_csv(marks_table(gold, preds), os.path.join(out, "mark_counts.csv"))

    spans = changed_spans(gold, preds["mT5"], preds["A"])
    save_csv(spans, os.path.join(out, "mt5_changed_words.csv"))
    print(f"\nmT5 changed spans: {len(spans):,} (saved to mt5_changed_words.csv)")
    print(f"\nCSV files written to {out}")


if __name__ == "__main__":
    main()
