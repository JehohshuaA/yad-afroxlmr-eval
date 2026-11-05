"""
yad_eval.py -- scorer and checks for the YAD-AfroXLMR paper.

Everything that turns prediction files into reported numbers lives here, so the
notebook and the command-line script share exactly one implementation.

Metrics (all corpus-level):
  CER   character edits (Levenshtein over grapheme clusters) / reference clusters
  WER   word edits (Levenshtein over words) / reference words
  DER   reference letters carrying a diacritic whose marks are not reproduced
        exactly at the aligned position / reference letters carrying a diacritic.
        A letter with two marks (e.g. U+1ECD U+0301) is one position.
  WDER  reference words carrying a diacritic that are not reproduced exactly
        at the aligned position / reference words carrying a diacritic.
  BLEU  NLTK corpus BLEU, smoothing method 1
  ChrF  NLTK corpus chrF

Before scoring, U+0329 (combining vertical line below) is folded into U+0323
(combining dot below) in both the reference and the prediction. Both code
points write the Yoruba underdot, and the YAD reference uses each in places.
"""

import csv
import os
import unicodedata
from difflib import SequenceMatcher

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
from rapidfuzz.distance import Levenshtein

GRAVE, ACUTE, DOT_BELOW, VLINE_BELOW = "̀", "́", "̣", "̩"

# The combining marks of Yoruba orthography. Any other combining mark in an
# output is outside the orthography.
MARKS = {GRAVE, ACUTE, DOT_BELOW, VLINE_BELOW}

SYSTEMS = ["B0", "B1", "B2", "B3", "A", "C", "mT5"]
LABELS = {
    "B0": "B0  MENYO-20k only",
    "B1": "B1  + secular",
    "B2": "B2  + Mormon/Qur'an",
    "B3": "B3  + Bible (full)",
    "A": "A   full + Igbo",
    "C": "C   no Phase 1",
    "mT5": "mT5 (greedy)",
}


# --------------------------------------------------------------------------
# File and text helpers
# --------------------------------------------------------------------------
def read_lines(path):
    """Read a file into a list of lines without line endings.

    A missing final newline is allowed; a trailing empty line is dropped so
    that files with and without a final newline give the same list.
    """
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def strip_marks(s):
    """Remove every combining mark, leaving only the base letters."""
    s = unicodedata.normalize("NFD", s)
    return unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))


def fold_underdot(s):
    """Write the underdot one way: U+0329 becomes U+0323."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", s).replace(VLINE_BELOW, DOT_BELOW))


def letters_key(word):
    """A word reduced to lower-case letters and digits, marks removed."""
    return "".join(c for c in strip_marks(word).lower() if c.isalnum())


def word_keys(s):
    return [k for k in (letters_key(w) for w in s.split()) if k]


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------
def clusters(s):
    """Split a string into (base character, frozenset of Yoruba marks) pairs."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if unicodedata.combining(ch) and out:
            out[-1].append(ch)
        else:
            out.append([ch])
    return [(c[0], frozenset(m for m in c[1:] if m in MARKS)) for c in out]


def _align(a, b):
    """Map each index of sequence a to its matched index in sequence b."""
    sm = SequenceMatcher(None, a, b, autojunk=False)
    m = {}
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            m[i + k] = j + k
    return m


def score(gold_lines, pred_lines):
    """All six metrics at corpus level. Lines are compared as given."""
    if len(gold_lines) != len(pred_lines):
        raise ValueError(f"line count mismatch: {len(gold_lines)} gold vs {len(pred_lines)} predicted")

    c_err = c_tot = w_err = w_tot = d_err = d_tot = wd_err = wd_tot = 0
    for g, p in zip(gold_lines, pred_lines):
        gc, pc = clusters(g), clusters(p)
        gw = unicodedata.normalize("NFD", g).split()
        pw = unicodedata.normalize("NFD", p).split()

        c_err += Levenshtein.distance(gc, pc)
        c_tot += len(gc)
        w_err += Levenshtein.distance(gw, pw)
        w_tot += len(gw)

        mc = _align([x[0] for x in gc], [x[0] for x in pc])
        for i, (_, gold_marks) in enumerate(gc):
            if gold_marks:
                d_tot += 1
                j = mc.get(i)
                if j is None or pc[j][1] != gold_marks:
                    d_err += 1

        mw = _align(gw, pw)
        for i, word in enumerate(gw):
            if any(unicodedata.combining(ch) for ch in word):
                wd_tot += 1
                j = mw.get(i)
                if j is None or pw[j] != word:
                    wd_err += 1

    gn = [unicodedata.normalize("NFC", l) for l in gold_lines]
    pn = [unicodedata.normalize("NFC", l) for l in pred_lines]
    bleu = corpus_bleu([[l.split()] for l in gn], [l.split() for l in pn],
                       smoothing_function=SmoothingFunction().method1)
    return {"CER": c_err / c_tot, "WER": w_err / w_tot, "DER": d_err / d_tot,
            "WDER": wd_err / wd_tot, "BLEU": bleu, "ChrF": corpus_chrf(gn, pn)}


def score_folded(gold_lines, pred_lines):
    """Score after folding U+0329 into U+0323 in both files (the paper's setting)."""
    return score([fold_underdot(l.strip()) for l in gold_lines],
                 [fold_underdot(l.strip()) for l in pred_lines])


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def mark_counts(lines):
    """Count every combining mark (any mark, not only Yoruba ones)."""
    counts = {}
    for l in lines:
        for ch in unicodedata.normalize("NFD", l):
            if unicodedata.combining(ch):
                counts[ch] = counts.get(ch, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def preservation(src_lines, pred_lines):
    """Compare each output against the undiacritized INPUT.

    lines unchanged          output with all marks removed equals the input
    lines with words changed the sequence of words (letters and digits only,
                             lower-cased) differs from the input's
    words altered            input words replaced or deleted, by alignment
    non-Yoruba marks         combining marks outside Yoruba orthography
    """
    changed_lines = words = total = foreign = 0
    for s, p in zip(src_lines, pred_lines):
        gk, pk = word_keys(s), word_keys(p)
        total += len(gk)
        if gk != pk:
            changed_lines += 1
            sm = SequenceMatcher(a=gk, b=pk, autojunk=False)
            words += sum(i2 - i1 for t, i1, i2, _, _ in sm.get_opcodes() if t != "equal")
        foreign += sum(1 for ch in unicodedata.normalize("NFD", p)
                       if unicodedata.combining(ch) and ch not in MARKS)
    unchanged = sum(strip_marks(p) == s for p, s in zip(pred_lines, src_lines))
    return {"lines unchanged": unchanged, "lines with words changed": changed_lines,
            "words altered": words, "words total": total,
            "words altered %": round(100 * words / total, 2), "non-Yoruba marks": foreign}


def changed_spans(gold_lines, other_lines, ours_lines):
    """Spans where `other` changed the words of the reference (case and
    punctuation ignored), shown next to our output for the same span."""
    rows = []
    for i, (g, m, o) in enumerate(zip(gold_lines, other_lines, ours_lines)):
        gw, mw, ow = g.split(), m.split(), o.split()
        sm = SequenceMatcher(a=[letters_key(w) for w in gw],
                             b=[letters_key(w) for w in mw], autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            gold_span, other_span = " ".join(gw[i1:i2]), " ".join(mw[j1:j2])
            if letters_key(gold_span) == letters_key(other_span):
                continue
            rows.append({"line": i, "type": tag, "gold": gold_span, "mT5": other_span,
                         "ours (A)": " ".join(ow[i1:i2]) if len(ow) == len(gw) else ""})
    return rows
