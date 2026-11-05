# YAD-AfroXLMR: evaluation code, reference and predictions

Evaluation code and system outputs for **YAD-AfroXLMR**, a token-classification model for Yorùbá automatic diacritic restoration built on AfroXLMR-base. The repository reproduces every accuracy and output-preservation number in the paper from the released prediction files, on a CPU, in under a minute.

| What | Reproducible here | Needs |
|---|---|---|
| Results table (CER, WER, DER, WDER, BLEU, ChrF) | yes | CPU |
| Data-versus-architecture comparison | yes | CPU |
| Base-letter preservation, non-Yorùbá marks, mT5 word changes | yes | CPU |
| Truncation check (sentences longer than the input limit) | yes | CPU, public tokenizer |
| Windowed inference, timing, memory, parameter counts | code and saved outputs only | GPU and the checkpoints, which will be released after the paper is published |

## Quick start

```bash
git clone https://github.com/JehohshuaA/yad-afroxlmr-eval.git
cd yad-afroxlmr-eval
pip install -r requirements.txt
python evaluate.py          # prints the tables and writes CSVs to results/
```

Or open `evaluate.ipynb`, which runs the same code step by step with its outputs saved. It also runs on Colab: the first cell clones the repository.

## Repository layout

```
├── evaluate.py            # command-line reproduction of the results (CPU)
├── evaluate.ipynb         # the same, as a notebook with saved outputs (CPU)
├── efficiency.ipynb       # truncation check, windowing, timing and memory (GPU)
├── yad_eval.py            # the scorer and checks, shared by all of the above
├── data/
│   ├── unyo.txt           # YAD test set, undiacritized input (3,330 lines)
│   └── dcyo.txt           # YAD test set, diacritized reference (3,330 lines)
├── predictions/           # one output line per test line, same order
│   ├── B0.txt  B1.txt  B2.txt  B3.txt   # augmentation tiers
│   ├── A.txt   C.txt                    # architecture configurations
│   └── mT5.txt                          # mT5_base_yoruba_adr, greedy decoding
├── results/               # CSVs written by evaluate.py, and the length-scaling figure
├── checksums.txt          # SHA-256 of every data and prediction file
├── requirements.txt       # scoring
└── requirements-gpu.txt   # additionally, for efficiency.ipynb
```

### Systems

| File | System |
|---|---|
| `B0` | Two-phase fine-tuning on MENYO-20k only |
| `B1` | B0 data + secular text |
| `B2` | B1 data + Book of Mormon and Qur'an |
| `B3` | B2 data + full Bible |
| `A` | B3 data, with Igbo cross-lingual warm-up before Phase 1 |
| `C` | B3 data, without Phase 1 |
| `mT5` | `Davlan/mT5_base_yoruba_adr`, greedy decoding, maximum length 256 |

## How the scoring works

All metrics are computed at **corpus level**: errors and totals are summed over the whole test set, not averaged per sentence.

- **CER**: Levenshtein edits between grapheme clusters (a base character plus its marks), divided by the number of reference clusters.
- **WER**: Levenshtein edits between words, divided by the number of reference words.
- **DER**: reference letters that carry a diacritic are aligned with the prediction by base letter. A position is wrong if the aligned letter is missing or its marks differ. A letter with two marks, such as ọ́, is one position. The denominator is the number of marked reference letters.
- **WDER**: reference words that carry a diacritic, aligned by word, must be reproduced exactly.
- **BLEU / ChrF**: NLTK `corpus_bleu` (smoothing method 1) and `corpus_chrf`.

**Underdot fold.** Yorùbá's underdot can be written as U+0323 (combining dot below) or U+0329 (combining vertical line below). The YAD reference uses U+0329 on 160 of 3,330 lines (1,195 marks). Before scoring, U+0329 is replaced with U+0323 in both the reference and every prediction, so a correctly placed underdot is not counted as wrong because of its encoding.

**Windowing.** The models take at most 256 tokens, which is roughly 254 characters, since the input is split into characters. 72 test sentences (2.16%) are longer than that, and they hold 1,141 of the 119,533 reference diacritic positions (0.95%). Every long sentence was split at spaces into windows of at most 254 tokens, each window was labelled, and the windows were joined back together. mT5 was given **the same windows** (cut using AfroXLMR token counts), so both systems see identical pieces of text. All released predictions use this windowed setting.

## Results

Running `python evaluate.py` gives:

| System | CER | WER | DER | WDER | BLEU | ChrF |
|---|---|---|---|---|---|---|
| B0 | 0.0617 | 0.2260 | 0.1400 | 0.2305 | 0.5502 | 0.7718 |
| B1 | 0.0479 | 0.1808 | 0.1088 | 0.1851 | 0.6284 | 0.8172 |
| B2 | 0.0445 | 0.1696 | 0.1014 | 0.1742 | 0.6473 | 0.8279 |
| B3 | 0.0424 | 0.1583 | 0.0997 | 0.1647 | 0.6678 | 0.8338 |
| A | **0.0410** | 0.1537 | 0.0956 | 0.1591 | 0.6760 | 0.8397 |
| C | 0.0418 | 0.1565 | 0.0973 | 0.1622 | 0.6717 | 0.8365 |
| mT5 (greedy) | 0.0462 | **0.1423** | **0.0796** | **0.1371** | **0.7042** | **0.8542** |

**Data versus architecture.** Adding training data (B0 → B3) lowers DER by 4.03 points. The three configurations trained on the full data (A, B3, C) span 0.41 points. The ratio is 9.7×.

**Base-letter preservation**, checked against the undiacritized input:

| System | Lines unchanged | Lines with words changed | Words altered | Non-Yorùbá marks |
|---|---|---|---|---|
| B0, B1, B2, B3, A, C (each) | 3,330 / 3,330 | 0 | 0 (0.00%) | 0 |
| mT5 | 1,954 / 3,330 | 826 | 1,311 of 76,122 (1.72%) | 21 |

Token classification can only add marks to characters already present, so the six configurations never change a letter. `results/mt5_changed_words.csv` lists every span where mT5 changed a word, next to the reference and Configuration A's output.

## Efficiency (`efficiency.ipynb`)

The GPU notebook covers the truncation check, windowed inference for all systems, parameter counts (277,457,670 for AfroXLMR and 582,401,280 for mT5), and speed and memory against sentence length.

![Inference time and working memory against sentence length](results/length_scaling.png)

On an NVIDIA A100-SXM4-40GB, with batch size 1 and forty sentences per length group:

| Length (characters) | Ours (s/sentence) | mT5 (s/sentence) | Ratio | mT5 decoder passes | Ours peak MB | mT5 peak MB |
|---|---|---|---|---|---|---|
| < 50 | 0.0096 | 0.500 | 52× | 23.3 | 1.0 | 5.6 |
| 50–100 | 0.0099 | 0.987 | 100× | 46.7 | 1.9 | 8.2 |
| 100–200 | 0.0101 | 1.932 | 192× | 92.3 | 4.4 | 13.3 |
| 200+ | 0.0104 | 3.266 | 313× | 156.3 | 6.7 | 20.6 |

**Full test set timings.** The paper's full-test timings (ours 29.6 s at batch size 1 in FP32 and 5.0 s at batch size 16 in FP16; mT5 4,727.4 s and 1,361.7 s in the same settings) were measured together in a **separate session on the same GPU model**. They give the 160× and 272× speed-ups. `efficiency.ipynb` re-times our model at 30.9 s and 4.7 s. Its mT5 full-test runs (about 79 and 23 minutes) are included but switched off by default. All timings use single-window inference.

**Checkpoints.** Part B of the notebook loads the six configurations from the Hugging Face repositories `Jehohshua/YAD-AfroXLMR-{A,B0,B1,B2,B3,C}`. These are **not yet publicly released**; they will be made public after the paper is published. Until then, Part B runs only for an account with access. The saved outputs record the run that produced the paper's numbers. Part A (the truncation check) needs only the public `Davlan/afro-xlmr-base` tokenizer.

## Notes on the data

- `data/unyo.txt` and `data/dcyo.txt` are the YAD test set, unchanged from the original release.
- The reference contains three combining marks outside Yorùbá orthography: two U+0308 (diaeresis) and one U+0328 (ogonek). They are part of the published test set and are left as they are. The scorer ignores marks outside the orthography when comparing diacritics.
- `checksums.txt` lists the SHA-256 of every data and prediction file. To verify them, run `sha256sum -c checksums.txt`.

## Licences

- **Code** (`*.py`, `*.ipynb`): MIT, see [`LICENSE`](LICENSE).
- **Data and predictions** (`data/`, `predictions/`): CC BY-NC 4.0, see [`LICENSE-DATA`](LICENSE-DATA). The YAD test set is derived from MENYO-20k, which is released under CC BY-NC 4.0, and the predictions are line-by-line outputs on that test set.

## Citation

The paper describing YAD-AfroXLMR is currently under review; citation details will be added on publication. Meanwhile, see [`CITATION.cff`](CITATION.cff) to cite this repository.

If you use the test set, please also cite YAD and MENYO-20k:

```bibtex
@article{olawole2024yad,
  title   = {{YAD}: Leveraging {T5} for Improved Automatic Diacritization of {Y}or{\`u}b{\'a} Text},
  author  = {Olawole, Akindele Michael and Alabi, Jesujoba O. and Sakpere, Aderonke Busayo and Adelani, David I.},
  journal = {arXiv preprint arXiv:2412.20218},
  year    = {2024}
}

@inproceedings{adelani2021menyo,
  title     = {The Effect of Domain and Diacritics in {Y}oruba--{E}nglish Neural Machine Translation},
  author    = {Adelani, David and Ruiter, Dana and Alabi, Jesujoba and Adebonojo, Damilola and Ayeni, Adesina and Adeyemi, Mofe and Awokoya, Ayodele Esther and Espa{\~n}a-Bonet, Cristina},
  booktitle = {Proceedings of Machine Translation Summit XVIII: Research Track},
  pages     = {61--75},
  year      = {2021}
}
```

## Contact

Joshua Ayoola, University of Johannesburg. Questions and issues are welcome through GitHub Issues.
