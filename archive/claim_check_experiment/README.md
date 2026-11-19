# Claim-check experiment (archived, negative result)

This experiment ran on 2026-11-18. It is kept here as a record, and as presentation material on what was tried. It is **not** part of the Kaggle runner or the product.

**Question:** does the sentence a claim quotes actually support the claim? If not, can we drop the claim to cut sampling noise on test?

**Verifiers tried, under rules written before any claim was scored** (see the docstring of `claim_check.py`):
- grounding: the quote must appear on the page;
- NLI: `cross-encoder/nli-deberta-v3-base` (ONNX, `nli_scorer.py`);
- Laya, a zero-shot yes/no decision model (`laya_scorer.py`).

**Result:** all of them failed the pre-registered rules.
- They rejected correct claims on dev, where any change counts as a regression.
- Laya accepted claims at very different rates on dev and on test, which is a sign that test phrasings differ from train.

No filter was adopted, and nothing was submitted from it.

**Files:**
- `claim_check.py`: exports (quote, hypothesis) pairs and judges the filters.
- `nli_scorer.py`: the ONNX NLI scorer.
- `laya_scorer.py`: the Laya side experiment.

**Running it:** the files moved here from `eval/` on 2026-11-19, and `claim_check.py`'s import paths were adjusted so it still finds the `eval/` modules. The other two take explicit paths. The commands in the docstrings still say `eval/…`; replace that with `archive/claim_check_experiment/…`. They need the gitignored `data_kaggle/` and the cached extractions. The NLI scorer also needs `onnxruntime` and the model files, which are not in `requirements.txt`.
