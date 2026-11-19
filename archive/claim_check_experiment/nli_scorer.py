"""
eval/nli_scorer.py — scores (premise, hypothesis) pairs with a BERT-style NLI cross-encoder (ONNX, CPU). Part of eval/claim_check.py.

Reads pairs.jsonl ({key, premise, hypothesis}), writes one JSON line per pair: {key, p: [contradiction, entailment, neutral], supported}.
`supported` is the pre-registered rule: entailment is the argmax. There is no threshold. Needs only onnxruntime + tokenizers; the model
files are plain ONNX/JSON (no pickle, no remote code). Nothing is sent anywhere: run it with the network off if the data is sensitive.

  python eval/nli_scorer.py --model-dir <dir with onnx/model.onnx + tokenizer.json> --pairs pairs.jsonl --out scores_nli.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

LABELS = ("contradiction", "entailment", "neutral")     # config.json id2label of cross-encoder/nli-deberta-v3-base
SMOKE = [("A man is eating pizza.", "A man eats something.", "entailment"),
         ("The sky is blue.", "The sky is not blue.", "contradiction"),
         ("A woman is playing the violin.", "A man is sleeping.", "contradiction"),
         ("The company has offices in Berlin and Paris.", "The company sells cars.", "neutral")]


class Nli:
    def __init__(self, model_dir: Path, threads: int = 0):
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        got = tuple(cfg["id2label"][str(i)] for i in range(3))
        if got != LABELS:
            sys.exit(f"label order differs from what this scorer assumes: {got}")
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_length=512)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]

    def probs(self, batch: list[tuple[str, str]]) -> np.ndarray:
        enc = [self.tok.encode(p, h) for p, h in batch]
        width = max(len(e.ids) for e in enc)
        pad = self.tok.token_to_id("[PAD]") or 0
        ids = np.full((len(enc), width), pad, dtype=np.int64)
        mask = np.zeros((len(enc), width), dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        logits = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    nli = Nli(Path(a.model_dir))
    pr = nli.probs([(p, h) for p, h, _ in SMOKE])
    for (p, h, want), row in zip(SMOKE, pr):
        got = LABELS[int(row.argmax())]
        print(f"smoke: {want:13s} -> {got:13s} {np.round(row, 3).tolist()}")
        if got != want:
            sys.exit("the smoke test failed: tokenization or label order is wrong; nothing was scored")
    pairs = [json.loads(line) for line in Path(a.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i]["premise"]) + len(pairs[i]["hypothesis"]))   # similar lengths per batch
    done, out = 0, Path(a.out)
    with out.open("w", encoding="utf-8") as f:
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            pr = nli.probs([(pairs[i]["premise"], pairs[i]["hypothesis"]) for i in idx])
            for i, row in zip(idx, pr):
                f.write(json.dumps({"key": pairs[i]["key"], "p": [round(float(x), 5) for x in row], "supported": int(row.argmax()) == 1}) + "\n")
            done += len(idx)
            if done % 800 < a.batch:
                print(f"  {done}/{len(pairs)}", flush=True)
    print(f"scored {done} pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
