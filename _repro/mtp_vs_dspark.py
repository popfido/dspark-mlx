"""Head-to-head: Google's official Gemma-4 MTP assistant draft vs the DSpark draft,
BOTH under MLX, BOTH targeting google/gemma-4-12b-it (bf16), on identical prompts.

Fair-comparison controls:
  - same target (gemma-4-12b-it, bf16), same greedy decode, same 128-token budget, same
    per-prompt warmup, same chat template + prompt selection (one math/code/chat dataset).
  - reported at BOTH each drafter's native block (deployment) AND a unified ~7-drafts/round
    depth (the assistant is pushed past its block-4 via prefer_requested_block_size). tau is
    depth-confounded; accept_rate (per-drafted-token quality) + speedup are the headline metrics.
  - --precision {bf16,q8}: the DRAFT is quantized (target stays bf16) to isolate quantization
    effects -- mirrors DSpark's "quantize the draft, not the verify".

Run as separate processes (each loads the 22GB target once -> no double-load), then merge:
    python _repro/mtp_vs_dspark.py --mode mtp    --n 20 --precision bf16
    python _repro/mtp_vs_dspark.py --mode dspark --n 20 --precision bf16
    python _repro/mtp_vs_dspark.py --mode mtp    --n 20 --precision q8
    python _repro/mtp_vs_dspark.py --mode dspark --n 20 --precision q8
    python _repro/mtp_vs_dspark.py --mode merge
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

ROOT = "/Users/Fido/workspace/dspark-mlx"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))
sys.path.insert(0, os.path.join(ROOT, "_repro"))

from eval_accept import DATASETS  # noqa: E402

DATASETS_USED = ["gsm8k", "humaneval", "mtbench"]  # one math, one code, one chat
DOMAIN = {"gsm8k": "math", "humaneval": "code", "mtbench": "chat"}
MAXT = 128


def _out(mode, precision):
    return os.path.join(ROOT, f"_repro/cmp_{mode}_{precision}.json")


def _prompts(n):
    return {d: DATASETS[d](n) for d in DATASETS_USED}


def _quantize_draft_model(draft_model, bits=8, group_size=64):
    """Quantize the assistant draft in place (Linears + Embedding); target stays bf16."""
    import mlx.nn as nn

    def pred(_path, m):
        return (isinstance(m, (nn.Linear, nn.Embedding))
                and getattr(m, "weight", None) is not None
                and m.weight.shape[-1] % group_size == 0)

    nn.quantize(draft_model, group_size=group_size, bits=bits, class_predicate=pred)
    mx.eval(draft_model.parameters())


# --------------------------------------------------------------------------- MTP assistant
UNIFY_BLOCK_TOTAL = 8  # request 8 -> drafts 7, matching DSpark's depth (see --unify)


def _mtp_pass(model, processor, draft_model, draft_kind, n, draft_block_size):
    from mtp_assistant_probe import run as probe_run

    rows = {}
    for d, prompts in _prompts(n).items():
        taus, rates, speeds, drafts = [], [], [], []
        for q in prompts:
            r, _ = probe_run(model, processor, draft_model, draft_kind, q, MAXT, draft_block_size)
            taus.append(r["tau"]); rates.append(r["accept_rate"])
            speeds.append(r["speedup"]); drafts.append(r["drafts"])
        rows[d] = {"domain": DOMAIN[d], "drafts": float(np.mean(drafts)),
                   "tau": float(np.mean(taus)), "accept_rate": float(np.mean(rates)),
                   "speedup": float(np.median(speeds)), "n": len(prompts)}
        print(f"  [mtp blk={draft_block_size or 'native'}] {d:9s} drafts={rows[d]['drafts']:.1f} "
              f"tau={rows[d]['tau']:.2f} acc={rows[d]['accept_rate']:.1f}% x{rows[d]['speedup']:.2f}",
              flush=True)
    return rows


def run_mtp(n, precision):
    from mtp_assistant_probe import _load

    model, processor, draft_model, draft_kind = _load()
    if precision == "q8":
        _quantize_draft_model(draft_model)
    tag = "q8" if precision == "q8" else "bf16"
    native = _mtp_pass(model, processor, draft_model, draft_kind, n, None)
    # Unified depth: push the assistant past its configured block-4 to DSpark's depth (drafts 7).
    draft_model.prefer_requested_block_size = True
    unified = _mtp_pass(model, processor, draft_model, draft_kind, n, UNIFY_BLOCK_TOTAL)
    json.dump({"native": {"drafter": f"MTP assistant ({tag}, native blk-4)", "rows": native},
               "unified": {"drafter": f"MTP assistant ({tag}, depth-matched)", "rows": unified}},
              open(_out("mtp", precision), "w"), indent=2)
    print("wrote", _out("mtp", precision))


# --------------------------------------------------------------------------- DSpark draft
def _encode(tok, text):
    out = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return np.array(out).reshape(-1).astype("int32").tolist()


def run_dspark(n, precision):
    from run_dspark import _load_base, _load_drafter
    from dspark_mlx.events import SummaryEvent, TokenEvent
    from dspark_mlx.loop import generate_eager

    model, tok, adapter = _load_base("gemma4", "bf16")
    drafter = _load_drafter("gemma4", 8 if precision == "q8" else 0)  # draft precision; target stays bf16
    block = drafter.block_size
    tag = "q8" if precision == "q8" else "bf16"

    def base_greedy(ids, k):
        adapter.reset()
        t0 = time.perf_counter()
        s = adapter.prefill(mx.array([ids], dtype=mx.int32))
        t = int(mx.argmax(s.logits[0].astype(mx.float32)).item()); c = 1
        while c < k:
            s = adapter.decode_step(mx.array([t], dtype=mx.int32))
            t = int(mx.argmax(s.logits[0].astype(mx.float32)).item()); c += 1
        return time.perf_counter() - t0

    def dspark(ids, k):
        adapter.reset(); drafter.reset()
        t0 = time.perf_counter()
        ev = list(generate_eager(adapter, drafter, [ids], max_new_tokens=k, eos_id=None))
        sec = time.perf_counter() - t0
        nt = sum(1 for e in ev if isinstance(e, TokenEvent))
        s = next(e for e in ev if isinstance(e, SummaryEvent))
        return nt, sec, s.n_accepted, s.n_drafted

    rows = {}
    for d, prompts in _prompts(n).items():
        taus, rates, speeds = [], [], []
        for q in prompts:
            ids = _encode(tok, q)
            dspark(ids, 8); base_greedy(ids, 8)  # warmup
            nt, ds_sec, n_acc, n_drafted = dspark(ids, MAXT)
            b_sec = base_greedy(ids, nt)
            blocks = n_drafted // block
            taus.append(n_acc / blocks + 1 if blocks else 1.0)
            rates.append(100.0 * n_acc / n_drafted if n_drafted else 0.0)
            speeds.append(b_sec / ds_sec)
        rows[d] = {"domain": DOMAIN[d], "drafts": float(block),
                   "tau": float(np.mean(taus)), "accept_rate": float(np.mean(rates)),
                   "speedup": float(np.median(speeds)), "n": len(prompts)}
        print(f"  [dspark {tag}] {d:9s} drafts={block} tau={rows[d]['tau']:.2f} "
              f"acc={rows[d]['accept_rate']:.1f}% x{rows[d]['speedup']:.2f}", flush=True)
    json.dump({"native": {"drafter": f"DSpark ({tag}, native blk-7)", "rows": rows}},
              open(_out("dspark", precision), "w"), indent=2)
    print("wrote", _out("dspark", precision))


def _table(title, pairs):
    print(f"\n### {title}")
    print(f"{'dataset':10s} {'domain':5s} | {'drafter':30s} {'drafts':>6} {'tau':>5} {'acc%':>6} {'speedup':>8}")
    print("-" * 84)
    for d in DATASETS_USED:
        for src in pairs:
            r = src["rows"][d]
            print(f"{d:10s} {r['domain']:5s} | {src['drafter']:30s} {r['drafts']:>6.1f} "
                  f"{r['tau']:>5.2f} {r['accept_rate']:>5.1f}% {r['speedup']:>7.2f}x")
        print()


def merge():
    for prec in ("bf16", "q8"):
        mtp_p, ds_p = _out("mtp", prec), _out("dspark", prec)
        if not (os.path.exists(mtp_p) and os.path.exists(ds_p)):
            continue
        mtp = json.load(open(mtp_p)); ds = json.load(open(ds_p))
        print(f"\n========== draft precision: {prec} ==========")
        _table(f"[{prec}] Deployment (each at its native block)", [mtp["native"], ds["native"]])
        _table(f"[{prec}] Unified depth (~7 drafts/round each)", [mtp["unified"], ds["native"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mtp", "dspark", "merge"], required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--precision", choices=["bf16", "q8"], default="bf16")
    args = ap.parse_args()
    if args.mode == "merge":
        merge()
    else:
        {"mtp": run_mtp, "dspark": run_dspark}[args.mode](args.n, args.precision)


if __name__ == "__main__":
    main()
