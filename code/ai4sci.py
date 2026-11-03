"""Loop over `ai4sci_bench.json`, POST each item's `input` to Jev, score, save a log.

    python eval_jev.py --api-key <KEY>                  # full 1800-item run
    python eval_jev.py --limit 20                       # smoke test
    python eval_jev.py --dry-run                        # no requests, random p
    python eval_jev.py --state-variant descriptors      # descriptors, not SMILES
    python eval_jev.py --state-variant smiles_desc --with-criteria
    python eval_jev.py --proxy http://127.0.0.1:7890     # where direct is blocked
    python eval_jev.py --sequential                      # one request at a time

API key: --api-key > $TYPESAFE_API_KEY > secrets.env (this dir, then its parent).

Stdlib only and self-contained: it builds its own requests, so this file plus
`ai4sci_bench.json` can be dropped anywhere - nothing else from the repo needed.

Every item is one `noul` question, so Jev returns a single 0-1 probability.
Decision rule: predicted = p > 0.5.  Confidence = max(p, 1-p), i.e. how far the
answer sits from the coin-flip boundary (direction-agnostic).

Output (into --out-dir, default `logs/`, timestamped so nothing is overwritten):

    jev_eval_<tag>_<variant>_<ts>.jsonl   one record per item: p, predicted, gold,
                                          correct, confidence, latency, tokens
    jev_eval_<tag>_<variant>_<ts>.txt     the human-readable report
"""
from __future__ import (annotations)

import argparse
import json
import os
import queue
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://api.typesafe.ai/v1/systemone"
IN_PRICE_PER_M = 0.042          # $/1M input tokens; output tokens are free
QUESTION_KEY = "answer"
STATE_VARIANTS = ("smiles", "descriptors", "smiles_desc")


# --------------------------------------------------------------------------- #
# key resolution
# --------------------------------------------------------------------------- #
def resolve_key(explicit: str | None) -> str | None:
    """CLI > $TYPESAFE_API_KEY > secrets.env (here, then parent dir)."""
    if explicit:
        return explicit
    env = os.environ.get("TYPESAFE_API_KEY")
    if env:
        return env.strip()
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (here, os.path.dirname(here)):
        path = os.path.join(d, "secrets.env")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                m = re.search(r"^\s*TYPESAFE_API_KEY\s*=\s*(\S+)", f.read(), re.M)
                if m:
                    return m.group(1).strip("\"'")
    return None


# --------------------------------------------------------------------------- #
# client: bounded concurrency + a refill-based rate limiter
# --------------------------------------------------------------------------- #
class Jev:
    def __init__(self, key: str, workers: int = 8, tries: int = 5,
                 url: str = URL, rate_limit: bool = True):
        self.key, self.url, self.tries = key, url, tries
        self.workers = max(1, int(workers))
        if rate_limit:
            self.slots: queue.Queue = queue.Queue(maxsize=self.workers)
            self._fill = threading.Thread(target=self._refill, daemon=True)
            self._fill.start()
        self.lock = threading.Lock()
        self.requests = self.errors = self.retries = 0
        self.input_tokens = self.output_tokens = 0

    def _refill(self):
        while True:
            time.sleep(1.0)
            while self.slots.qsize() < self.workers:
                self.slots.put(1)

    def evaluate(self, body: dict) -> tuple[dict, float]:
        """-> (response, e2e_ms). Retries transient failures, else raises."""
        data = json.dumps(body).encode("utf-8")
        last = None
        for attempt in range(self.tries):
            if hasattr(self, "slots"):                      # rate-limit gate
                self.slots.get()
            req = urllib.request.Request(
                self.url, data=data,
                headers={"Authorization": f"Bearer {self.key}",
                         "Content-Type": "application/json"},
                method="POST")
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8")[:200]
                except Exception:
                    pass
                last = f"HTTP {e.code} {detail}"
                if e.code in (429, 500, 502, 503, 504):
                    with self.lock:
                        self.retries += 1
                    time.sleep(min(1.5 * (2 ** attempt), 30.0))
                    continue
                with self.lock:
                    self.errors += 1
                raise RuntimeError(last)
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:160]}"
                with self.lock:
                    self.retries += 1
                time.sleep(min(1.5 * (2 ** attempt), 30.0))
                continue

            e2e = (time.time() - t0) * 1000
            with self.lock:
                self.requests += 1
                u = out.get("usage") or {}
                self.input_tokens += u.get("input_tokens", 0) or 0
                self.output_tokens += u.get("output_tokens", 0) or 0
            return out, e2e

        with self.lock:
            self.errors += 1
        raise RuntimeError(f"gave up after {self.tries} tries: {last}")

    @property
    def cost(self) -> float:
        return self.input_tokens / 1_000_000 * IN_PRICE_PER_M


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    hit = tot = 0
    for p in pos:
        for n in neg:
            tot += 1
            if p > n:
                hit += 1
            elif p == n:
                hit += 0.5
    return hit / tot


def auroc(pairs: list[tuple[float, bool]]) -> float:
    """P(score(positive) > score(negative)); ties count 0.5."""
    return _auroc([s for s, y in pairs if y], [s for s, y in pairs if not y])


def youden(pairs: list[tuple[float, bool]]) -> tuple[float, float]:
    """-> (max Youden J, accuracy at that threshold)."""
    if not pairs:
        return float("nan"), float("nan")
    best = (-2.0, float("nan"))
    for t in sorted({round(s, 3) for s, _ in pairs}):
        tp = sum(1 for s, y in pairs if y and s > t)
        fn = sum(1 for s, y in pairs if y and s <= t)
        fp = sum(1 for s, y in pairs if not y and s > t)
        tn = sum(1 for s, y in pairs if not y and s <= t)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if tpr - fpr > best[0]:
            best = (tpr - fpr, (tp + tn) / len(pairs))
    return best


# --------------------------------------------------------------------------- #
# one item
# --------------------------------------------------------------------------- #
def build_body(item: dict, variant: str, with_criteria: bool,
               model: str | None) -> dict:
    """Assemble this item's request at the chosen state variant."""
    q = dict(item["input"]["questions"][QUESTION_KEY])
    q.pop("criteria", None)
    if with_criteria and item.get("criteria"):
        q["criteria"] = item["criteria"]
    return {"model": model or item["input"]["model"],
            "state": item["states"][variant],
            "questions": {QUESTION_KEY: q}}


def run_item(item: dict, gw: Jev, body_for, idx: int,
             rng: "random.Random | None") -> dict:
    gold = bool(item["ground_truth"])
    rec = {"idx": idx, "id": item["id"], "task": item["task"],
           "domain": item["domain"], "category": item["category"],
           "gold": gold}

    if rng is not None:                                       # dry-run
        p = round(rng.uniform(0.0, 1.0), 4)
        rec.update(ok=True, p=p, predicted=p > 0.5,
                   correct=(p > 0.5) == gold,
                   confidence=round(max(p, 1 - p), 4),
                   e2e_ms=0.0, input_tokens=0)
        return rec

    try:
        out, e2e = gw.evaluate(body_for(item))
    except Exception as e:
        rec.update(ok=False, error=str(e)[:300], p=None,
                   predicted=None, correct=False)
        return rec

    ans = (out.get("answers") or {}).get(QUESTION_KEY) or {}
    raw = ans.get("noul", ans.get("probability", ans.get("p")))
    p = float(raw) if isinstance(raw, (int, float)) else None
    pred = (p > 0.5) if p is not None else None
    rec.update(ok=p is not None, p=p, predicted=pred,
               correct=(pred == gold) if pred is not None else False,
               confidence=max(p, 1 - p) if p is not None else None,
               e2e_ms=round(e2e, 2),
               input_tokens=(out.get("usage") or {}).get("input_tokens"),
               model_id=out.get("model"))
    if p is None:
        rec["error"] = f"no noul in response: {json.dumps(ans)[:200]}"
    return rec


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _fmt(x: float, w: int = 3) -> str:
    return "   n/a" if x != x else f"{x:.{w}f}"


def report(recs: list[dict], gw: Jev, wall: float, model: str,
           args: "argparse.Namespace | None" = None) -> str:
    ok = [r for r in recs if r.get("ok")]
    bad = [r for r in recs if not r.get("ok")]
    n = len(ok)
    L = []
    A = L.append

    def rows_for(key: str, keys: list[str] | None = None) -> None:
        groups: dict[str, list[bool]] = {}
        for r in ok:
            groups.setdefault(r[key], []).append(r["correct"])
        order = keys or sorted(groups, key=lambda k: -sum(groups[k]) / len(groups[k]))
        for k in order:
            v = groups.get(k)
            if v:
                A(f"  {k:<30s}{len(v):6d}{sum(v) / len(v) * 100:8.1f}%")

    A("=" * 68)
    A("Jev eval · ai4sci_bench.json  (TDC molecule endpoints, noul binary)")
    A("=" * 68)
    A(f"run at           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"model            : {model}")
    if args is not None:
        A(f"bench            : {os.path.basename(args.bench)} "
          f"({os.path.getsize(args.bench) / 1e6:.1f} MB)")
        A(f"state variant    : {args.state_variant}"
          f"{'  (+criteria)' if args.with_criteria else ''}")
        A("mode             : " + ("dry-run (no HTTP)" if args.dry_run else
                                  ("sequential" if args.sequential else
                                   f"{args.workers} workers")))
        A(f"flags            : limit={args.limit or 'all'} every={args.every} "
          f"seed={args.seed} tries={args.tries} tag='{args.tag}'")
    A(f"items            : {len(recs)} attempted, {n} scored, {len(bad)} failed")
    A(f"wall clock       : {wall:.1f}s   (requests {gw.requests}, "
      f"retries {gw.retries}, errors {gw.errors})")
    if ok:
        lat = sorted(r["e2e_ms"] for r in ok)
        A(f"e2e latency      : median {statistics.median(lat):.0f} ms, "
          f"p90 {lat[int(len(lat) * 0.9)]:.0f} ms, mean {statistics.mean(lat):.0f} ms")
    A(f"input tokens     : {gw.input_tokens:,}   cost ~${gw.cost:.2f}")

    # ---- accuracy ---- #
    A("")
    A("-- ACCURACY " + "-" * 56)
    if n == 0:
        A("  no item scored - see FAILURES below")
    else:
        acc = sum(1 for r in ok if r["correct"])
        A(f"  {'overall':<30s}{n:6d}{acc / n * 100:8.1f}%"
          f"   (random baseline 50.0%)")
        A("")
        A(f"  {'by domain':<30s}{'n':>6s}{'acc':>9s}")
        rows_for("domain", keys=["chemistry", "physics"])
        A("")
        A(f"  {'by endpoint (task)':<30s}{'n':>6s}{'acc':>9s}")
        rows_for("task")

    # ---- confusion ---- #
    if n:
        tp = sum(1 for r in ok if r["gold"] and r["correct"])
        tn = sum(1 for r in ok if not r["gold"] and r["correct"])
        fp = sum(1 for r in ok if not r["gold"] and not r["correct"])
        fn = sum(1 for r in ok if r["gold"] and not r["correct"])
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec_ = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec_ / (prec + rec_) if prec and rec_ else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        A("")
        A("-- CONFUSION (positive = the property holds) " + "-" * 23)
        A(f"  TP {tp:5d}   FP {fp:5d}   TN {tn:5d}   FN {fn:5d}")
        A(f"  precision {_fmt(prec)}   recall {_fmt(rec_)}   "
          f"F1 {_fmt(f1)}   specificity {_fmt(spec)}   "
          f"balanced acc {_fmt((rec_ + spec) / 2 if rec_ == rec_ and spec == spec else float('nan'))}")
        pred_pos = sum(1 for r in ok if r["predicted"])
        gold_pos = sum(1 for r in ok if r["gold"])
        A(f"  predicted TRUE {pred_pos}/{n} = {pred_pos / n * 100:.1f}%   "
          f"(gold TRUE {gold_pos / n * 100:.1f}%)")

    # ---- probability quality ---- #
    A("")
    A("-- PROBABILITY QUALITY " + "-" * 45)
    A("  score = the noul probability itself (higher should mean TRUE)")
    pairs = [(r["p"], r["gold"]) for r in ok
             if isinstance(r.get("p"), (int, float))]
    if pairs and len({p for p, _ in pairs}) > 1:
        A(f"  AUROC(p)         : {_fmt(auroc(pairs))}")
        j, acc_t = youden(pairs)
        A(f"  Youden J         : {_fmt(j)}   acc@t* {acc_t * 100:.1f}%  "
          f"(vs {sum(1 for r in ok if r['correct']) / n * 100:.1f}% at the fixed 0.5)")
        pos = [r["p"] for r in ok if r["gold"] and r.get("p") is not None]
        neg = [r["p"] for r in ok if not r["gold"] and r.get("p") is not None]
        if pos and neg:
            A(f"  mean p           : TRUE {statistics.mean(pos):.3f} vs "
              f"FALSE {statistics.mean(neg):.3f}  "
              f"(gap {statistics.mean(pos) - statistics.mean(neg):+.3f})")
    else:
        A("  n/a (too few scored items, or every probability identical)")

    conf_ok = [r for r in ok if isinstance(r.get("confidence"), (int, float))]
    cc = [r["confidence"] for r in conf_ok if r["correct"]]
    cw = [r["confidence"] for r in conf_ok if not r["correct"]]
    if cc and cw:
        A(f"  confidence       : correct {statistics.mean(cc):.3f} vs "
          f"wrong {statistics.mean(cw):.3f}  "
          f"(gap {statistics.mean(cc) - statistics.mean(cw):+.3f})")
    A("  (confidence = max(p, 1-p), i.e. how far from the coin flip)")
    if n:
        for thr in (0.6, 0.7, 0.8, 0.9, 0.95):
            sel = [r for r in conf_ok if r["confidence"] >= thr]
            if sel:
                a = sum(1 for r in sel if r["correct"])
                A(f"    conf>={thr:.2f}      : {len(sel):5d} items "
                  f"({len(sel) / n * 100:5.1f}% kept) acc {a / len(sel) * 100:5.1f}%")

    # ---- per-endpoint signal ---- #
    if n > 20:
        A("")
        A("-- PER-ENDPOINT SIGNAL " + "-" * 46)
        A(f"  {'task':<30s}{'n':>5s}{'acc':>8s}{'AUROC':>8s}")
        out = []
        for t in {r["task"] for r in ok}:
            v = [r["correct"] for r in ok if r["task"] == t]
            pr = [(r["p"], r["gold"]) for r in ok
                  if r["task"] == t and isinstance(r.get("p"), (int, float))]
            out.append((t, len(v), sum(v) / len(v),
                        auroc(pr) if len({p for p, _ in pr}) > 1 else float("nan")))
        for t, cnt, acc_, a in sorted(out, key=lambda x: -(x[3] if x[3] == x[3] else -1)):
            A(f"  {t:<30s}{cnt:5d}{acc_ * 100:7.1f}%{_fmt(a):>8s}")

    if bad:
        A("")
        A("-- FAILURES " + "-" * 57)
        for r in bad[:10]:
            A(f"  {r['id']}: {r.get('error', 'unknown')[:110]}")
        if len(bad) > 10:
            A(f"  ... and {len(bad) - 10} more")
    A("")
    A("=" * 68)
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(here, "ai4sci_bench.json"))
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default=None,
                    help="override the model inside each item's input")
    ap.add_argument("--state-variant", default="smiles",
                    choices=STATE_VARIANTS,
                    help="which rendering to send as `state` (default: SMILES)")
    ap.add_argument("--with-criteria", action="store_true",
                    help="also send each endpoint's true/false criteria wording")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1,
                    help="take every Nth item instead of a prefix - stratified, "
                         "because the file is ordered by task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(here, "logs"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--proxy", default=None,
                    help="e.g. http://127.0.0.1:7890 - needed wherever "
                         "api.typesafe.ai is not reachable directly")
    ap.add_argument("--sequential", action="store_true",
                    help="one request at a time, no concurrency")
    ap.add_argument("--dry-run", action="store_true",
                    help="fake probabilities, no HTTP - validates the pipeline")
    args = ap.parse_args()

    if args.proxy:                      # urllib picks these up automatically
        os.environ["http_proxy"] = os.environ["https_proxy"] = args.proxy

    if not os.path.exists(args.bench):
        print(f"missing {args.bench}", file=sys.stderr)
        return 2
    with open(args.bench, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list) or not items:
        print(f"{args.bench} is empty or not a list", file=sys.stderr)
        return 2
    if args.every > 1:
        items = items[:: args.every]
    if args.limit:
        items = items[: args.limit]

    missing = [it["id"] for it in items
               if args.state_variant not in it.get("states", {})]
    if missing:
        print(f"{len(missing)} items lack the '{args.state_variant}' variant "
              f"(first: {missing[0]})", file=sys.stderr)
        return 2

    def body_for(item: dict) -> dict:
        return build_body(item, args.state_variant, args.with_criteria,
                          args.model)

    def tick(k: int, total: int, done: list[dict]) -> None:
        if k % 100 and k != total:
            return
        good = sum(1 for r in done if r.get("ok"))
        hit = sum(1 for r in done if r.get("correct"))
        print(f"  {k}/{total}  ok={good}  acc so far "
              f"{hit / max(good, 1) * 100:.1f}%", flush=True)

    if args.dry_run:
        rng = random.Random(args.seed)
        gw = Jev("dry-run", rate_limit=False)
        t0 = time.time()
        recs = [run_item(it, None, body_for, i, rng)
                for i, it in enumerate(items)]
        wall = time.time() - t0
    else:
        key = resolve_key(args.api_key)
        if not key:
            print("no API key: use --api-key, or set TYPESAFE_API_KEY, "
                  "or put it in secrets.env", file=sys.stderr)
            return 2
        w = 1 if args.sequential else args.workers
        gw = Jev(key, workers=w, tries=args.tries)
        t0 = time.time()
        total = len(items)
        if args.sequential:
            recs = []
            for k, it in enumerate(items, 1):
                recs.append(run_item(it, gw, body_for, k - 1, None))
                tick(k, total, recs)
        else:
            with ThreadPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(run_item, it, gw, body_for, i, None)
                        for i, it in enumerate(items)]
                recs = []
                for k, fu in enumerate(futs, 1):
                    recs.append(fu.result())
                    tick(k, total, recs)
        wall = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    vtag = f"_{args.state_variant}" if args.state_variant != "smiles" else ""
    if args.with_criteria:
        vtag += "+crit"
    base = f"jev_eval{tag}{vtag}_{ts}"
    jf = os.path.join(args.out_dir, base + ".jsonl")
    tf = os.path.join(args.out_dir, base + ".txt")

    model = args.model or items[0]["input"]["model"]
    text = report(recs, gw, wall, model, args)

    with open(jf, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(tf, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(text)
    print(f"\nwrote {jf}\n      {tf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
