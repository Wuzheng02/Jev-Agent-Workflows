#!/usr/bin/env python3
"""Loop over skill_domain.json, call Jev, report scores and save JSONL logs.

Python 3.10+; standard library only. Run inside this folder:
    python skill_domain.py --validate-only
    python skill_domain.py --live --limit 10
    python skill_domain.py --live --log runs/skill_run1.jsonl

The API key comes from TYPESAFE_API_KEY or a hidden input prompt.
Without --live, only offline validation runs. Existing logs are never overwritten.
Fixed setup: jev-1.13.0, 88 candidates, concurrency 4, retries 0, timeout 30 s.
The summary reports accuracy over valid outputs, accuracy over all scheduled cases,
failure counts, coverage and request latency. --limit marks a diagnostic subset.
"""

# Upstream attribution for the SRA-Bench material included in skill_domain.json.
# https://github.com/oneal2000/SR-Agents/blob/277fd8d2bbd7d3b81a5cf4ffa6e87e18c7906e4f/LICENSE
# This notice concerns upstream material; the evaluation code follows the team repository license.
# MIT License
#
# Copyright (c) 2026 Weihang Su
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import argparse
import concurrent.futures as futures
import errno
import getpass
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
QUESTION = "skill_selection"
INSTRUCTION = "Select the single skill that should be used to solve the task described in the state."
DATA_SHA256 = "b64a083f117e18e5d34e3e75e2bea02af53af5459bc95a14d98a20365a2494e5"
TOTAL = 3290
CONCURRENCY = 4
OPTIONS = [f"option_{i:03d}" for i in range(1, 89)]
FATAL = {"MODEL_IDENTITY_MISMATCH", "LOCAL_INFRASTRUCTURE_DNS_TCP_BLOCKER",
         "RUNNER_INTERNAL_ERROR"}


def load_cases(path: Path) -> list[dict]:
    raw = path.read_bytes()
    # Windows Git may check out JSON with CRLF. This changes no JSON string value.
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset SHA-256 differs from this release; no request sent")
    cases = json.loads(raw.decode("utf-8"))
    if not isinstance(cases, list) or len(cases) != TOTAL:
        raise ValueError("Expected the full 3,290-case JSON list")
    inventory = None
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"input", "ground truth"}:
            raise ValueError(f"Case {index}: expected input and ground truth")
        payload = case["input"]
        if not isinstance(payload, dict) or set(payload) != {"state", "model", "questions"}:
            raise ValueError(f"Case {index}: input must be a complete Jev request")
        if payload["model"] != MODEL or not isinstance(payload["state"], str):
            raise ValueError(f"Case {index}: unexpected model or state")
        questions = payload["questions"]
        if not isinstance(questions, dict) or set(questions) != {QUESTION}:
            raise ValueError(f"Case {index}: expected one skill_selection question")
        q = questions[QUESTION]
        if (not isinstance(q, dict) or set(q) != {"type", "instructions", "criteria"}
                or q["type"] != "choice" or q["instructions"] != INSTRUCTION):
            raise ValueError(f"Case {index}: frozen Choice question differs")
        criteria = q["criteria"]
        if not isinstance(criteria, dict) or list(criteria) != OPTIONS:
            raise ValueError(f"Case {index}: expected ordered option_001 through option_088")
        if any(not isinstance(v, str) or not v.strip() or "\n" not in v for v in criteria.values()):
            raise ValueError(f"Case {index}: invalid name/description")
        values = frozenset(criteria.values())
        if len(values) != 88 or (inventory is not None and values != inventory):
            raise ValueError(f"Case {index}: candidate inventory differs")
        inventory = values
        if not isinstance(case["ground truth"], str) or case["ground truth"] not in criteria:
            raise ValueError(f"Case {index}: gold option is not a candidate")
    return cases


def _local_setup_failure(reason) -> bool:
    return isinstance(reason, socket.gaierror) or (
        isinstance(reason, OSError) and reason.errno in {
            errno.ECONNREFUSED, errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EADDRNOTAVAIL
        }
    )


def call_jev(payload: dict, api_key: str, timeout: float) -> tuple:
    """One HTTP attempt. Never sends the containing case or its gold label."""
    request = Request(ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST")
    started = time.perf_counter()
    status = None
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            body = response.read()
        latency_ms = (time.perf_counter() - started) * 1000
    except HTTPError as exc:
        failure = f"HTTP_{exc.code}"
        return {"_provider_error": failure}, (time.perf_counter() - started) * 1000, exc.code
    except (URLError, OSError, TimeoutError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        if isinstance(reason, (TimeoutError, socket.timeout)):
            failure = "PROVIDER_TIMEOUT"
        elif _local_setup_failure(reason):
            failure = "LOCAL_INFRASTRUCTURE_DNS_TCP_BLOCKER"
        else:
            failure = "PROVIDER_TIMEOUT_OR_TRANSPORT"
        return {"_provider_error": failure}, (time.perf_counter() - started) * 1000, status
    try:
        result = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeError):
        result = {"_provider_error": "RESPONSE_SCHEMA"}
    return result, latency_ms, status


def parse_response(raw: object, options: dict) -> tuple[str, dict]:
    """Use the API Choice field, as in the frozen original Skill runner."""
    if not isinstance(raw, dict):
        raise ValueError("RESPONSE_SCHEMA")
    if raw.get("model") is not None and raw["model"] != MODEL:
        raise ValueError("MODEL_IDENTITY_MISMATCH")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or set(answers) != {QUESTION}:
        raise ValueError("RESPONSE_SCHEMA")
    answer = answers[QUESTION]
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("RESPONSE_SCHEMA")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in options:
        raise ValueError("OPTION_MEMBERSHIP_FAILURE")
    probabilities = answer.get("probabilities")
    if probabilities is not None and (not isinstance(probabilities, dict) or set(probabilities) != set(options)):
        raise ValueError("RESPONSE_SCHEMA")
    usage = raw.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("RESPONSE_SCHEMA")
    for key in ("input_tokens", "output_tokens"):
        if key in usage and (type(usage[key]) is not int or usage[key] < 0):
            raise ValueError("RESPONSE_SCHEMA")
    return choice, {key: usage[key] for key in ("input_tokens", "output_tokens") if key in usage}


def evaluate_case(index: int, case: dict, api_key: str, timeout: float, provider) -> dict:
    payload = case["input"]
    row = {"index": index, "ground truth": case["ground truth"], "success": False,
           "correct": False, "requested_model": MODEL, "attempts": 1,
           "input_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False,
               separators=(",", ":")).encode("utf-8")).hexdigest()}
    try:
        raw, latency, http_status = provider(payload, api_key, timeout)
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise RuntimeError("invalid local latency")
        row.update(latency_ms=latency, http_status=http_status)
        if isinstance(raw, dict) and "_provider_error" in raw:
            row["failure_class"] = raw["_provider_error"]
            return row
        try:
            option, usage = parse_response(raw, payload["questions"][QUESTION]["criteria"])
        except ValueError as exc:
            row["failure_class"] = str(exc)
            return row
        row.update(success=True, predicted_option=option, correct=option == case["ground truth"],
                   returned_model=raw.get("model"), usage=usage)
    except Exception:
        # Raw exception text may contain credentials or provider content.
        row["failure_class"] = "RUNNER_INTERNAL_ERROR"
    return row


def summarize(cases: list[dict], records: list[dict], status: str, wall_seconds: float) -> dict:
    indices = [r["index"] for r in records]
    if len(indices) != len(set(indices)) or any(type(i) is not int or not 0 <= i < len(cases) for i in indices):
        raise ValueError("Duplicate or unexpected result indices")
    if status == "completed" and len(records) != len(cases):
        raise ValueError("Incomplete output cannot receive a completed score")
    correct = sum(r.get("success") is True and r.get("predicted_option") == cases[r["index"]]["ground truth"] for r in records)
    scored = sum(r.get("success") is True for r in records)
    complete = status == "completed"
    latencies = [r["latency_ms"] for r in records if "latency_ms" in r]
    usage = [r.get("usage", {}) for r in records if r.get("success") is True]
    input_tokens = [u["input_tokens"] for u in usage if "input_tokens" in u]
    output_tokens = [u["output_tokens"] for u in usage if "output_tokens" in u]
    return {
        "status": status, "model": MODEL,
        "scope": "FULL_3290" if len(cases) == TOTAL else "DIAGNOSTIC_SUBSET",
        "formal_score_valid": complete and len(cases) == TOTAL,
        "total": len(cases), "processed": len(records), "scored": scored,
        "failed": len(records) - scored, "unprocessed": len(cases) - len(records),
        "correct": correct, "incorrect": scored - correct,
        "accuracy": correct / scored if complete and scored else None,
        "score_percent": 100 * correct / scored if complete and scored else None,
        "scheduled_exact_top1_accuracy": correct / len(cases) if complete else None,
        "coverage": scored / len(cases),
        "failure_classes": dict(Counter(r["failure_class"] for r in records if not r.get("success"))),
        "provider_latency_ms": {"count": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95_nearest_rank": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1] if latencies else None},
        "usage": {"input_tokens": sum(input_tokens) if input_tokens else None,
            "input_token_observations": len(input_tokens),
            "output_tokens": sum(output_tokens) if output_tokens else None,
            "output_token_observations": len(output_tokens)},
        "elapsed_seconds": wall_seconds,
    }


def emit(handle, event: str, **fields) -> None:
    handle.write(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run(cases: list[dict], api_key: str, args, handle, provider=None) -> dict:
    provider = call_jev if provider is None else provider
    started = time.perf_counter()
    records = []
    next_index = 0
    status = "completed"
    emit(handle, "config", started_at=datetime.now(timezone.utc).isoformat(),
         model=MODEL, endpoint=ENDPOINT, data_sha256=DATA_SHA256, total=len(cases),
         data_hash_normalization="CRLF_TO_LF",
         concurrency=CONCURRENCY, retries=0, timeout_seconds=args.timeout)
    with futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        pending = set()
        def fill() -> None:
            nonlocal next_index
            while len(pending) < CONCURRENCY and next_index < len(cases):
                pending.add(pool.submit(evaluate_case, next_index, cases[next_index], api_key, args.timeout, provider))
                next_index += 1
        def record(done) -> None:
            nonlocal status
            for future in done:
                row = future.result()
                records.append(row)
                emit(handle, "case", **row)
                if row.get("failure_class") in FATAL:
                    status = "aborted"
        try:
            fill()
            while pending:
                done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                record(done)
                if status == "completed":
                    fill()
                if len(records) % 100 == 0 or not pending:
                    print(f"[{len(records)}/{len(cases)}] {len(records)/len(cases):.2%}", flush=True)
        except KeyboardInterrupt:
            status = "interrupted"
            # Drain already submitted calls; do not refill or retry.
            if pending:
                done, _ = futures.wait(pending)
                record(done)
            status = "interrupted"
        except Exception:
            status = "aborted"
            # The ledger remains available if local logging or scheduling fails.
            raise
    summary = summarize(cases, records, status, time.perf_counter() - started)
    emit(handle, "summary", **summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "skill_domain.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true", help="Offline validation only")
    modes.add_argument("--live", action="store_true", help="Explicitly enable paid Jev requests")
    parser.add_argument("--limit", type=int, help="First N cases, marked DIAGNOSTIC_SUBSET")
    parser.add_argument("--log", type=Path, help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.limit is not None and not 1 <= args.limit <= TOTAL:
        parser.error("--limit must be between 1 and 3290")
    try:
        cases = load_cases(args.data)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.limit is not None:
        cases = cases[:args.limit]
    if not args.live:
        print(json.dumps({"status": "VALIDATED", "total": len(cases), "full_dataset_total": TOTAL,
            "candidates_per_case": 88, "model": MODEL, "concurrency": CONCURRENCY,
            "data_hash_normalization": "CRLF_TO_LF",
            "retries": 0, "api_requests": 0, "data_sha256": DATA_SHA256}, ensure_ascii=False))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = args.log or HERE / "runs" / f"skill_domain_{stamp}.jsonl"
    if log.resolve() in {args.data.resolve(), Path(__file__).resolve()} or log.exists():
        parser.error("--log must be a fresh file separate from data and script")
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("TypeSafe API key (hidden): ").strip()
    if not api_key:
        parser.error("No API key; no request sent")
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("x", encoding="utf-8", newline="\n") as handle:
            summary = run(cases, api_key, args, handle)
    except OSError:
        print("Local file or logging failure. Run invalid; inspect the partial ledger.", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Log saved to {log}")
    if summary["status"] == "interrupted":
        return 130
    return 0 if summary["status"] == "completed" and summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
