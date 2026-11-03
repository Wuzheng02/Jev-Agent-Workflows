#!/usr/bin/env python3
"""Loop over search_domain.json, call Jev, and write JSONL audit logs.

Python 3.9+; standard library only. Example:
    export TYPESAFE_API_KEY='...'
    python search_domain.py --limit 10
    python search_domain.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
THRESHOLD = 0.5


def load_cases(data: Path) -> list[dict]:
    cases = json.loads(data.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must be a nonempty JSON list")
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"input", "ground truth"}:
            raise ValueError(f"Case {index}: expected input and ground truth keys")
        if type(case["ground truth"]) is not int or case["ground truth"] not in (0, 1):
            raise ValueError(f"Case {index}: ground truth must be integer 0 or 1")
        payload = case["input"]
        if not isinstance(payload, dict) or set(payload) != {"model", "state", "questions"}:
            raise ValueError(f"Case {index}: input must be a complete Jev request body")
        if not isinstance(payload["model"], str) or not payload["model"]:
            raise ValueError(f"Case {index}: invalid model")
        state = payload["state"]
        if not isinstance(state, dict) or not all(
            isinstance(state.get(key), str) for key in ("task_query", "base_llm_answer")
        ):
            raise ValueError(f"Case {index}: missing query or base answer")
        questions = payload["questions"]
        if not isinstance(questions, dict) or set(questions) != {"needs_search"}:
            raise ValueError(f"Case {index}: expected one needs_search question")
        question = questions["needs_search"]
        if not isinstance(question, dict) or question.get("type") != "noul" or not question.get("instructions"):
            raise ValueError(f"Case {index}: expected a noul question with instructions")
    return cases


def emit(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False))


def call_jev(payload: dict, api_key: str, args, logger, index: int) -> tuple:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    started = time.perf_counter()
    for attempt in range(1, args.retries + 2):
        try:
            request = Request(
                args.endpoint,
                data=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=args.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            probability = result["answers"]["needs_search"]["noul"]
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValueError("Jev returned a nonnumeric noul probability")
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Jev returned a noul probability outside [0, 1]")
            return float(probability), result, attempt, time.perf_counter() - started
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise PermissionError(f"Jev authentication failed: HTTP {exc.code}") from exc
            if exc.code not in (408, 429) and not 500 <= exc.code <= 599:
                raise RuntimeError(f"Jev request rejected: HTTP {exc.code}") from exc
            error = f"HTTP {exc.code}"
        except (URLError, OSError, ValueError, KeyError, TypeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt > args.retries:
            raise RuntimeError(f"Request failed after {attempt} attempts: {error}")
        emit(logger, "retry", index=index, attempt=attempt, error=error,
             retry_delay_seconds=args.retry_delay)
        time.sleep(args.retry_delay)
    raise RuntimeError("Unreachable retry state")


def run(cases: list[dict], api_key: str, args, logger) -> dict:
    started = time.perf_counter()
    correct = scored = failed = 0
    confusion = Counter()
    status = "completed"
    emit(logger, "config", started_at=datetime.now(timezone.utc).isoformat(),
         data=str(args.data.resolve()), data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
         total=len(cases), endpoint=args.endpoint, model_override=args.model,
         threshold=THRESHOLD, retries=args.retries, timeout_seconds=args.timeout,
         retry_delay_seconds=args.retry_delay)
    try:
        for index, case in enumerate(cases):
            payload = dict(case["input"])
            if args.model:
                payload["model"] = args.model
            gold = case["ground truth"]
            row = {"index": index, "ground truth": gold, "requested_model": payload["model"],
                   "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True,
                       ensure_ascii=False).encode("utf-8")).hexdigest()}
            try:
                probability, response, attempts, seconds = call_jev(
                    payload, api_key, args, logger, index
                )
            except PermissionError as exc:
                failed += 1
                status = "aborted"
                emit(logger, "case", **row, status="failed", error=str(exc))
                break
            except Exception as exc:
                failed += 1
                emit(logger, "case", **row, status="failed", error=str(exc))
            else:
                prediction = int(probability >= THRESHOLD)
                is_correct = prediction == gold
                scored += 1
                correct += int(is_correct)
                confusion[(gold, prediction)] += 1
                emit(logger, "case", **row, status="ok", predicted_is_search=prediction,
                     probability_yes=probability, correct=is_correct, attempts=attempts,
                     total_seconds=seconds, response=response)
            accuracy = f"{correct / scored:.2%}" if scored else "N/A"
            print(f"[{index + 1}/{len(cases)}] accuracy={accuracy} correct={correct}/{scored} failed={failed}",
                  flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
    summary = {
        "status": status, "total": len(cases), "processed": scored + failed,
        "scored": scored, "failed": failed, "unprocessed": len(cases) - scored - failed,
        "correct": correct, "incorrect": scored - correct,
        "accuracy": correct / scored if scored else None,
        "score_percent": 100 * correct / scored if scored else None,
        "coverage": scored / len(cases),
        "confusion_matrix": {
            f"gold_{gold}_pred_{pred}": confusion[(gold, pred)]
            for gold in (0, 1) for pred in (0, 1)
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    emit(logger, "summary", **summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Log saved to {args.log.resolve()}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "search_domain.json")
    parser.add_argument("--log", type=Path, default=HERE / "search_domain.log",
                        help="JSONL log file; overwritten on each run")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N cases")
    parser.add_argument("--model", default=os.getenv("TYPESAFE_MODEL"),
                        help="Override input.model; otherwise use each case's model")
    parser.add_argument("--endpoint", default=os.getenv(
        "TYPESAFE_ENDPOINT", "https://api.typesafe.ai/v1/systemone"))
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=5, help="Retries after the first attempt")
    parser.add_argument("--retry-delay", type=float, default=10)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate the dataset without making API calls")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.retries < 0 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("--retries and --retry-delay must be nonnegative and finite")
    try:
        cases = load_cases(args.data)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.validate_only:
        counts = Counter(case["ground truth"] for case in cases)
        print(json.dumps({"total": len(cases), "ground_truth_0": counts[0],
                          "ground_truth_1": counts[1]}, ensure_ascii=False))
        return 0
    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("Set TYPESAFE_API_KEY before running the Jev benchmark")
    if args.log.resolve() in (args.data.resolve(), Path(__file__).resolve()):
        parser.error("--log must not overwrite the dataset or script")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("search_domain")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(args.log, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        summary = run(cases, api_key, args, logger)
    finally:
        handler.close()
        logger.removeHandler(handler)
    if summary["status"] == "interrupted":
        return 130
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
