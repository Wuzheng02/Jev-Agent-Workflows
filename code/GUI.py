#!/usr/bin/env python3
"""Evaluate GUI.json with Jev and save per-case JSONL logs.

PowerShell example:
    $env:TYPESAFE_API_KEY = "..."
    python GUI.py --limit 5
    python GUI.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset must be a non-empty JSON list")
    for i, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {"input", "ground truth"}:
            raise ValueError(f"case {i}: expected exactly input and ground truth")
        payload, gold = case["input"], case["ground truth"]
        if not isinstance(payload, dict) or set(payload) != {"model", "state", "questions"}:
            raise ValueError(f"case {i}: input is not a complete Jev request")
        target = payload.get("questions", {}).get("target")
        if not isinstance(target, dict) or target.get("type") != "choice":
            raise ValueError(f"case {i}: questions.target must be choice")
        criteria = target.get("criteria")
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"case {i}: target criteria are missing")
        if not isinstance(gold, str) or gold not in criteria:
            raise ValueError(f"case {i}: ground truth is not a candidate id")
    return cases


def emit(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False))


def extract_prediction(response: dict) -> tuple[str, dict | None]:
    """Accept the common Jev choice response shapes while retaining raw output."""
    answer = response.get("answers", {}).get("target")
    if isinstance(answer, str):
        return answer, None
    if not isinstance(answer, dict):
        raise ValueError("response has no answers.target")
    for key in ("choice", "value", "answer", "selected"):
        if key in answer:
            return str(answer[key]), answer.get("probabilities")
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        return str(max(probabilities, key=probabilities.get)), probabilities
    raise ValueError("cannot find selected choice in answers.target")


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
            with urlopen(request, timeout=args.timeout) as reply:
                response = json.loads(reply.read().decode("utf-8"))
            prediction, probabilities = extract_prediction(response)
            return prediction, probabilities, response, attempt, time.perf_counter() - started
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise PermissionError(f"Jev authentication failed: HTTP {exc.code}") from exc
            if exc.code not in (408, 429) and not 500 <= exc.code <= 599:
                raise RuntimeError(f"Jev request rejected: HTTP {exc.code}") from exc
            error = f"HTTP {exc.code}"
        except (URLError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt > args.retries:
            raise RuntimeError(f"request failed after {attempt} attempts: {error}")
        emit(logger, "retry", index=index, attempt=attempt, error=error,
             retry_delay_seconds=args.retry_delay)
        time.sleep(args.retry_delay)
    raise RuntimeError("unreachable")


def run(cases: list[dict], api_key: str, args, logger: logging.Logger) -> dict:
    started = time.perf_counter()
    correct = scored = failed = 0
    errors = Counter()
    status = "completed"
    emit(logger, "config", started_at=datetime.now(timezone.utc).isoformat(),
         data=str(args.data.resolve()),
         data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
         total=len(cases), endpoint=args.endpoint, model_override=args.model,
         retries=args.retries, timeout_seconds=args.timeout)

    try:
        for index, case in enumerate(cases):
            payload = dict(case["input"])
            if args.model:
                payload["model"] = args.model
            gold = case["ground truth"]
            row = {"index": index, "ground truth": gold, "requested_model": payload["model"]}
            try:
                prediction, probabilities, response, attempts, seconds = call_jev(
                    payload, api_key, args, logger, index
                )
                if prediction not in payload["questions"]["target"]["criteria"]:
                    raise ValueError(f"unknown candidate id returned: {prediction!r}")
            except PermissionError as exc:
                failed += 1
                status = "aborted"
                emit(logger, "case", **row, status="failed", error=str(exc))
                break
            except Exception as exc:
                failed += 1
                errors[type(exc).__name__] += 1
                emit(logger, "case", **row, status="failed", error=str(exc))
            else:
                is_correct = prediction == gold
                scored += 1
                correct += int(is_correct)
                emit(logger, "case", **row, status="ok", prediction=prediction,
                     probabilities=probabilities, correct=is_correct, attempts=attempts,
                     total_seconds=seconds, response=response)
            accuracy = f"{correct / scored:.2%}" if scored else "N/A"
            print(f"[{index + 1}/{len(cases)}] accuracy={accuracy} "
                  f"correct={correct}/{scored} failed={failed}", flush=True)
    except KeyboardInterrupt:
        status = "interrupted"

    summary = {
        "status": status,
        "total": len(cases),
        "processed": scored + failed,
        "scored": scored,
        "failed": failed,
        "unprocessed": len(cases) - scored - failed,
        "correct": correct,
        "incorrect": scored - correct,
        "accuracy": correct / scored if scored else None,
        "score_percent": 100 * correct / scored if scored else None,
        "coverage": scored / len(cases),
        "error_types": dict(errors),
        "elapsed_seconds": time.perf_counter() - started,
    }
    emit(logger, "summary", **summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Log saved to {args.log.resolve()}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "GUI.json")
    parser.add_argument("--log", type=Path, default=HERE / "GUI.log",
                        help="JSONL audit log; overwritten each run")
    parser.add_argument("--limit", type=int, help="only evaluate the first N cases")
    parser.add_argument("--model", default=os.getenv("TYPESAFE_MODEL"),
                        help="override input.model")
    parser.add_argument("--endpoint", default=os.getenv(
        "TYPESAFE_ENDPOINT", "https://api.typesafe.ai/v1/systemone"))
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=10)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.timeout <= 0 or args.retries < 0 or args.retry_delay < 0:
        parser.error("timeout must be positive; retries and retry delay must be nonnegative")
    try:
        cases = load_cases(args.data)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.validate_only:
        print(json.dumps({"valid": True, "total": len(cases)}, ensure_ascii=False))
        return 0

    api_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if not api_key:
        parser.error("set TYPESAFE_API_KEY before running the Jev benchmark")
    if args.log.resolve() in (args.data.resolve(), Path(__file__).resolve()):
        parser.error("--log must not overwrite the dataset or script")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("GUI")
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
