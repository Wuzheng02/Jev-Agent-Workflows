#!/usr/bin/env python3
"""Evaluate Embodied.json using Jev; Python 3.9+, standard library only.

Original evaluation protocol: OpenRouter Decisions, typesafe/jev-1.13.
Set OPENROUTER_API_KEY in your environment (never put it in this file).

    python Embodied.py --validate-only
    python Embodied.py --limit 1
    python Embodied.py

Optional TypeSafe endpoint, using the reference package's protocol:
    python Embodied.py --endpoint https://api.typesafe.ai/v1/systemone \
        --model jev-latest --key-env TYPESAFE_API_KEY

Changing endpoint/model is a new run, not reproduction of the historical model.
Logs default to a new timestamped directory. No existing files are overwritten.
Only case['input'] is sent; ground truth stays local.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

HERE = Path(__file__).resolve().parent
ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'


def load_cases(path):
    cases = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('Expected a nonempty JSON list.')
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) != {'input', 'ground truth'}:
            raise ValueError(f'Case {index}: expected input and ground truth only.')
        p = case['input']
        if not isinstance(p, dict) or set(p) != {'model', 'state', 'questions'}:
            raise ValueError(f'Case {index}: input must be a complete request body.')
        if not isinstance(p['model'], str) or not p['model']:
            raise ValueError(f'Case {index}: invalid model.')
        if set(p['questions']) != {'next_action'}:
            raise ValueError(f'Case {index}: expected next_action question.')
        question = p['questions']['next_action']
        if question.get('type') != 'choice' or not question.get('instructions'):
            raise ValueError(f'Case {index}: invalid choice question.')
        opts = p['state']['options']
        criteria = {o['id']: o['action'] for o in opts}
        if len(opts) != 6 or set(criteria) != set('ABCDEF') or criteria != question['criteria']:
            raise ValueError(f'Case {index}: inconsistent six-option criteria.')
        if not isinstance(case['ground truth'], str) or case['ground truth'] not in criteria:
            raise ValueError(f'Case {index}: invalid ground truth.')
    return cases


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call_jev(payload, key, endpoint, timeout):
    request = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                      headers={'Authorization': 'Bearer ' + key,
                               'Content-Type': 'application/json'}, method='POST')
    with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        return json.load(response)


def parse_response(response, payload):
    if not isinstance(response, dict) or 'error' in response:
        raise ValueError('Invalid or error response.')
    answer = response.get('answers', {}).get('next_action', {})
    prediction = answer.get('choice')
    if answer.get('type') != 'choice' or prediction not in payload['questions']['next_action']['criteria']:
        raise ValueError('Missing or invalid next_action choice.')
    probabilities = answer.get('probabilities')
    selected = None
    if isinstance(probabilities, dict):
        value = probabilities.get(prediction)
        if (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1):
            selected = value
    # A valid choice remains scoreable even if optional probabilities are absent.
    return prediction, probabilities, selected, answer.get('confidence')


def emit(handle, event, **fields):
    handle.write(json.dumps({'event': event, **fields}, ensure_ascii=False) + '\n')
    handle.flush()


def evaluate(cases, key, args, handle):
    correct = scored = failed = attempts = 0
    status = 'completed'
    started = time.perf_counter()
    try:
        for index, case in enumerate(cases):
            payload = dict(case['input'])
            if args.model:
                payload['model'] = args.model
            for attempt in range(1, args.retries + 2):
                before = time.perf_counter()
                raw = None
                attempts += 1
                retryable = False
                fatal = False
                try:
                    raw = call_jev(payload, key, args.endpoint, args.timeout)
                    prediction, probabilities, selected, confidence = parse_response(raw, payload)
                except HTTPError as exc:
                    error = f'HTTP {exc.code}'
                    retryable = exc.code in (408, 429) or 500 <= exc.code <= 599
                    fatal = not retryable
                except (URLError, OSError, ValueError, KeyError, TypeError) as exc:
                    error = type(exc).__name__
                    retryable = True
                else:
                    is_correct = prediction == case['ground truth']
                    scored += 1
                    correct += int(is_correct)
                    emit(handle, 'attempt', index=index, attempt=attempt, status='ok',
                         requested_model=payload['model'], prediction=prediction,
                         ground_truth=case['ground truth'], correct=is_correct,
                         probabilities=probabilities, selected_probability=selected,
                         confidence=confidence, response=raw,
                         elapsed_seconds=time.perf_counter() - before)
                    break
                emit(handle, 'attempt', index=index, attempt=attempt, status='error',
                     error=error, response=raw, elapsed_seconds=time.perf_counter() - before)
                if fatal or not retryable or attempt > args.retries:
                    failed += 1
                    if fatal:
                        status = 'aborted'
                    break
                time.sleep(args.retry_delay)
            print(f'[{index+1}/{len(cases)}] correct={correct}/{scored}, failed={failed}', flush=True)
            if status == 'aborted':
                break
    except KeyboardInterrupt:
        status = 'interrupted'
    summary = {'status': status, 'total': len(cases), 'scored': scored, 'failed': failed,
               'unprocessed': len(cases)-scored-failed, 'correct': correct,
               'incorrect': scored-correct, 'accuracy': correct/len(cases),
               'accuracy_scored': correct/scored if scored else None,
               'coverage': scored/len(cases), 'attempt_count': attempts,
               'elapsed_seconds': time.perf_counter()-started,
               'accuracy_definition': 'correct / all selected cases; missing responses count as incorrect'}
    emit(handle, 'summary', **summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=HERE/'Embodied.json')
    parser.add_argument('--out', type=Path, help='New log directory; must not already exist')
    parser.add_argument('--limit', type=int, help='First N cases; default all 1000')
    parser.add_argument('--endpoint', default=ENDPOINT)
    parser.add_argument('--model', help='Optional override of input.model')
    parser.add_argument('--key-env', default='OPENROUTER_API_KEY')
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--retries', type=int, default=0, help='Extra tries for transient failures; default 0')
    parser.add_argument('--retry-delay', type=float, default=5)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    if args.retries < 0 or not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error('Invalid retry settings')
    url = urlsplit(args.endpoint)
    if url.scheme != 'https' or not url.hostname or url.username or url.password:
        parser.error('--endpoint must be HTTPS without embedded credentials')
    try:
        cases = load_cases(args.data)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.limit is not None:
        if args.limit > len(cases):
            parser.error('--limit exceeds dataset size')
        cases = cases[:args.limit]
    if args.validate_only:
        print(json.dumps({'valid_cases': len(cases), 'options_per_case': 6, 'api_calls': 0}))
        return 0
    key = os.environ.get(args.key_env, '').strip()
    if not key:
        parser.error(f'Set {args.key_env} in this terminal first.')
    out = args.out or HERE/'logs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error('--out already exists; use a new directory')
    with (out/'attempts.jsonl').open('w', encoding='utf-8') as handle:
        emit(handle, 'config', data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
             selected_cases=len(cases), endpoint=args.endpoint, model_override=args.model,
             model_in_data=sorted({c['input']['model'] for c in cases}),
             started_at_utc=datetime.now(timezone.utc).isoformat(),
             timeout=args.timeout, retries=args.retries, retry_delay=args.retry_delay,
             index_convention='zero-based position in Embodied.json')
        summary = evaluate(cases, key, args, handle)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Logs: {out.resolve()}')
    return 130 if summary['status'] == 'interrupted' else int(summary['status'] != 'completed' or summary['failed'] > 0)


if __name__ == '__main__':
    sys.exit(main())
