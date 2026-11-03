import os
import json
import time
import asyncio
import argparse
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient

def load_json(file: str | Path):
    file = Path(file)
    with file.open("r", encoding="utf-8") as fp:
        return json.load(fp)

def save_json(file: str | Path, data):
    file = Path(file)
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
        fp.write("\n")

async def run(data: dict, id: int, client: AsyncTypeSafeClient):
    start_time = time.perf_counter()
    output: dict = {}
    try:
        result = await client.system_one(**data["input"])
        output = {
            "response_type": str(result.answers["response_type"].choice),
            "feedback_type": str(result.answers["feedback_type"].choice),
        }
    except Exception as exc:
        print(f"Task {id} failed: {exc}.")
        output = {
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
        }
    output["time"] = time.perf_counter() - start_time
    return str(id), output

def parse_argument():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--api-key-name", type=str, default="AI_GATEWAY_API_KEY")
    arg_parser.add_argument("--base-url", type=str, default="https://ai-gateway.vercel.sh/typesafe")
    arg_parser.add_argument("--concurrency", type=int, default=8)
    arg_parser.add_argument("--data-file", type=str,
                            default=str(Path(__file__).resolve().parents[1] / "datasets" / "Personality.json"))
    arg_parser.add_argument("--output-file", type=str, default="jev-outputs.json")
    args = arg_parser.parse_args()
    if args.concurrency < 1:
        arg_parser.error("--concurrency must be greater than 0.")
    return args

async def main():
    args = parse_argument()
    datas = load_json(args.data_file)
    outputs = load_json(args.output_file) if Path(args.output_file).exists() else {}
    semaphore = asyncio.Semaphore(args.concurrency)

    async with AsyncTypeSafeClient(
        api_key=os.environ[args.api_key_name],
        base_url=args.base_url,
    ) as client:
        async def run_one(data: dict, id: int):
            output = outputs.get(str(id))
            if (isinstance(output, dict)
                and {"response_type", "feedback_type", "time"} <= output.keys()
                and output.get("error_type") is None
                and output.get("error_msg") is None
            ):
                return str(id), output
            async with semaphore:
                return await run(data, id, client)

        tasks = [run_one(data, id) for id, data in enumerate(datas)]
        jev_output = dict(await asyncio.gather(*tasks))

    summary = {
        "total_time": sum((output["time"] for output in jev_output.values()), 0.0),
    }
    summary_file = Path(args.output_file).parent / "summary.json"
    save_json(args.output_file, jev_output)
    save_json(summary_file, summary)
    print(f"Saved Jev outputs to {args.output_file} and summary to {summary_file}.")


if __name__ == "__main__":
    asyncio.run(main())
