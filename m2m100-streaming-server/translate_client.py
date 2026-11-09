#!/usr/bin/env python3
"""Small JSONL/TCP client for smoke-testing the translation server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def run(host: str, port: int, language: str, chunks: list[str]) -> None:
    reader, writer = await asyncio.open_connection(host, port)

    async def send(row: dict) -> dict:
        writer.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        await writer.drain()
        while True:
            response = json.loads(await reader.readline())
            print(json.dumps(response, ensure_ascii=False))
            if response.get("type") in {"opened", "translation_update", "translation_final", "error"}:
                return response

    stream_id = "example-1"
    await send({
        "type": "open", "stream_id": stream_id, "utterance_id": "example-utterance",
        "source_language": language, "lid_state": "stable", "lid_confidence": 1.0,
    })
    for index, chunk in enumerate(chunks):
        await send({
            "type": "append", "stream_id": stream_id, "text": chunk,
            "input_mode": "delta", "source_time": index / 10,
        })
    await send({"type": "final", "stream_id": stream_id})
    writer.close()
    await writer.wait_closed()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument("--language", default="es")
    parser.add_argument("chunks", nargs="*", default=["Buenos", " días", " a todos."])
    args = parser.parse_args()
    try:
        asyncio.run(run(args.host, args.port, args.language, args.chunks))
    except (ConnectionError, OSError) as exc:
        print(f"client error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
