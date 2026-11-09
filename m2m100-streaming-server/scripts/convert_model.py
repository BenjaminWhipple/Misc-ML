#!/usr/bin/env python3
"""Convert a base or merged M2M100 Transformers checkpoint to CTranslate2."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


COPY_FILES = (
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
    "vocab.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or merged checkpoint path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        choices=("float16", "int8_float16", "int8", "float32"),
        default="int8_float16",
    )
    parser.add_argument("--force", action="store_true", help="allow conversion into an existing directory")
    args = parser.parse_args()

    converter = shutil.which("ct2-transformers-converter")
    if not converter:
        raise SystemExit("ct2-transformers-converter was not found; install requirements.txt first")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"output directory is not empty: {args.output_dir} (pass --force to overwrite)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        converter,
        "--model", args.model,
        "--output_dir", str(args.output_dir),
        "--quantization", args.quantization,
        "--copy_files", *COPY_FILES,
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, check=True)
    print(f"Converted model and tokenizer files are in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
