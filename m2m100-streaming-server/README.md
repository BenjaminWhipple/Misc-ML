# M2M100 Streaming Translation Server

Low-latency, concurrent **text-to-text** streaming translation from eleven source languages into
English, using M2M100-418M-compatible checkpoints and CTranslate2.

> **Work in progress:** this repository is not yet a polished production release. The minimal viable
> program is available: model conversion, a concurrent JSONL/TCP server, prefix retranslation,
> provisional and confirmed output, an example client, and unit tests. Authentication, TLS,
> observability integrations, container images, and production deployment recipes remain future work.

The service supports exactly:

`zh`, `es`, `fr`, `pt`, `fa`, `ar`, `ko`, `ja`, `ru`, `uk`, `he` → `en`

It can serve the original [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M)
checkpoint or a merged Transformers checkpoint with the same architecture. No weights are committed
to this repository.

## What it does

Each open stream repeatedly retranslates its newest source prefix. Consecutive hypotheses are compared
at the SentencePiece-token level:

- their longest common prefix becomes irrevocably confirmed after one token is held back to find a safe
  English word boundary;
- the remaining target text is returned as a replaceable provisional tail;
- confirmed target tokens are forced as a prefix on later generations;
- `final` commits the entire remaining hypothesis;
- long utterances use a bounded 64-source-unit rolling window with a 16-unit rollover and up to 256
  target-context tokens.

Chinese and Japanese source units are Unicode characters. The other languages use whitespace-delimited
units. One resident CTranslate2 `Translator` serves all clients. A fair scheduler coalesces superseded
prefixes, permits one in-flight request per stream, waits up to 5 ms for a batch, and dynamically batches
up to eight mixed-source requests. Every item carries its own M2M100 source-language token and all target
prefixes begin with English.

The broad server direction was inspired in part by
[`ufal/SimulStreaming`](https://github.com/ufal/SimulStreaming), which explores simultaneous speech and
text translation. This repository is an independent, smaller text-only implementation built specifically
around M2M100 and CTranslate2; it does not copy or require SimulStreaming code.

## Installation

An NVIDIA GPU is strongly recommended for low latency. The reference machine used an RTX 5060 Ti 16 GB,
CUDA 12.9, Python 3.12, CTranslate2 4.8.5, SentencePiece 0.2.1, and Transformers 5.19.0.

### Conda

The exported serving environment is intentionally small and is named `translate-server`:

```bash
conda env create -f environment.yml
conda activate translate-server
```

If your system uses a different CUDA release, adjust or remove the `cuda-version` line in
`environment.yml`; the installed NVIDIA driver must support the CTranslate2 CUDA runtime.

### venv / pip

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This is deliberately not installed as a Python package. Run the checked-out scripts directly.

Development and evaluation tools are kept separate:

```bash
python -m pip install -r requirements-dev.txt
```

## Convert a model

The server expects a CTranslate2 model. Conversion also copies the tokenizer files so one directory can
be supplied to both `--model` and `--tokenizer`.

Base M2M100-418M, using the lower-memory mixed INT8/FP16 format:

```bash
python scripts/convert_model.py \
  --model facebook/m2m100_418M \
  --output-dir models/m2m100-418m-int8-float16 \
  --quantization int8_float16
```

A local merged fine-tuned checkpoint works the same way:

```bash
python scripts/convert_model.py \
  --model /absolute/path/to/merged-transformers-checkpoint \
  --output-dir models/our-m2m100-checkpoint \
  --quantization int8_float16
```

The checkpoint must be a complete Transformers model. Merge PEFT/LoRA adapters before conversion. Use
`--quantization float16` when you want an FP16 reference export; validate quality before selecting a
quantized model for deployment.

## Run the server

```bash
python translate_server.py \
  --model models/m2m100-418m-int8-float16 \
  --device cuda \
  --compute-type int8_float16 \
  --host 127.0.0.1 \
  --port 43007
```

`--tokenizer` defaults to the model directory. For a split export, pass it explicitly. Use
`python translate_server.py --help` for batching, rolling-context, input-limit, and CPU options.

In another shell:

```bash
python translate_client.py --language es "Buenos" " días" " a todos."
```

The TCP transport is newline-delimited JSON. A client can hold one connection or multiplex distinct
`stream_id` values; stream identifiers must be globally unique while open.

## Protocol

Open a stream with an upstream language-identification result:

```json
{"type":"open","stream_id":"feed-1","utterance_id":"utt-1","source_language":"zh","lid_confidence":0.98,"lid_state":"stable"}
```

Append a delta and finalize it:

```json
{"type":"append","stream_id":"feed-1","text":"新的文本","input_mode":"delta","source_time":1.24}
{"type":"final","stream_id":"feed-1"}
```

Complete snapshots are also accepted with `"input_mode":"snapshot"`. Snapshots may revise only source
text that has not already produced confirmed output or crossed a rolling commit boundary.

An update looks like:

```json
{
  "type": "translation_update",
  "stream_id": "feed-1",
  "utterance_id": "utt-1",
  "revision": 3,
  "source_language": "zh",
  "target_language": "en",
  "confirmed_delta": "Good ",
  "confirmed_text": "Good ",
  "provisional_tail": "morning everyone",
  "full_text": "Good morning everyone",
  "is_final": false,
  "timing_ms": {
    "queue": 4.1,
    "tokenization": 1.8,
    "inference": 51.2,
    "serialization": 0.1,
    "total": 58.7
  }
}
```

`confirmed_text` is append-only. Replace the previously displayed `provisional_tail` with each revision.
On `translation_final`, `full_text` is fully confirmed.

The server does not perform language identification. It buffers input while `lid_state` is `collecting`
or `unknown` and starts translation only after a stable supported label arrives in a `language` event. A
language correction before target commitment cancels the old interpretation and retranslates buffered
source. A correction after commitment is rejected; open a new utterance instead.

## Base M2M100-418M results

These are measurements of the **original M2M100-418M model, not our target-language fine-tuned model**.
They establish a transparent baseline for the server. Quality was measured on our frozen, licensed
English-paired held-out split. Final metrics used 500 examples per language; streaming metrics used the
first 50, so a streaming number should not be interpreted as a paired gain or loss against its final
number. SacreBLEU and chrF++ are corpus scores.

| Source | Final SacreBLEU | Final chrF++ | Streaming chrF++ |
|---|---:|---:|---:|
| Mandarin Chinese (`zh`) | 20.84 | 46.46 | 48.17 |
| Spanish (`es`) | 43.28 | 64.17 | 62.46 |
| French (`fr`) | 37.45 | 60.10 | 60.55 |
| Arabic (`ar`) | 27.40 | 52.73 | 54.21 |

On the RTX 5060 Ti, a short loopback smoke test of the base FP16 export reported:

| Concurrent streams | Updates | Server p50 | Server p95 |
|---:|---:|---:|---:|
| 4 | 40 | 41.7 ms | 53.3 ms |
| 8 | 80 | 59.3 ms | 89.5 ms |

These latency figures exclude network transit and language-identification time. They are indicative, not
a cross-hardware guarantee: chunk length, target length, cold starts, GPU contention, driver/runtime
versions, and quantization all matter. The exact machine-readable summary is in
[`benchmarks/base_m2m100_418m_fp16_summary.json`](benchmarks/base_m2m100_418m_fp16_summary.json).

## Tests

The unit suite uses a deterministic fake backend and therefore does not require model weights or a GPU:

```bash
python -m unittest discover -s tests -v
```

It checks LocalAgreement commitment, provisional replacement, final flushing, mixed-language dynamic
batching, unstable-LID buffering, late language-correction rejection, and the language allowlist.

## Current limitations

- X→English only; English→X and X→Y are not implemented.
- TCP/JSONL only; there is no HTTP or WebSocket adapter yet.
- Language labels must come from an upstream component.
- The process has no built-in authentication, TLS, quotas, or multi-tenant isolation.
- Greedy decoding is intentional for latency; this is not a maximum-quality offline decoder.
- The supplied client is a smoke-test utility, not a reconnecting production SDK.

## License

The server code is available under the MIT License. Models and dependencies retain their own licenses;
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
