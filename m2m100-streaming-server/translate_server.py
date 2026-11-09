#!/usr/bin/env python3
"""Concurrent JSONL/TCP server for CTranslate2 M2M100 streaming translation."""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import json
import logging
import math
import queue
import threading
import time
from pathlib import Path
from typing import Awaitable, Callable, Protocol, Sequence

TARGET_LANGUAGE = "en"
SOURCE_LANGUAGES = ("zh", "es", "fr", "pt", "fa", "ar", "ko", "ja", "ru", "uk", "he")
NON_SPACED_LANGUAGES = frozenset({"zh", "ja"})


def validate_source_language(language: object) -> str:
    if not isinstance(language, str) or language not in SOURCE_LANGUAGES:
        supported = ", ".join(SOURCE_LANGUAGES)
        raise ValueError(f"source_language must be one of: {supported}")
    return language


Emit = Callable[[dict], Awaitable[None]]


def source_units(text: str, language: str) -> list[str]:
    if language in NON_SPACED_LANGUAGES:
        return [character for character in text.strip() if not character.isspace()]
    return text.strip().split()


def join_source_units(units: Sequence[str], language: str) -> str:
    return ("" if language in NON_SPACED_LANGUAGES else " ").join(units)


def append_text(existing: str, delta: str) -> str:
    if not delta:
        return existing
    if not existing:
        return delta.lstrip()
    if existing[-1].isspace() or delta[0].isspace():
        return existing + delta
    return existing + " " + delta


def common_prefix(left: Sequence, right: Sequence) -> int:
    length = 0
    for old, new in zip(left, right):
        if old != new:
            break
        length += 1
    return length


@dataclasses.dataclass(frozen=True)
class ServiceConfig:
    batch_wait_ms: float = 5.0
    max_batch_size: int = 8
    rolling_window_units: int = 64
    rolling_stride_units: int = 16
    rolling_target_tokens: int = 256
    max_source_tokens: int = 256
    max_new_tokens: int = 128
    target_length_ratio: float = 1.6
    holdback_tokens: int = 1
    max_source_characters: int = 32_768

    def validate(self) -> None:
        if self.batch_wait_ms < 0 or self.max_batch_size < 1:
            raise ValueError("invalid batching configuration")
        if self.rolling_stride_units < 1 or self.rolling_window_units < 2 * self.rolling_stride_units:
            raise ValueError("rolling window must be at least twice the positive stride")
        if self.rolling_target_tokens < 16 or self.max_source_tokens < 8 or self.max_new_tokens < 1:
            raise ValueError("token limits are too small")
        if self.target_length_ratio <= 0 or self.holdback_tokens < 0:
            raise ValueError("invalid target ratio or holdback")


@dataclasses.dataclass(frozen=True)
class BackendRequest:
    source_language: str
    source_text: str
    forced_prefix_ids: tuple[int, ...]
    max_target_tokens: int


@dataclasses.dataclass(frozen=True)
class BackendResult:
    token_ids: tuple[int, ...]
    source_tokens: int
    inference_ms: float
    tokenization_ms: float = 0.0


class TranslationBackend(Protocol):
    def translate_batch(self, requests: Sequence[BackendRequest]) -> list[BackendResult]: ...
    def decode(self, token_ids: Sequence[int]) -> str: ...
    def token_pieces(self, token_ids: Sequence[int]) -> list[str]: ...


class _BackendThread:
    """Run blocking CTranslate2 calls without blocking the asyncio transport loop."""

    def __init__(self, backend: TranslationBackend):
        self.backend = backend
        self.commands: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="m2m100-ct2-runtime", daemon=True)
        self.started = False

    def start(self) -> None:
        if not self.started:
            self.thread.start()
            self.started = True

    async def translate(self, requests: Sequence[BackendRequest]) -> list[BackendResult]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.commands.put((requests, loop, future))
        return await future

    def _run(self) -> None:
        while True:
            command = self.commands.get()
            if command is None:
                return
            requests, loop, future = command
            try:
                result = self.backend.translate_batch(requests)
            except BaseException as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                loop.call_soon_threadsafe(future.set_result, result)

    def close(self) -> None:
        if self.started:
            self.commands.put(None)
            self.thread.join()
            self.started = False


class CT2Backend:
    """One resident translator plus an immutable tokenizer per source language."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        *,
        device: str = "cuda",
        device_index: int = 0,
        compute_type: str = "auto",
        max_source_tokens: int = 256,
    ) -> None:
        try:
            import ctranslate2
            from transformers import M2M100Tokenizer
        except ImportError as exc:
            raise RuntimeError("serving requires ctranslate2, transformers, and sentencepiece") from exc
        self.translator = ctranslate2.Translator(
            str(model_path),
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            inter_threads=1,
        )
        self.tokenizers = {
            language: M2M100Tokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                src_lang=language,
                tgt_lang=TARGET_LANGUAGE,
            )
            for language in SOURCE_LANGUAGES
        }
        self.target_tokenizer = M2M100Tokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            src_lang="en",
            tgt_lang=TARGET_LANGUAGE,
        )
        self.target_language_token = self.target_tokenizer.lang_code_to_token[TARGET_LANGUAGE]
        self.special_ids = set(self.target_tokenizer.all_special_ids)
        self.max_source_tokens = max_source_tokens

    def _source_tokens(self, language: str, text: str) -> list[str]:
        tokenizer = self.tokenizers[language]
        ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_source_tokens,
        )
        return tokenizer.convert_ids_to_tokens(ids)

    def translate_batch(self, requests: Sequence[BackendRequest]) -> list[BackendResult]:
        if not requests:
            return []
        source_tokens = []
        tokenization_ms = []
        for request in requests:
            started = time.perf_counter()
            source_tokens.append(self._source_tokens(request.source_language, request.source_text))
            tokenization_ms.append((time.perf_counter() - started) * 1000.0)
        prefixes = [
            [self.target_language_token]
            + self.target_tokenizer.convert_ids_to_tokens(list(request.forced_prefix_ids))
            for request in requests
        ]
        max_decoding_length = max(
            request.max_target_tokens + len(request.forced_prefix_ids) + 2 for request in requests
        )
        started = time.perf_counter()
        translated = self.translator.translate_batch(
            source_tokens,
            target_prefix=prefixes,
            beam_size=1,
            max_decoding_length=max_decoding_length,
            return_scores=False,
            disable_unk=True,
            use_vmap=True,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output = []
        for source, result, tokenize_ms in zip(source_tokens, translated, tokenization_ms, strict=True):
            tokens = list(result.hypotheses[0])
            if tokens and tokens[0] == self.target_language_token:
                tokens.pop(0)
            ids = self.target_tokenizer.convert_tokens_to_ids(tokens)
            content = tuple(token for token in ids if token not in self.special_ids)
            output.append(BackendResult(content, len(source), elapsed_ms, tokenize_ms))
        return output

    def decode(self, token_ids: Sequence[int]) -> str:
        if not token_ids:
            return ""
        return self.target_tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def token_pieces(self, token_ids: Sequence[int]) -> list[str]:
        return [str(piece) for piece in self.target_tokenizer.convert_ids_to_tokens(list(token_ids))]


@dataclasses.dataclass(frozen=True)
class InferenceJob:
    stream_id: str
    version: int
    received_at: float
    request: BackendRequest
    active_units: tuple[str, ...]
    context_target_tokens: tuple[int, ...]
    rollover: bool
    is_final: bool
    source_time: float | None


class StreamState:
    def __init__(
        self,
        stream_id: str,
        utterance_id: str,
        language: str,
        lid_confidence: float | None,
        lid_state: str,
        emit: Emit,
        config: ServiceConfig,
    ) -> None:
        self.stream_id = stream_id
        self.utterance_id = utterance_id
        self.language = validate_source_language(language)
        self.lid_confidence = lid_confidence
        self.lid_state = lid_state
        self.emit = emit
        self.config = config
        self.context_source_chunks: list[list[str]] = []
        self.context_target_chunks: list[list[int]] = []
        self.active_units: list[str] = []
        self.last_snapshot_units: list[str] = []
        self.source_text_raw = ""
        self.finalized_source_units = 0
        self.active_confirmed_tokens: list[int] = []
        self.previous_hypothesis: list[int] = []
        self.global_confirmed_text = ""
        self.revision = 0
        self.version = 0
        self.final_requested = False
        self.closed = False
        self.last_source_time: float | None = None
        self.last_received_at = time.perf_counter()

    @property
    def stable(self) -> bool:
        return self.lid_state == "stable"

    @property
    def has_committed_output(self) -> bool:
        return bool(self.global_confirmed_text or self.active_confirmed_tokens)

    def change_language(self, language: str, confidence: float | None, state: str) -> None:
        language = validate_source_language(language)
        if state not in {"collecting", "stable", "unknown"}:
            raise ValueError("lid_state must be collecting, stable, or unknown")
        if language != self.language and self.has_committed_output:
            raise ValueError("source language cannot change after target text has been committed")
        if language != self.language:
            self.language = language
            self.context_source_chunks.clear()
            self.context_target_chunks.clear()
            self.active_confirmed_tokens.clear()
            self.previous_hypothesis.clear()
            self.global_confirmed_text = ""
            self.finalized_source_units = 0
            self.last_snapshot_units = source_units(self.source_text_raw, language)
            self.active_units = list(self.last_snapshot_units)
            self.version += 1
        self.lid_confidence = confidence
        self.lid_state = state

    def append(self, text: str, input_mode: str, source_time: float | None, received_at: float) -> None:
        if self.closed or self.final_requested:
            raise ValueError("cannot append after final")
        if not isinstance(text, str):
            raise ValueError("append text must be a string")
        if input_mode not in {"delta", "snapshot"}:
            raise ValueError("input_mode must be delta or snapshot")
        if input_mode == "delta":
            new_units = source_units(text, self.language)
            self.active_units.extend(new_units)
            self.last_snapshot_units.extend(new_units)
            self.source_text_raw = append_text(self.source_text_raw, text)
        else:
            snapshot = source_units(text, self.language)
            common = common_prefix(self.last_snapshot_units, snapshot)
            if common < self.finalized_source_units:
                raise ValueError("snapshot revised source that already crossed a rolling commit boundary")
            new_active = snapshot[self.finalized_source_units :]
            if self.active_confirmed_tokens and common < len(self.last_snapshot_units):
                old_active = self.last_snapshot_units[self.finalized_source_units :]
                if new_active[: len(old_active)] != old_active:
                    raise ValueError("snapshot revised source associated with committed target tokens")
            self.active_units = new_active
            self.last_snapshot_units = snapshot
            self.source_text_raw = text
        if len(self.source_text_raw) > self.config.max_source_characters:
            raise ValueError("source text exceeds the configured character limit")
        self.last_source_time = source_time
        self.last_received_at = received_at
        self.version += 1

    def request_final(self, received_at: float) -> None:
        if self.closed:
            raise ValueError("stream is already final")
        self.final_requested = True
        self.last_received_at = received_at
        self.version += 1

    def _context_source(self) -> list[str]:
        return [unit for chunk in self.context_source_chunks for unit in chunk]

    def _context_target(self) -> list[int]:
        return [token for chunk in self.context_target_chunks for token in chunk]

    def _trim_context(self, active_count: int) -> None:
        capacity = max(0, self.config.rolling_window_units - active_count)
        while self.context_source_chunks and (
            sum(map(len, self.context_source_chunks)) > capacity
            or sum(map(len, self.context_target_chunks)) > self.config.rolling_target_tokens
        ):
            self.context_source_chunks.pop(0)
            self.context_target_chunks.pop(0)

    def make_job(self, received_at: float) -> InferenceJob | None:
        if not self.stable:
            return None
        stride = self.config.rolling_stride_units
        rollover = len(self.active_units) > stride or (
            len(self.active_units) == stride and not self.final_requested
        )
        if not self.active_units:
            return None
        active = list(self.active_units[:stride] if rollover else self.active_units)
        self._trim_context(len(active))
        context_source = self._context_source()
        context_target = self._context_target()
        forced = context_target + self.active_confirmed_tokens
        active_budget = min(
            self.config.rolling_target_tokens,
            max(8, math.ceil(max(1, len(active)) * self.config.target_length_ratio) + 8),
        )
        request = BackendRequest(
            source_language=self.language,
            source_text=join_source_units(context_source + active, self.language),
            forced_prefix_ids=tuple(forced),
            max_target_tokens=min(self.config.max_new_tokens, len(context_target) + active_budget),
        )
        return InferenceJob(
            stream_id=self.stream_id,
            version=self.version,
            received_at=received_at,
            request=request,
            active_units=tuple(active),
            context_target_tokens=tuple(context_target),
            rollover=rollover,
            is_final=self.final_requested and not rollover,
            source_time=self.last_source_time,
        )

    def job_still_applicable(self, job: InferenceJob) -> bool:
        if job.version == self.version:
            return True
        return job.rollover and tuple(self.active_units[: len(job.active_units)]) == job.active_units

    def _safe_boundary(self, backend: TranslationBackend, token_ids: Sequence[int], proposed: int) -> int:
        if proposed <= 0 or proposed >= len(token_ids):
            return proposed
        pieces = backend.token_pieces(token_ids)
        while proposed > 0 and proposed < len(pieces) and not pieces[proposed].startswith("▁"):
            proposed -= 1
        return proposed

    def apply(self, job: InferenceJob, result: BackendResult, backend: TranslationBackend) -> dict:
        token_ids = list(result.token_ids)
        context = list(job.context_target_tokens)
        if token_ids[: len(context)] != context:
            raise RuntimeError("model generation violated the rolling target context")
        active_result = token_ids[len(context) :]
        if active_result[: len(self.active_confirmed_tokens)] != self.active_confirmed_tokens:
            raise RuntimeError("model generation violated the immutable active target prefix")
        old_count = len(self.active_confirmed_tokens)
        if job.rollover or job.is_final:
            confirmed_through = len(active_result)
        elif self.previous_hypothesis:
            agreed = common_prefix(self.previous_hypothesis, active_result)
            confirmed_through = max(old_count, agreed - self.config.holdback_tokens)
            confirmed_through = max(old_count, self._safe_boundary(backend, active_result, confirmed_through))
        else:
            confirmed_through = old_count
        confirmed_through = min(confirmed_through, len(active_result))
        new_confirmed = active_result[old_count:confirmed_through]
        self.active_confirmed_tokens = active_result[:confirmed_through]
        self.previous_hypothesis = list(active_result)
        confirmed_delta = backend.decode(new_confirmed)
        self.global_confirmed_text = append_text(self.global_confirmed_text, confirmed_delta)
        provisional = backend.decode(active_result[confirmed_through:])
        full_text = append_text(self.global_confirmed_text, provisional)
        self.revision += 1
        total_ms = (time.perf_counter() - job.received_at) * 1000.0
        queue_ms = max(0.0, total_ms - result.tokenization_ms - result.inference_ms)
        event = {
            "type": "translation_final" if job.is_final else "translation_update",
            "stream_id": self.stream_id,
            "utterance_id": self.utterance_id,
            "revision": self.revision,
            "source_language": self.language,
            "target_language": TARGET_LANGUAGE,
            "confirmed_delta": confirmed_delta,
            "confirmed_text": self.global_confirmed_text,
            "provisional_tail": "" if job.rollover or job.is_final else provisional,
            "full_text": self.global_confirmed_text if job.rollover or job.is_final else full_text,
            "is_final": job.is_final,
            "window_shift": job.rollover,
            "source_time": job.source_time,
            "timing_ms": {
                "queue": round(queue_ms, 3),
                "tokenization": round(result.tokenization_ms, 3),
                "inference": round(result.inference_ms, 3),
                "serialization": 0.0,
                "total": round(total_ms, 3),
            },
            "source_tokens": result.source_tokens,
            "target_tokens": len(token_ids),
        }
        if job.rollover:
            committed_source = list(job.active_units)
            self.context_source_chunks.append(committed_source)
            self.context_target_chunks.append(list(active_result))
            self.finalized_source_units += len(committed_source)
            del self.active_units[: len(committed_source)]
            self.active_confirmed_tokens = []
            self.previous_hypothesis = []
            self._trim_context(0)
        elif job.is_final:
            self.closed = True
        return event

    def empty_final_event(self) -> dict:
        self.closed = True
        self.revision += 1
        return {
            "type": "translation_final",
            "stream_id": self.stream_id,
            "utterance_id": self.utterance_id,
            "revision": self.revision,
            "source_language": self.language,
            "target_language": TARGET_LANGUAGE,
            "confirmed_delta": "",
            "confirmed_text": self.global_confirmed_text,
            "provisional_tail": "",
            "full_text": self.global_confirmed_text,
            "is_final": True,
            "window_shift": False,
            "timing_ms": {
                "queue": 0.0,
                "tokenization": 0.0,
                "inference": 0.0,
                "serialization": 0.0,
                "total": 0.0,
            },
        }


class TranslationService:
    """Multiplex stream state and dynamically batch one latest request per stream."""

    def __init__(self, backend: TranslationBackend, config: ServiceConfig | None = None):
        self.backend = backend
        self.config = config or ServiceConfig()
        self.config.validate()
        self.sessions: dict[str, StreamState] = {}
        self.pending: collections.OrderedDict[str, float] = collections.OrderedDict()
        self.in_flight: set[str] = set()
        self.wake = asyncio.Event()
        self.worker: asyncio.Task | None = None
        self.backend_thread = _BackendThread(backend)
        self.stopping = False

    async def start(self) -> None:
        if self.worker is None:
            self.backend_thread.start()
            self.worker = asyncio.create_task(self._run(), name="m2m100-ct2-batcher")

    async def close(self) -> None:
        self.stopping = True
        self.wake.set()
        if self.worker is not None:
            await self.worker
            self.worker = None
        self.backend_thread.close()

    def _queue(self, stream_id: str, received_at: float) -> None:
        self.pending[stream_id] = received_at
        self.pending.move_to_end(stream_id)
        self.wake.set()

    async def handle(self, row: dict, emit: Emit) -> str | None:
        if not isinstance(row, dict):
            raise ValueError("each input line must be a JSON object")
        event_type = row.get("type")
        stream_id = row.get("stream_id")
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id is required")
        now = time.perf_counter()
        if event_type == "open":
            if stream_id in self.sessions:
                raise ValueError(f"duplicate stream_id: {stream_id}")
            lid_state = str(row.get("lid_state", "collecting"))
            if lid_state not in {"collecting", "stable", "unknown"}:
                raise ValueError("lid_state must be collecting, stable, or unknown")
            state = StreamState(
                stream_id,
                str(row.get("utterance_id") or stream_id),
                row.get("source_language"),
                float(row["lid_confidence"]) if row.get("lid_confidence") is not None else None,
                lid_state,
                emit,
                self.config,
            )
            self.sessions[stream_id] = state
            await emit({
                "type": "opened",
                "stream_id": stream_id,
                "utterance_id": state.utterance_id,
                "source_language": state.language,
                "target_language": TARGET_LANGUAGE,
                "lid_state": state.lid_state,
            })
            return stream_id
        state = self.sessions.get(stream_id)
        if state is None:
            raise ValueError(f"unknown stream_id: {stream_id}")
        if event_type == "language":
            state.change_language(
                row.get("source_language"),
                float(row["lid_confidence"]) if row.get("lid_confidence") is not None else None,
                str(row.get("lid_state", state.lid_state)),
            )
            if state.stable and state.active_units:
                self._queue(stream_id, now)
            return stream_id
        if event_type == "append":
            state.append(
                row.get("text", ""),
                str(row.get("input_mode", "delta")),
                float(row["source_time"]) if row.get("source_time") is not None else None,
                now,
            )
            if state.stable and state.active_units:
                self._queue(stream_id, now)
            else:
                await emit({
                    "type": "buffered",
                    "stream_id": stream_id,
                    "utterance_id": state.utterance_id,
                    "lid_state": state.lid_state,
                })
            return stream_id
        if event_type == "final":
            if not state.stable:
                raise ValueError("cannot finalize translation without a stable source language")
            state.request_final(now)
            if state.active_units:
                self._queue(stream_id, now)
            else:
                await emit(state.empty_final_event())
                self.sessions.pop(stream_id, None)
            return stream_id
        if event_type == "close":
            self.remove(stream_id)
            return stream_id
        raise ValueError("type must be open, language, append, final, or close")

    def remove(self, stream_id: str) -> None:
        self.pending.pop(stream_id, None)
        self.sessions.pop(stream_id, None)

    async def _run(self) -> None:
        while not self.stopping or self.pending or self.in_flight:
            if not self.pending:
                self.wake.clear()
                if self.stopping and not self.in_flight:
                    break
                await self.wake.wait()
                continue
            if self.config.batch_wait_ms:
                await asyncio.sleep(self.config.batch_wait_ms / 1000.0)
            selected = []
            for stream_id in list(self.pending):
                if stream_id in self.in_flight:
                    continue
                received_at = self.pending.pop(stream_id)
                state = self.sessions.get(stream_id)
                if state is None:
                    continue
                job = state.make_job(received_at)
                if job is None:
                    if state.final_requested and not state.active_units:
                        await state.emit(state.empty_final_event())
                        self.sessions.pop(stream_id, None)
                    continue
                selected.append((state, job))
                self.in_flight.add(stream_id)
                if len(selected) >= self.config.max_batch_size:
                    break
            if not selected:
                await asyncio.sleep(0)
                continue
            requests = [job.request for _, job in selected]
            try:
                results = await self.backend_thread.translate(requests)
                if len(results) != len(selected):
                    raise RuntimeError("translation backend returned the wrong batch length")
                for (state, job), result in zip(selected, results, strict=True):
                    if state.stream_id not in self.sessions:
                        continue
                    if state.job_still_applicable(job):
                        event = state.apply(job, result, self.backend)
                        await state.emit(event)
                        if state.closed:
                            self.sessions.pop(state.stream_id, None)
                    needs_followup = (
                        state.version != job.version
                        or (job.rollover and bool(state.active_units))
                        or (state.final_requested and not state.closed and not job.is_final)
                    )
                    if state.stream_id in self.sessions and needs_followup:
                        self._queue(state.stream_id, state.last_received_at)
            except Exception as exc:
                for state, _ in selected:
                    if state.stream_id in self.sessions:
                        await state.emit({
                            "type": "error",
                            "stream_id": state.stream_id,
                            "utterance_id": state.utterance_id,
                            "message": str(exc),
                        })
            finally:
                for state, _ in selected:
                    self.in_flight.discard(state.stream_id)
            if self.pending:
                self.wake.set()


LOG = logging.getLogger("m2m100.streaming_server")


class ConnectionEmitter:
    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer
        self.lock = asyncio.Lock()
        self.closed = False

    async def __call__(self, event: dict) -> None:
        if self.closed:
            return
        if event.get("type") in {"translation_update", "translation_final"}:
            started = time.perf_counter()
            json.dumps(event, ensure_ascii=False, separators=(",", ":"))
            serialization_ms = (time.perf_counter() - started) * 1000.0
            timing = event.setdefault("timing_ms", {})
            timing["serialization"] = round(serialization_ms, 3)
            timing["total"] = round(float(timing.get("total", 0.0)) + serialization_ms, 3)
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        async with self.lock:
            self.writer.write(payload)
            await self.writer.drain()


async def serve_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    service: TranslationService,
    client_timeout: float,
) -> None:
    emitter = ConnectionEmitter(writer)
    owned_streams: set[str] = set()
    peer = writer.get_extra_info("peername")
    LOG.info("connected: %s", peer)
    row: object = {}
    try:
        while True:
            try:
                raw = (
                    await asyncio.wait_for(reader.readline(), timeout=client_timeout)
                    if client_timeout > 0
                    else await reader.readline()
                )
            except asyncio.TimeoutError:
                await emitter({"type": "error", "message": "client idle timeout", "is_final": True})
                break
            if not raw:
                break
            if len(raw) > 1024 * 1024:
                await emitter({"type": "error", "message": "JSON line exceeds 1 MiB"})
                continue
            try:
                row = json.loads(raw.replace(b"\0", b"").decode("utf-8"))
                stream_id = await service.handle(row, emitter)
                if row.get("type") == "open" and stream_id:
                    owned_streams.add(stream_id)
                elif row.get("type") == "close" and stream_id:
                    owned_streams.discard(stream_id)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                await emitter({
                    "type": "error",
                    "stream_id": row.get("stream_id") if isinstance(row, dict) else None,
                    "message": str(exc),
                })
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        emitter.closed = True
        for stream_id in owned_streams:
            service.remove(stream_id)
        writer.close()
        await writer.wait_closed()
        LOG.info("disconnected: %s", peer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="CTranslate2 model directory")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer directory; defaults to --model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--rolling-window-units", type=int, default=64)
    parser.add_argument("--rolling-stride-units", type=int, default=16)
    parser.add_argument("--rolling-target-tokens", type=int, default=256)
    parser.add_argument("--max-source-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--target-length-ratio", type=float, default=1.6)
    parser.add_argument("--holdback-tokens", type=int, default=1)
    parser.add_argument("--max-source-characters", type=int, default=32_768)
    parser.add_argument("--client-timeout", type=float, default=0.0)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    config = ServiceConfig(
        batch_wait_ms=args.batch_wait_ms,
        max_batch_size=args.max_batch_size,
        rolling_window_units=args.rolling_window_units,
        rolling_stride_units=args.rolling_stride_units,
        rolling_target_tokens=args.rolling_target_tokens,
        max_source_tokens=args.max_source_tokens,
        max_new_tokens=args.max_new_tokens,
        target_length_ratio=args.target_length_ratio,
        holdback_tokens=args.holdback_tokens,
        max_source_characters=args.max_source_characters,
    )
    config.validate()
    model_path = args.model.resolve()
    tokenizer_path = (args.tokenizer or args.model).resolve()
    backend = CT2Backend(
        model_path,
        tokenizer_path,
        device=args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
        max_source_tokens=args.max_source_tokens,
    )
    service = TranslationService(backend, config)
    await service.start()
    server = await asyncio.start_server(
        lambda reader, writer: serve_connection(reader, writer, service, args.client_timeout),
        args.host,
        args.port,
        limit=1024 * 1024 + 1,
    )
    addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    LOG.info("listening on %s", addresses)
    try:
        async with server:
            await server.serve_forever()
    finally:
        await service.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s\t%(message)s")
    if not 0 <= args.port <= 65535 or args.client_timeout < 0:
        raise SystemExit("invalid port or timeout")
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
