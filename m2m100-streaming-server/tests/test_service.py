import asyncio
import unittest

from translate_server import BackendResult, ServiceConfig, TranslationService, validate_source_language


class FakeBackend:
    def __init__(self):
        self.batch_sizes = []
        self.words = dict(enumerate((
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        )))

    def translate_batch(self, requests):
        self.batch_sizes.append(len(requests))
        output = []
        for request in requests:
            unit_count = (
                len(request.source_text)
                if request.source_language in {"zh", "ja"}
                else len(request.source_text.split())
            )
            base = list(range(1, max(2, unit_count + 2)))
            prefix = list(request.forced_prefix_ids)
            if base[: len(prefix)] != prefix:
                base = prefix + list(range(len(prefix) + 1, max(len(prefix) + 2, unit_count + 2)))
            output.append(BackendResult(tuple(base), unit_count, 1.0))
        return output

    def decode(self, token_ids):
        return " ".join(self.words.get(token, f"t{token}") for token in token_ids)

    def token_pieces(self, token_ids):
        return [f"▁{self.words.get(token, token)}" for token in token_ids]


async def wait_for(events, predicate, timeout=1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for event in events:
            if predicate(event):
                return event
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for event; got {events}")


class LanguageTests(unittest.TestCase):
    def test_supported_and_unsupported_languages(self):
        self.assertEqual(validate_source_language("uk"), "uk")
        with self.assertRaisesRegex(ValueError, "source_language"):
            validate_source_language("de")


class StreamingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = FakeBackend()
        self.service = TranslationService(
            self.backend,
            ServiceConfig(
                batch_wait_ms=5,
                max_batch_size=8,
                rolling_window_units=8,
                rolling_stride_units=4,
                holdback_tokens=1,
            ),
        )
        await self.service.start()

    async def asyncTearDown(self):
        await self.service.close()

    async def test_local_agreement_provisional_commit_and_final(self):
        events = []

        async def emit(event):
            events.append(event)

        await self.service.handle({
            "type": "open", "stream_id": "s", "utterance_id": "u", "source_language": "es",
            "lid_state": "stable", "lid_confidence": 0.99,
        }, emit)
        await self.service.handle({
            "type": "append", "stream_id": "s", "text": "hola", "input_mode": "delta",
        }, emit)
        first = await wait_for(events, lambda row: row["type"] == "translation_update")
        self.assertEqual(first["confirmed_text"], "")
        self.assertEqual(first["provisional_tail"], "one two")

        await self.service.handle({
            "type": "append", "stream_id": "s", "text": "mundo", "input_mode": "delta",
        }, emit)
        second = await wait_for(events, lambda row: row.get("revision") == 2)
        self.assertEqual(second["confirmed_text"], "one")
        self.assertEqual(second["provisional_tail"], "two three")

        await self.service.handle({"type": "final", "stream_id": "s"}, emit)
        final = await wait_for(events, lambda row: row["type"] == "translation_final")
        self.assertEqual(final["full_text"], "one two three")
        self.assertTrue(final["is_final"])

    async def test_four_streams_share_one_dynamic_batch(self):
        streams = (
            ("a", "es", "hola"), ("b", "fr", "bonjour"),
            ("c", "zh", "你好"), ("d", "uk", "вітаю"),
        )
        events = {stream_id: [] for stream_id, _, _ in streams}

        def emitter(stream_id):
            async def emit(event):
                events[stream_id].append(event)
            return emit

        for stream_id, language, _ in streams:
            await self.service.handle({
                "type": "open", "stream_id": stream_id, "source_language": language,
                "lid_state": "stable",
            }, emitter(stream_id))
        for stream_id, _, text in streams:
            await self.service.handle({
                "type": "append", "stream_id": stream_id, "text": text,
            }, emitter(stream_id))
        await asyncio.gather(*[
            wait_for(events[stream_id], lambda row: row["type"] == "translation_update")
            for stream_id, _, _ in streams
        ])
        self.assertIn(4, self.backend.batch_sizes)

    async def test_unstable_lid_buffers_and_late_correction_is_rejected(self):
        events = []

        async def emit(event):
            events.append(event)

        await self.service.handle({
            "type": "open", "stream_id": "lid", "source_language": "es", "lid_state": "collecting",
        }, emit)
        await self.service.handle({"type": "append", "stream_id": "lid", "text": "hola"}, emit)
        self.assertTrue(any(event["type"] == "buffered" for event in events))
        await self.service.handle({
            "type": "language", "stream_id": "lid", "source_language": "fr", "lid_state": "stable",
        }, emit)
        await wait_for(events, lambda row: row["type"] == "translation_update")
        await self.service.handle({"type": "append", "stream_id": "lid", "text": "monde"}, emit)
        await wait_for(events, lambda row: row.get("confirmed_text") == "one")
        with self.assertRaisesRegex(ValueError, "cannot change"):
            await self.service.handle({
                "type": "language", "stream_id": "lid", "source_language": "es", "lid_state": "stable",
            }, emit)


if __name__ == "__main__":
    unittest.main()
