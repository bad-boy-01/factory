"""
Audio Adapter — Novel Video Factory v5
PRIMARY:  edge-tts (Microsoft Edge TTS, FREE, no API key, very natural voices)
FALLBACK: Silent WAV (keeps pipeline running if TTS fails)

Install: pip install edge-tts
"""
import asyncio
import logging
import os
import struct
import wave
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LocalAudioAdapter:
    def __init__(self, config: dict = None):
        cfg = config or {}
        audio_cfg = cfg.get("models", {}).get("audio", {})
        self.provider = audio_cfg.get("provider", "edge_tts")
        self.voice = audio_cfg.get("voice", "en-US-AndrewNeural")
        self.max_chars_per_call = audio_cfg.get("max_chars_per_tts_call", 8000)
        self._edge_tts_warned = False   # log missing edge-tts only once
        logger.info(f"Audio adapter: provider={self.provider}, voice={self.voice}")

    # ── Public Interface ──────────────────────────────────────────────────────
    def generate_audio(self, text: str, output_path: str):
        """Generate speech from text and save as WAV. Single-clip, no timing."""
        if not text or not text.strip():
            text = "..."

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if self.provider == "edge_tts":
            if self._edge_tts(text, output_path):
                return

        # Last resort: silence (duration based on word count)
        duration = max(1.5, len(text.split()) * 0.38)
        self._mock_wav(output_path, duration=duration)

    def generate_audio_with_timestamps(
        self, text: str, output_path: str
    ) -> Optional[List[Dict]]:
        """
        Generate one continuous narration WAV for `text` and return
        word-level timing metadata, so callers can compute exact
        (start_time, end_time) windows for arbitrary substrings of `text`
        without needing separate per-substring TTS calls.

        Returns a list of {"text_offset": int, "word_len": int,
        "start": float, "end": float} (times in seconds, text_offset is
        the character offset of the word's start in `text`), or None if
        word-boundary capture failed (caller should fall back to
        per-scene generate_audio()).

        If `text` exceeds max_chars_per_tts_call, it's split on sentence/
        whitespace boundaries into multiple TTS calls, whose audio is
        concatenated and whose timestamps are offset to stay correct
        against the single combined output file.
        """
        if not text or not text.strip():
            return None
        if self.provider != "edge_tts":
            return None

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        chunks = self._split_for_tts(text, self.max_chars_per_call)
        all_words: List[Dict] = []
        segment_mp3_paths: List[str] = []
        char_cursor = 0
        time_cursor = 0.0

        try:
            import edge_tts  # type: ignore
        except ImportError:
            if not self._edge_tts_warned:
                logger.warning("edge-tts not installed. Run: pip install edge-tts")
                self._edge_tts_warned = True
            return None

        for i, chunk_text in enumerate(chunks):
            tmp_mp3 = output_path.replace(".wav", f"_seg{i}.mp3")
            words, seg_duration = self._edge_tts_with_boundaries(chunk_text, tmp_mp3, edge_tts)
            if words is None:
                # Clean up any partial segments and bail — caller falls back
                for p in segment_mp3_paths:
                    if os.path.exists(p):
                        os.remove(p)
                if os.path.exists(tmp_mp3):
                    os.remove(tmp_mp3)
                return None

            for w in words:
                all_words.append({
                    "text_offset": w["text_offset"] + char_cursor,
                    "word_len": w["word_len"],
                    "start": w["start"] + time_cursor,
                    "end": w["end"] + time_cursor,
                })

            segment_mp3_paths.append(tmp_mp3)
            char_cursor += len(chunk_text) + 1  # +1 for the joining space removed by _split_for_tts
            time_cursor += seg_duration

        if not segment_mp3_paths:
            return None

        # Concatenate segment MP3s (if more than one) and convert to WAV
        ok = self._concat_and_convert(segment_mp3_paths, output_path)
        for p in segment_mp3_paths:
            if os.path.exists(p):
                os.remove(p)
        if not ok:
            return None

        return all_words

    @staticmethod
    def _split_for_tts(text: str, max_chars: int) -> List[str]:
        """Split text into <=max_chars pieces on sentence boundaries where
        possible, falling back to whitespace boundaries for any single
        run-on sentence longer than the limit."""
        if len(text) <= max_chars:
            return [text]

        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for sent in sentences:
            if len(sent) > max_chars:
                # Pathological single sentence — hard-split on whitespace
                words = sent.split(" ")
                piece = ""
                for word in words:
                    if len(piece) + len(word) + 1 > max_chars:
                        if piece:
                            chunks.append(piece)
                        piece = word
                    else:
                        piece = f"{piece} {word}" if piece else word
                if piece:
                    chunks.append(piece)
                continue
            candidate = f"{current} {sent}" if current else sent
            if len(candidate) > max_chars:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [text]

    def _edge_tts_with_boundaries(
        self, text: str, tmp_mp3_path: str, edge_tts_module
    ) -> Tuple[Optional[List[Dict]], float]:
        """
        Stream synthesis for one chunk, capturing both audio bytes and
        WordBoundary events. WordBoundary offsets/durations come back in
        100-nanosecond units (Azure Speech SDK convention) — divide by
        1e7 for seconds. text_offset is reported in the same units used
        by the underlying SDK; for ASCII/Latin narration text (the normal
        case post-translation) this lines up with plain character index.
        """
        words: List[Dict] = []
        audio_bytes = bytearray()

        async def _gen():
            comm = edge_tts_module.Communicate(text, self.voice)
            async for event in comm.stream():
                if event["type"] == "audio":
                    audio_bytes.extend(event["data"])
                elif event["type"] == "WordBoundary":
                    words.append({
                        "text_offset": event["text_offset"],
                        "word_len": event.get("word_len", len(event.get("text", ""))),
                        "start": event["offset"] / 1e7,
                        "end": (event["offset"] + event["duration"]) / 1e7,
                    })

        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_gen())
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"edge-tts streaming (word boundaries) failed: {e}")
            return None, 0.0

        if not audio_bytes:
            return None, 0.0

        with open(tmp_mp3_path, "wb") as f:
            f.write(audio_bytes)

        duration = words[-1]["end"] if words else 0.0
        # Refine duration against the actual audio length via ffprobe so
        # trailing silence after the last word isn't lost.
        try:
            import subprocess, json as _json
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "json", tmp_mp3_path],
                capture_output=True,
            )
            if probe.returncode == 0:
                real_dur = float(_json.loads(probe.stdout)["format"]["duration"])
                duration = max(duration, real_dur)
        except Exception:
            pass

        return words, duration

    @staticmethod
    def _concat_and_convert(mp3_paths: List[str], output_wav_path: str) -> bool:
        """Concatenate one or more MP3 segments (in order) into a single WAV."""
        import subprocess
        try:
            if len(mp3_paths) == 1:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", mp3_paths[0], "-ar", "22050", "-ac", "1", output_wav_path],
                    capture_output=True,
                )
                return result.returncode == 0 and os.path.exists(output_wav_path)

            list_path = output_wav_path + ".concat.txt"
            with open(list_path, "w", encoding="utf-8") as f:
                for p in mp3_paths:
                    f.write(f"file '{os.path.abspath(p)}'\n")
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                 "-ar", "22050", "-ac", "1", output_wav_path],
                capture_output=True,
            )
            if os.path.exists(list_path):
                os.remove(list_path)
            return result.returncode == 0 and os.path.exists(output_wav_path)
        except Exception as e:
            logger.warning(f"Audio concat/convert failed: {e}")
            return False

    # ── edge-tts ──────────────────────────────────────────────────────────────
    def _edge_tts(self, text: str, output_path: str) -> bool:
        """
        Microsoft Edge TTS — completely free, no API key, very natural voices.
        Generates MP3 then converts to WAV with ffmpeg.
        """
        try:
            import edge_tts  # type: ignore

            tmp_mp3 = output_path.replace(".wav", "_tmp.mp3")

            async def _gen():
                comm = edge_tts.Communicate(text, self.voice)
                await comm.save(tmp_mp3)

            # Run async in a fresh event loop (safe for Kaggle)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_gen())
            finally:
                loop.close()

            # Convert MP3 → WAV with ffmpeg (available on Kaggle/Colab)
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "22050", "-ac", "1", output_path],
                capture_output=True,
            )
            if os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)

            if result.returncode == 0 and os.path.exists(output_path):
                logger.debug(f"edge-tts ✓: {os.path.basename(output_path)}")
                return True
            else:
                logger.warning(f"ffmpeg WAV conversion failed: {result.stderr.decode()[:200]}")
                return False

        except ImportError:
            if not self._edge_tts_warned:
                logger.warning("edge-tts not installed. Run: pip install edge-tts")
                self._edge_tts_warned = True
            return False
        except Exception as e:
            logger.warning(f"edge-tts failed: {e}")
            return False

    # ── Silent WAV fallback ───────────────────────────────────────────────────
    def _mock_wav(self, output_path: str, duration: float = 2.0):
        """Create a silent WAV file so the video pipeline doesn't crash."""
        sample_rate = 22050
        n_frames = int(sample_rate * duration)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
        logger.debug(f"Silent WAV ({duration:.1f}s): {os.path.basename(output_path)}")
