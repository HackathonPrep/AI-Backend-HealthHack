import asyncio
import base64
import json
import logging
import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import Settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHUNK_MS = 400
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1_000)
CONTEXT_SAMPLES = SAMPLE_RATE * 3
ENERGY_THRESHOLD = 150.0
TRANSCRIBE_INTERVAL_SECONDS = 0.3
AUTO_STOP_SILENCE_SECONDS = 1.5


def pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def has_speech(audio: np.ndarray) -> bool:
    return audio.size > 0 and np.abs(audio).mean() * 32768.0 > ENERGY_THRESHOLD


def transcribe_text(model: WhisperModel, audio: np.ndarray, **kwargs: object) -> str:
    """Consume Whisper's segment iterator off the event loop."""
    segments, _ = model.transcribe(audio, **kwargs)
    return " ".join(segment.text.strip() for segment in segments).strip()


class TranscriptionService:
    """Owns lazily loaded Whisper models shared by WebSocket sessions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: WhisperModel | None = None
        self.final_model: WhisperModel | None = None
        self._load_lock = asyncio.Lock()

    async def ensure_models_loaded(self) -> None:
        if self.model is not None:
            return
        async with self._load_lock:
            if self.model is None:
                await asyncio.to_thread(self._load_models)

    def _load_models(self) -> None:
        logger.info("Loading Whisper streaming model: %s", self.settings.whisper_model)
        self.model = WhisperModel(
            self.settings.whisper_model, device="cpu", compute_type="int8"
        )
        if self.settings.whisper_final_model != self.settings.whisper_model:
            logger.info(
                "Loading Whisper refinement model: %s",
                self.settings.whisper_final_model,
            )
            self.final_model = WhisperModel(
                self.settings.whisper_final_model, device="cpu", compute_type="int8"
            )

    async def run_session(self, websocket: WebSocket) -> None:
        await self.ensure_models_loaded()
        await TranscriptionSession(websocket, self).run()


class TranscriptionSession:
    def __init__(self, websocket: WebSocket, service: TranscriptionService) -> None:
        self.ws = websocket
        self.service = service
        self.full_buffer = np.array([], dtype=np.float32)
        self.context_buffer = np.array([], dtype=np.float32)
        self.locked_text = ""
        self.unstable_text = ""
        self.same_count = 0
        self.last_transcribe_time = 0.0
        self.last_speech_time: Optional[float] = None
        self.has_ever_spoken = False
        self.running = True
        self._transcribe_lock = asyncio.Lock()
        self._finalizing = False

    async def run(self) -> None:
        try:
            while self.running and not self._finalizing:
                try:
                    data = json.loads(
                        await asyncio.wait_for(self.ws.receive_text(), timeout=0.05)
                    )
                    if data.get("type") == "audio_chunk":
                        await self._handle_chunk(data.get("data", ""))
                    elif data.get("type") == "stop":
                        await self._finalize()
                        return
                    elif data.get("type") == "ping":
                        await self.ws.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    if self._should_auto_stop():
                        await self._finalize()
                        return
                    await self._try_transcribe()
        except WebSocketDisconnect:
            logger.info("Transcription client disconnected")
        except Exception:
            logger.exception("Transcription session failed")

    def _should_auto_stop(self) -> bool:
        return bool(
            self.has_ever_spoken
            and self.last_speech_time is not None
            and time.time() - self.last_speech_time >= AUTO_STOP_SILENCE_SECONDS
        )

    async def _handle_chunk(self, encoded_data: str) -> None:
        if self._finalizing:
            return
        try:
            chunk = pcm_to_float32(base64.b64decode(encoded_data))
            self.full_buffer = np.concatenate((self.full_buffer, chunk))
            self.context_buffer = np.concatenate((self.context_buffer, chunk))[
                -CONTEXT_SAMPLES:
            ]
            if has_speech(chunk):
                self.last_speech_time = time.time()
                self.has_ever_spoken = True
            if self._should_auto_stop():
                await self._finalize()
                return
            await self._try_transcribe()
        except Exception:
            logger.exception("Unable to process audio chunk")

    async def _try_transcribe(self) -> None:
        if (
            self._finalizing
            or time.time() - self.last_transcribe_time < TRANSCRIBE_INTERVAL_SECONDS
            or self.context_buffer.size < CHUNK_SAMPLES
            or self._transcribe_lock.locked()
        ):
            return
        async with self._transcribe_lock:
            self.last_transcribe_time = time.time()
            try:
                assert self.service.model is not None
                text = await asyncio.to_thread(
                    transcribe_text,
                    self.service.model,
                    self.context_buffer,
                    beam_size=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
                if not text:
                    return
                full_text = f"{self.locked_text} {text}".strip()
                if text == self.unstable_text:
                    self.same_count += 1
                    if self.same_count >= 2:
                        self.locked_text, self.unstable_text, self.same_count = (
                            full_text,
                            "",
                            0,
                        )
                else:
                    self.unstable_text, self.same_count = text, 0
                await self._safe_send({"type": "partial", "text": full_text})
            except Exception:
                logger.exception("Streaming transcription failed")

    async def _finalize(self) -> None:
        if self._finalizing:
            return
        self._finalizing, self.running = True, False
        if self.full_buffer.size == 0:
            await self._safe_send({"type": "final", "text": ""})
            return
        try:
            model = self.service.final_model or self.service.model
            assert model is not None
            text = await asyncio.to_thread(
                transcribe_text,
                model,
                self.full_buffer,
                beam_size=3,
                temperature=0.0,
                condition_on_previous_text=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            await self._safe_send(
                {
                    "type": "final",
                    "text": text,
                }
            )
        except Exception:
            logger.exception("Final transcription failed")
            await self._safe_send(
                {"type": "final", "text": f"{self.locked_text} {self.unstable_text}".strip()}
            )

    async def _safe_send(self, data: dict) -> None:
        try:
            await self.ws.send_json(data)
        except Exception:
            pass
