"""
Custom Faster-Whisper STT plugin for livekit-agents.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from livekit.agents import stt, utils

logger = logging.getLogger(__name__)


@dataclass
class FasterWhisperOptions:
    model_path: str = "./models/faster-whisper-small"
    device: str = "cpu"
    compute_type: str | None = None
    language: str | None = None
    beam_size: int = 1
    best_of: int = 1
    vad_filter: bool = True


class FasterWhisperSTT(stt.STT):
    """
    Faster-Whisper STT plugin for LiveKit Agents.
    Uses local Whisper models for speech-to-text.
    """

    def __init__(
        self,
        *,
        model_path: str = "./models/faster-whisper-small",
        device: str = "cpu",
        compute_type: str | None = None,
        language: str | None = None,
        beam_size: int = 1,
        best_of: int = 1,
        vad_filter: bool = True,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self._opts = FasterWhisperOptions(
            model_path=model_path,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            best_of=best_of,
            vad_filter=vad_filter,
        )

        self._model = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the model on first use."""
        if self._loaded:
            return

        # Import here to ensure CUDA context is set up properly
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self._opts.model_path,
            device=self._opts.device,
            compute_type=self._opts.compute_type,
            local_files_only=True,
        )
        self._loaded = True

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: stt.STTConnOptions | None = None,
    ) -> stt.SpeechEvent:
        """Transcribe audio buffer using Faster-Whisper."""
        self._ensure_loaded()

        # Convert audio buffer to numpy array
        audio_data = np.frombuffer(buffer.data, dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio_float,
                beam_size=self._opts.beam_size,
                best_of=self._opts.best_of,
                language=language or self._opts.language,
                task="transcribe",
                vad_filter=self._opts.vad_filter,
            ),
        )

        # Collect all text from segments
        text = "".join([segment.text for segment in segments]).strip()

        # Log recognized speech for visibility
        if text:
            logger.info(f"Client said: {text}")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=info.language if info.language else "en",
                    confidence=1.0,
                )
            ],
        )

    async def recognize(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: str | None = None,
        conn_options: stt.STTConnOptions | None = None,
    ) -> stt.SpeechEvent:
        """Recognize speech from the given audio buffer."""
        return await self._recognize_impl(
            buffer, language=language, conn_options=conn_options
        )
