# tts-service/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import numpy as np
import logging
import time
import os

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("f5-tts-service")


class HealthCheckFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/health" not in message


logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

# Model paths (pre-downloaded, no network access needed)
F5_MODEL_PATH = "./models/f5-tts/F5TTS_v1_Base/model_48000.safetensors"
F5_VOCAB_PATH = "./models/f5-tts/F5TTS_v1_Base/vocab.txt"
VOCOS_PATH = "./models/vocos-mel-24khz"
REF_AUDIO_PATH = "./ref/alfred-reference.wav"
with open("./ref/alfred-reference.txt", "r") as f:
    REF_TEXT = f.read()

# Global model state
state = {
    "model": None,
    "vocoder": None,
    "ref_audio": None,
    "ref_text_proc": None,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Metrics tracking
metrics = {
    "start_time": time.time(),
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_audio_seconds": 0.0,
    "total_latency_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at container startup (no downloads)"""
    logger.info(f"Loading F5-TTS on {state['device']}...")

    state["vocoder"] = load_vocoder(is_local=True, local_path=VOCOS_PATH)

    # Load DiT Model
    state["model"] = load_model(
        model_cls=DiT,
        model_cfg=dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        ),
        ckpt_path=F5_MODEL_PATH,
        mel_spec_type="vocos",
        vocab_file=F5_VOCAB_PATH,
        ode_method="euler",
        use_ema=True,
        device=state["device"],
    )

    logger.info("Pre-processing reference audio...")
    state["ref_audio"], state["ref_text_proc"] = preprocess_ref_audio_text(
        REF_AUDIO_PATH, REF_TEXT
    )

    logger.info("F5-TTS Ready!")
    yield


app = FastAPI(lifespan=lifespan)


class SynthesisRequest(BaseModel):
    text: str
    speed: float = 1.0


@app.post("/synthesize")
async def synthesize(req: SynthesisRequest):
    _synthesis_start_time = time.perf_counter()
    metrics["total_requests"] += 1

    if not req.text.strip():
        metrics["successful_requests"] += 1
        return Response(content=b"", media_type="audio/pcm")

    try:
        # Run Inference
        audio, sample_rate, _ = infer_process(
            state["ref_audio"],
            state["ref_text_proc"],
            req.text,
            state["model"],
            state["vocoder"],
            mel_spec_type="vocos",
            speed=req.speed,
            device=state["device"],
        )

        # Convert to PCM bytes (int16)
        audio = np.array(audio, dtype=np.float32)
        audio = audio / np.max(np.abs(audio))  # Normalize
        audio_int16 = (audio * 32767).astype(np.int16)

        latency_ms = (time.perf_counter() - _synthesis_start_time) * 1000
        logger.info(f"Synthesis time: {latency_ms:.0f}ms")

        # Update metrics
        audio_duration_seconds = len(audio) / sample_rate
        metrics["successful_requests"] += 1
        metrics["total_audio_seconds"] += audio_duration_seconds
        metrics["total_latency_ms"] += latency_ms

        return Response(content=audio_int16.tobytes(), media_type="audio/pcm")

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        metrics["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "device": state["device"]}


@app.get("/metrics")
def get_metrics():
    """Return service metrics for monitoring."""
    successful = metrics["successful_requests"]
    avg_latency = (
        metrics["total_latency_ms"] / successful if successful > 0 else 0.0
    )
    return {
        "total_requests": metrics["total_requests"],
        "successful_requests": successful,
        "failed_requests": metrics["failed_requests"],
        "total_audio_seconds": round(metrics["total_audio_seconds"], 1),
        "average_latency_ms": round(avg_latency, 1),
        "uptime_seconds": round(time.time() - metrics["start_time"]),
    }
