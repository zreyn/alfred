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

    if not req.text.strip():
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

        return Response(content=audio_int16.tobytes(), media_type="audio/pcm")

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "device": state["device"]}
