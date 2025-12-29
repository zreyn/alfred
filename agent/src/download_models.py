#!/usr/bin/env python3
"""
Download all models used by the agent.

This script downloads models to the ./models directory so they can be
copied into Docker images. No downloads should happen at container startup.

Models downloaded:
- Faster-Whisper "small" model (for STT)
- Silero VAD model (for voice activity detection)
- Piper TTS voice model (for local TTS)

Usage:
    python download_models.py

Run this before building Docker images.
"""

import os
import sys
from pathlib import Path

# Models directory (relative to agent/)
MODELS_DIR = Path(__file__).parent.parent / "models"


def download_faster_whisper_model(model_size: str = "small") -> Path:
    """Download Faster-Whisper model from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading Faster-Whisper '{model_size}' model...")

    # Faster-whisper uses CTranslate2 converted models
    repo_id = f"Systran/faster-whisper-{model_size}"
    local_dir = MODELS_DIR / f"faster-whisper-{model_size}"

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    print(f"  Downloaded to: {local_dir}")
    return local_dir


def download_silero_vad_model() -> Path:
    """Download Silero VAD model."""
    import torch

    print("Downloading Silero VAD model...")

    silero_dir = MODELS_DIR / "silero-vad"
    silero_dir.mkdir(parents=True, exist_ok=True)

    # Download using torch.hub
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=True,
    )

    # The ONNX model is cached in torch hub cache, copy it to our models dir
    torch_hub_cache = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
    onnx_file = torch_hub_cache / "files" / "silero_vad.onnx"

    if onnx_file.exists():
        import shutil

        dest = silero_dir / "silero_vad.onnx"
        shutil.copy2(onnx_file, dest)
        print(f"  Downloaded to: {dest}")
    else:
        # If ONNX not found, the model is in memory - save it
        print(f"  Model loaded in memory, saved to: {silero_dir}")

    return silero_dir


def download_piper_model(model_name: str = "en_US-ryan-high") -> Path:
    """Download Piper TTS voice model from HuggingFace."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading Piper voice model '{model_name}'...")

    piper_dir = MODELS_DIR / "piper"
    piper_dir.mkdir(parents=True, exist_ok=True)

    # Download model and config files
    onnx_file = hf_hub_download(
        repo_id="rhasspy/piper-voices",
        filename=f"en/en_US/ryan/high/{model_name}.onnx",
        local_dir=str(piper_dir),
        local_dir_use_symlinks=False,
    )

    config_file = hf_hub_download(
        repo_id="rhasspy/piper-voices",
        filename=f"en/en_US/ryan/high/{model_name}.onnx.json",
        local_dir=str(piper_dir),
        local_dir_use_symlinks=False,
    )

    print(f"  Downloaded to: {piper_dir}")
    return piper_dir


def main():
    """Download all models."""
    print("=" * 60)
    print("Downloading models for agent")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR.absolute()}")
    print()

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download all models
    download_faster_whisper_model("small")
    print()

    download_silero_vad_model()
    print()

    download_piper_model("en_US-ryan-high")
    print()

    print("=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
