#!/usr/bin/env python3
"""
Download all models used by the F5-TTS service.

This script downloads models to the ./models directory so they can be
copied into Docker images. No downloads should happen at container startup.

Models downloaded:
- F5-TTS model (DiT-based TTS)
- Vocos vocoder (mel spectrogram to audio)

Usage:
    python download_models.py

Run this before building Docker images.
"""

import os
from pathlib import Path

# Models directory (relative to f5-tts-service/)
MODELS_DIR = Path(__file__).parent.parent / "models"


def download_f5_tts_model() -> Path:
    """Download F5-TTS model from HuggingFace."""
    from huggingface_hub import hf_hub_download

    print("Downloading F5-TTS model...")

    f5_dir = MODELS_DIR / "f5-tts"
    f5_dir.mkdir(parents=True, exist_ok=True)

    # F5-TTS model checkpoint is managed by Git LFS.
    # This script assumes the file is already present.
    ckpt_file = f5_dir / "F5TTS_v1_Base" / "model_48000.safetensors"
    if not ckpt_file.is_file():
        raise RuntimeError(
            f"Model checkpoint not found: {ckpt_file}. "
            "Please make sure Git LFS is installed and the file is checked out."
        )

    # Download vocab file
    vocab_file = hf_hub_download(
        repo_id="SWivid/F5-TTS",
        filename="F5TTS_v1_Base/vocab.txt",
        local_dir=str(f5_dir),
        local_dir_use_symlinks=False,
    )

    print(f"  Downloaded to: {f5_dir}")
    return f5_dir


def download_vocos_vocoder() -> Path:
    """Download Vocos vocoder from HuggingFace."""
    from huggingface_hub import snapshot_download

    print("Downloading Vocos vocoder...")

    vocos_dir = MODELS_DIR / "vocos-mel-24khz"

    snapshot_download(
        repo_id="charactr/vocos-mel-24khz",
        local_dir=str(vocos_dir),
        local_dir_use_symlinks=False,
    )

    print(f"  Downloaded to: {vocos_dir}")
    return vocos_dir


def main():
    """Download all models."""
    print("=" * 60)
    print("Downloading models for F5-TTS Service")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR.absolute()}")
    print()

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download all models
    download_f5_tts_model()
    print()

    download_vocos_vocoder()
    print()

    print("=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
