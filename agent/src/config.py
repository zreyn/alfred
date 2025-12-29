import logging
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from speech_utils import format_date_speech_friendly, format_time_speech_friendly


def _setup_logging():
    """Configures the root logger with a single handler."""
    global _logging_setup_complete
    if _logging_setup_complete:
        return

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Create a single handler (e.g., StreamHandler for console output)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    # for name in [
    #     "livekit",
    #     "livekit.agents",
    #     "livekit.rtc",
    #     "faster_whisper",
    #     "asyncio",
    #     "httpx",
    #     "httpcore",
    # ]:
    #     lib_logger = logging.getLogger(name)
    #     lib_logger.handlers.clear()
    #     lib_logger.propagate = False
    #     lib_logger.addHandler(logging.NullHandler())

    # Monkey-patch to block ALL handler additions after setup
    _original_addHandler = logging.Logger.addHandler

    def _guarded_addHandler(self, handler):
        # Only allow NullHandler (used for silencing)
        if isinstance(handler, logging.NullHandler):
            _original_addHandler(self, handler)
        # Block everything else after setup

    logging.Logger.addHandler = _guarded_addHandler
    _logging_setup_complete = True




_logging_setup_complete = False
_setup_logging()
logger = logging.getLogger("agent")


# LiveKit Configuration
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")

TTS_HOST = os.environ.get("TTS_HOST", "http://localhost:11800")

# STT Device config
STT_DEVICE = os.environ.get("STT_DEVICE", "cpu").lower()

# Ollama Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.8"))

# TTS (Piper) config
PIPER_MODEL_PATH =  os.environ.get("PIPER_MODEL_PATH", "./models/piper/en/en_US/ryan/high/en_US-ryan-high.onnx")
PIPER_USE_CUDA =  os.environ.get("PIPER_USE_CUDA", False)
PIPER_SPEED =  os.environ.get("PIPER_SPEED", 1.0)
PIPER_VOLUME =  os.environ.get("PIPER_VOLUME", 1.0)
PIPER_NOISE_SCALE =  os.environ.get("PIPER_NOISE_SCALE", 0.667)
PIPER_NOISE_W =  os.environ.get("PIPER_NOISE_W", 0.8)

# Customization config
greeting_path = os.environ.get("GREETING_PATH", "./prompt/greeting.md")
GREETING = Path(greeting_path).read_text()

TIMEZONE_ID = os.environ.get("TIMEZONE_ID", "America/New_York")
TIMEZONE_DISPLAY = os.environ.get("TIMEZONE_DISPLAY", "US Eastern Time")

system_prompt_path = os.environ.get("SYSTEM_PROMPT_PATH", "./prompt/system.md")
system_prompt_template = Path(system_prompt_path).read_text()

now = datetime.now(ZoneInfo(TIMEZONE_ID))
date_context = (
    f"Today is {format_date_speech_friendly(now)}. "
    f"The current time is {format_time_speech_friendly(now)} {TIMEZONE_DISPLAY}."
)

prompt = system_prompt_template.replace("{{CURRENT_DATE_CONTEXT}}", date_context)
SYSTEM_PROMPT = prompt.replace("{{TIMEZONE}}", TIMEZONE_DISPLAY)
