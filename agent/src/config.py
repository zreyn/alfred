import logging
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from speech_utils import format_date_speech_friendly, format_time_speech_friendly


# Configure logging (LiveKit CLI reconfigures root logger, so set our level explicitly)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)

# Suppress verbose logs from dependencies
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)  # MCP client SSE/JSON-RPC spam
logging.getLogger("livekit").setLevel(logging.WARNING)  # LiveKit internal logs
logging.getLogger("livekit_api").setLevel(logging.WARNING)  # Rust bridge logs

# LiveKit Configuration
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")

# STT Device config
STT_DEVICE = os.environ.get("STT_DEVICE", "cpu").lower()

# Ollama Configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.8"))

# TTS config
TTS_HOST = os.environ.get("TTS_HOST", "http://localhost:11800")
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))

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
