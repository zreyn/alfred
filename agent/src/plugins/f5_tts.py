# plugins/f5_tts.py
import aiohttp
import logging
import uuid
from livekit.agents import tts, APIConnectOptions

logger = logging.getLogger(__name__)


class F5TTS(tts.TTS):
    def __init__(
        self,
        speed: float = 1.0,
        service_url: str = "http://localhost:11800",
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self.speed = speed
        self.service_url = service_url

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions | None = None
    ) -> tts.ChunkedStream:
        return _HTTPStream(self, text)


class _HTTPStream(tts.ChunkedStream):
    def __init__(self, tts_plugin: F5TTS, text: str):
        super().__init__(
            tts=tts_plugin, input_text=text, conn_options=APIConnectOptions()
        )
        self.plugin = tts_plugin

    async def _run(self, emitter: tts.AudioEmitter):
        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self.plugin.sample_rate,
            num_channels=self.plugin.num_channels,
            mime_type="audio/pcm",
        )

        async with aiohttp.ClientSession() as session:
            payload = {
                "text": self._input_text,
                "speed": self.plugin.speed,
            }

            try:
                async with session.post(
                    f"{self.plugin.service_url}/synthesize", json=payload
                ) as resp:
                    if resp.status == 200:
                        audio_data = await resp.read()
                        if audio_data:
                            emitter.push(audio_data)
                    else:
                        logger.error(
                            f"TTS Service Error: {resp.status} - {await resp.text()}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to TTS service: {e}")
