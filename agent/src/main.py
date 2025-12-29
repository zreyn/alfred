"""
Sir Henry - LiveKit Voice Agent

A voice AI agent using LiveKit Agents framework with custom F5-TTS and Faster-Whisper plugins.
"""

import logging
import time

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, JobContext, JobProcess
from livekit.plugins import openai as lk_openai
from livekit.plugins import silero

from config import (
    logger,
    SYSTEM_PROMPT,
    GREETING,
    STT_DEVICE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    PIPER_MODEL_PATH,
    PIPER_USE_CUDA,
    PIPER_SPEED,
    PIPER_VOLUME,
    PIPER_NOISE_SCALE,
    PIPER_NOISE_W,
)


# Load environment variables
load_dotenv()


class VoiceAgent(Agent):
    """The voice AI agent."""

    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)

    async def on_enter(self):
        """Called when the agent enters a session."""
        logger.info(f"Speaking greeting: '{GREETING}'")
        await self.session.say(GREETING)


def prewarm(proc: JobProcess):
    """
    Prewarm function to load models before the agent starts.
    This reduces latency when the first user connects.
    """
    # Import heavy plugins here to avoid multiprocessing spawn issues
    from plugins import FasterWhisperSTT, PiperTTS

    logger.info("Prewarming models...")

    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Silero VAD loaded.")

    proc.userdata["tts"] = PiperTTS(
        model_path=PIPER_MODEL_PATH,
        use_cuda=PIPER_USE_CUDA,
        speed=PIPER_SPEED,
        volume=PIPER_VOLUME,
        noise_scale=PIPER_NOISE_SCALE,
        noise_w=PIPER_NOISE_W,
    )
    logger.info("Piper TTS initialized.")
   

    proc.userdata["stt"] = FasterWhisperSTT(
        model_path="./models/faster-whisper-small",
        device=STT_DEVICE,
        language="en",  # force English to avoid misdetection
    )
    logger.info("Faster-Whisper STT initialized.")


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit agent.
    Called when a participant joins a room.
    """
    logger.info(f"Agent connecting to room: {ctx.room.name}")

    # Connect to the LiveKit room with auto_subscribe enabled
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    # Log track subscription events for debugging
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        logger.info(
            f"Agent subscribed to track {publication.sid} from {participant.identity} (kind: {track.kind})"
        )

    @ctx.room.on("track_published")
    def on_track_published(publication, participant):
        logger.info(
            f"Track published: {publication.sid} by {participant.identity} (kind: {publication.kind})"
        )

    # Log existing participants and their tracks
    logger.info(f"Remote participants in room: {len(ctx.room.remote_participants)}")
    for participant in ctx.room.remote_participants.values():
        logger.info(
            f"  - {participant.identity}: {len(participant.track_publications)} tracks"
        )
        for pub in participant.track_publications.values():
            logger.info(
                f"    - {pub.sid}: kind={pub.kind}, subscribed={pub.subscribed}"
            )

    # Create the agent session with our custom plugins
    session = AgentSession(
        stt=ctx.proc.userdata["stt"],
        tts=ctx.proc.userdata["tts"],
        vad=ctx.proc.userdata["vad"],
        llm=lk_openai.LLM.with_ollama(
            model=OLLAMA_MODEL,
            base_url=f"http://{OLLAMA_HOST}/v1",
            temperature=OLLAMA_TEMPERATURE,
        ),
    )

    # Round-trip latency tracking (STT done -> LLM + TTS start)
    _transcription_time: float | None = None

    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev) -> None:
        nonlocal _transcription_time
        _transcription_time = time.perf_counter()
        transcript = getattr(ev, "transcript", "")
        if transcript:
            logger.info(f"User said: {transcript[:80]}...")

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev) -> None:
        nonlocal _transcription_time
        if ev.new_state == "speaking" and _transcription_time is not None:
            latency_ms = (time.perf_counter() - _transcription_time) * 1000
            logger.info(f"ROUND-TRIP LATENCY: {latency_ms:.0f}ms (LLM + TTS)")
            _transcription_time = None

    # Start the session with our agent
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
    )

    logger.info("Agent session started.")


def main():
    """Run the LiveKit agent."""
    logger.info("Starting LiveKit Voice Agent...")

    # Create the agent server
    worker = agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)

    # Run the agent
    agents.cli.run_app(worker)


if __name__ == "__main__":
    main()
