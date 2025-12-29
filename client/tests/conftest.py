"""Pytest configuration and fixtures."""

import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("LIVEKIT_URL", "ws://localhost:7880")
    monkeypatch.setenv("ROOM_NAME", "test-room")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-api-secret")


@pytest.fixture
def mock_livekit_rtc():
    """Mock livekit.rtc module."""
    mock_rtc = MagicMock()

    # Mock Room
    mock_room = MagicMock()
    mock_room.connect = AsyncMock()
    mock_room.disconnect = AsyncMock()
    mock_room.on = MagicMock(side_effect=lambda event: lambda f: f)
    mock_room.local_participant = MagicMock()
    mock_room.local_participant.publish_track = AsyncMock()
    mock_room.name = "test-room"
    mock_rtc.Room = MagicMock(return_value=mock_room)

    # Mock TrackKind
    mock_rtc.TrackKind = MagicMock()
    mock_rtc.TrackKind.KIND_AUDIO = "audio"

    # Mock TrackSource
    mock_rtc.TrackSource = MagicMock()
    mock_rtc.TrackSource.SOURCE_MICROPHONE = 2

    # Mock TrackPublishOptions
    mock_rtc.TrackPublishOptions = MagicMock(return_value=MagicMock())

    # Mock AudioSource
    mock_audio_source = MagicMock()
    mock_audio_source.capture_frame = AsyncMock()
    mock_audio_source.aclose = AsyncMock()
    mock_rtc.AudioSource = MagicMock(return_value=mock_audio_source)

    # Mock AudioFrame
    mock_rtc.AudioFrame = MagicMock()

    # Mock AudioStream - yields AudioFrameEvent objects with .frame attribute
    class _MockAudioStream:
        def __init__(self, track, sample_rate=48000, num_channels=1):
            self.track = track
            self.sample_rate = sample_rate
            self.num_channels = num_channels

        def __aiter__(self):
            async def gen():
                mock_frame_event = MagicMock()
                mock_frame_event.frame = MagicMock()
                mock_frame_event.frame.data = MagicMock()
                mock_frame_event.frame.data.tobytes = MagicMock(
                    return_value=b"\x00\x00" * 960
                )
                yield mock_frame_event

            return gen()

    mock_rtc.AudioStream = MagicMock(side_effect=_MockAudioStream)

    # Mock LocalAudioTrack
    mock_audio_track = MagicMock()
    mock_audio_track.sid = "TR_test"
    mock_rtc.LocalAudioTrack = MagicMock()
    mock_rtc.LocalAudioTrack.create_audio_track = MagicMock(
        return_value=mock_audio_track
    )

    # Mock RemoteParticipant
    mock_rtc.RemoteParticipant = MagicMock()
    mock_rtc.RemoteTrackPublication = MagicMock()
    mock_rtc.Track = MagicMock()

    return mock_rtc


@pytest.fixture
def mock_livekit_apm():
    """Mock livekit.rtc.apm module."""
    mock_apm = MagicMock()
    mock_processor = MagicMock()
    mock_processor.process_stream = MagicMock()
    mock_processor.process_reverse_stream = MagicMock()
    mock_processor.set_stream_delay_ms = MagicMock()
    mock_apm.AudioProcessingModule = MagicMock(return_value=mock_processor)
    return mock_apm


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module."""
    mock_sd = MagicMock()

    # Mock InputStream
    mock_input_stream = MagicMock()
    mock_input_stream.active = True
    mock_sd.InputStream = MagicMock(return_value=mock_input_stream)

    # Mock OutputStream
    mock_output_stream = MagicMock()
    mock_output_stream.active = True
    mock_sd.OutputStream = MagicMock(return_value=mock_output_stream)

    # Mock RawOutputStream (for compatibility)
    mock_sd.RawOutputStream = MagicMock(return_value=mock_output_stream)

    # Mock default device
    mock_sd.default = MagicMock()
    mock_sd.default.device = (0, 1)  # (input_device, output_device)

    # Mock query_devices
    mock_sd.query_devices = MagicMock(
        return_value={
            "name": "Test Microphone",
            "max_input_channels": 2,
            "default_samplerate": 48000,
        }
    )

    return mock_sd


@pytest.fixture
def mock_livekit_api():
    """Mock livekit.api module."""
    mock_api = MagicMock()
    mock_api.AccessToken = MagicMock()
    mock_api.RoomAgentDispatch = MagicMock()
    mock_api.RoomConfiguration = MagicMock()
    mock_api.VideoGrants = MagicMock()
    return mock_api


@pytest.fixture
def mock_auth():
    """Mock auth module."""
    mock = MagicMock()
    mock.generate_token = MagicMock(return_value="test-token")
    return mock


@pytest.fixture
def mock_list_devices():
    """Mock list_devices module."""
    mock = MagicMock()
    mock.list_audio_devices = MagicMock()
    return mock
