"""Unit tests for client main.py module."""

import asyncio
import sys
import time
import threading
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest
import numpy as np


class TestHelperFunctions:
    """Test helper functions."""

    def test_esc_generates_escape_codes(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _esc generates proper ANSI escape codes."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import _esc

            assert _esc(31) == "\033[31m"
            assert _esc(1, 32) == "\033[1;32m"
            assert _esc(0) == "\033[0m"

    def test_normalize_db_clamps_values(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _normalize_db clamps and normalizes dB values."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import _normalize_db

            # Test normalization
            assert _normalize_db(-35.0, -70.0, 0.0) == 0.5  # Midpoint
            assert _normalize_db(-70.0, -70.0, 0.0) == 0.0  # Min
            assert _normalize_db(0.0, -70.0, 0.0) == 1.0  # Max

            # Test clamping
            assert _normalize_db(-100.0, -70.0, 0.0) == 0.0  # Below min
            assert _normalize_db(10.0, -70.0, 0.0) == 1.0  # Above max


class TestAudioStreamer:
    """Test AudioStreamer class."""

    def test_initialization(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test AudioStreamer initializes correctly."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=True)

            assert streamer.running is True
            assert streamer.is_muted is False
            assert streamer.enable_aec is True
            assert streamer.audio_processor is not None

    def test_initialization_without_aec(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test AudioStreamer initializes without AEC."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)

            assert streamer.enable_aec is False
            assert streamer.audio_processor is None

    def test_toggle_mute(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test toggle_mute toggles mute state."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)

            assert streamer.is_muted is False
            streamer.toggle_mute()
            assert streamer.is_muted is True
            streamer.toggle_mute()
            assert streamer.is_muted is False

    def test_start_audio_devices(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test start_audio_devices creates input and output streams."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            streamer.start_audio_devices()

            # Verify streams were created
            mock_sounddevice.InputStream.assert_called_once()
            mock_sounddevice.OutputStream.assert_called_once()

            # Verify streams were started
            mock_sounddevice.InputStream.return_value.start.assert_called_once()
            mock_sounddevice.OutputStream.return_value.start.assert_called_once()

    def test_stop_audio_devices(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test stop_audio_devices stops and closes streams."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            streamer.start_audio_devices()
            streamer.stop_audio_devices()

            # Verify streams were stopped and closed
            mock_input = mock_sounddevice.InputStream.return_value
            mock_output = mock_sounddevice.OutputStream.return_value

            mock_input.stop.assert_called_once()
            mock_input.close.assert_called_once()
            mock_output.stop.assert_called_once()
            mock_output.close.assert_called_once()

    def test_input_callback_processes_audio(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _input_callback processes microphone audio."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, FRAME_SAMPLES, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=False, loop=loop)

            # Create test audio data (must be correct shape for BLOCKSIZE)
            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            indata[:, 0] = 1000  # Some audio level

            # Create time_info mock
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            # Call the callback
            initial_count = streamer.input_callback_count
            streamer._input_callback(indata, BLOCKSIZE, time_info, None)

            # Verify callback was processed
            assert streamer.input_callback_count == initial_count + 1
            assert streamer.frames_processed > 0

            loop.close()

    def test_input_callback_handles_mute(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _input_callback respects mute state."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=False, loop=loop)
            streamer.is_muted = True

            # Create test audio data
            indata = np.ones((BLOCKSIZE, 1), dtype=np.int16) * 1000

            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            # Call the callback while muted
            streamer._input_callback(indata, BLOCKSIZE, time_info, None)

            # Should still process (for meter) but send silence
            assert streamer.frames_processed > 0

            loop.close()

    def test_output_callback_plays_audio(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _output_callback plays received audio."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            streamer = AudioStreamer(enable_aec=False)

            # Add some audio to the output buffer
            test_audio = b"\x00\x10" * BLOCKSIZE  # 2 bytes per sample
            streamer.output_buffer.extend(test_audio)

            # Create output array
            outdata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)

            time_info = MagicMock()
            time_info.outputBufferDacTime = 0.0
            time_info.currentTime = 0.0

            # Call the callback
            streamer._output_callback(outdata, BLOCKSIZE, time_info, None)

            # Buffer should be consumed
            assert len(streamer.output_buffer) == 0
            assert streamer.output_callback_count == 1

    def test_output_callback_handles_empty_buffer(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _output_callback handles empty buffer gracefully."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            streamer = AudioStreamer(enable_aec=False)

            # Output buffer is empty
            outdata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)

            time_info = MagicMock()
            time_info.outputBufferDacTime = 0.0
            time_info.currentTime = 0.0

            # Should not raise
            streamer._output_callback(outdata, BLOCKSIZE, time_info, None)

            # Output should be zeros (silence)
            assert np.all(outdata == 0)

    def test_print_audio_meter(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test print_audio_meter generates output."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            streamer.micro_db = -30.0

            # Should not raise
            with patch("sys.stdout") as mock_stdout:
                streamer.print_audio_meter()

    def test_init_and_restore_terminal(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test init_terminal and restore_terminal."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)

            with patch("sys.stdout") as mock_stdout:
                streamer.init_terminal()
                streamer.restore_terminal()


class TestParticipantTracking:
    """Test participant tracking functionality."""

    def test_participant_added_on_connect(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test participants are tracked when they connect."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)

            # Simulate adding a participant
            participant_id = "test-participant-123"
            participant_name = "Test User"

            with streamer.participants_lock:
                streamer.participants[participant_id] = {
                    "name": participant_name,
                    "db_level": -40.0,
                    "last_update": time.time(),
                }

            assert participant_id in streamer.participants
            assert streamer.participants[participant_id]["name"] == participant_name

    def test_participant_removed_on_disconnect(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test participants are removed when they disconnect."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)

            # Add then remove a participant
            participant_id = "test-participant-123"

            with streamer.participants_lock:
                streamer.participants[participant_id] = {
                    "name": "Test User",
                    "db_level": -40.0,
                    "last_update": time.time(),
                }

            with streamer.participants_lock:
                del streamer.participants[participant_id]

            assert participant_id not in streamer.participants


class TestMainFunction:
    """Test main async function."""

    @pytest.mark.asyncio
    async def test_main_connects_to_room(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test main function connects to LiveKit room."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
                "termios": MagicMock(),
                "tty": MagicMock(),
                "select": MagicMock(select=MagicMock(return_value=([], [], []))),
            },
        ):
            from main import main

            # Make the main loop exit quickly
            call_count = 0

            async def mock_sleep(duration):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise KeyboardInterrupt()
                await asyncio.sleep(0)

            with patch("asyncio.sleep", side_effect=mock_sleep):
                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                    try:
                        await main("test-user", enable_aec=False)
                    except KeyboardInterrupt:
                        pass

            mock_room_instance.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_publishes_track(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test main function publishes microphone track."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
                "termios": MagicMock(),
                "tty": MagicMock(),
                "select": MagicMock(select=MagicMock(return_value=([], [], []))),
            },
        ):
            from main import main

            call_count = 0

            async def mock_sleep(duration):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise KeyboardInterrupt()
                await asyncio.sleep(0)

            with patch("asyncio.sleep", side_effect=mock_sleep):
                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                    try:
                        await main("test-user", enable_aec=False)
                    except KeyboardInterrupt:
                        pass

            # Verify track was published
            mock_room_instance.local_participant.publish_track.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_disconnects_on_interrupt(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test main function disconnects on KeyboardInterrupt."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_room_instance = mock_livekit_rtc.Room.return_value

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
                "termios": MagicMock(),
                "tty": MagicMock(),
                "select": MagicMock(select=MagicMock(return_value=([], [], []))),
            },
        ):
            from main import main

            call_count = 0

            async def mock_sleep(duration):
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise KeyboardInterrupt()
                await asyncio.sleep(0)

            with patch("asyncio.sleep", side_effect=mock_sleep):
                with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                    try:
                        await main("test-user", enable_aec=False)
                    except KeyboardInterrupt:
                        pass

            mock_room_instance.disconnect.assert_called_once()


class TestConfigurationConstants:
    """Test configuration constants."""

    def test_constants_defined(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test required constants are defined."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import SAMPLE_RATE, NUM_CHANNELS, FRAME_SAMPLES, BLOCKSIZE

            assert SAMPLE_RATE == 48000
            assert NUM_CHANNELS == 1
            assert FRAME_SAMPLES == 480  # 10ms at 48kHz
            assert BLOCKSIZE == 4800  # 100ms buffer


class TestInputCallbackEdgeCases:
    """Test edge cases in _input_callback."""

    def test_input_callback_logs_status(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _input_callback logs status warnings."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=False, loop=loop)

            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            # Call with status
            streamer._input_callback(indata, BLOCKSIZE, time_info, "input overflow")

            assert streamer.input_callback_count == 1
            loop.close()

    def test_input_callback_not_running(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _input_callback returns early when not running."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=False, loop=loop)
            streamer.running = False

            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            initial_frames = streamer.frames_processed
            streamer._input_callback(indata, BLOCKSIZE, time_info, None)

            # Should not process when not running
            assert streamer.frames_processed == initial_frames
            loop.close()

    def test_input_callback_with_aec(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _input_callback with AEC enabled."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=True, loop=loop)

            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            indata[:, 0] = 1000
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            streamer._input_callback(indata, BLOCKSIZE, time_info, None)

            # AEC should have been called
            assert streamer.audio_processor.process_stream.called
            loop.close()

    def test_input_callback_aec_delay_error(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _input_callback handles AEC delay error."""
        if "main" in sys.modules:
            del sys.modules["main"]

        # Make set_stream_delay_ms raise an error
        mock_livekit_apm.AudioProcessingModule.return_value.set_stream_delay_ms.side_effect = RuntimeError(
            "Delay error"
        )

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=True, loop=loop)

            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            # Should not raise, error is caught
            streamer._input_callback(indata, BLOCKSIZE, time_info, None)
            assert streamer.input_callback_count == 1
            loop.close()


class TestOutputCallbackEdgeCases:
    """Test edge cases in _output_callback."""

    def test_output_callback_logs_status(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _output_callback logs status warnings."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            streamer = AudioStreamer(enable_aec=False)
            outdata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.outputBufferDacTime = 0.0
            time_info.currentTime = 0.0

            # Call with status
            streamer._output_callback(outdata, BLOCKSIZE, time_info, "output underflow")
            assert streamer.output_callback_count == 1

    def test_output_callback_not_running(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _output_callback fills zeros when not running."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            streamer = AudioStreamer(enable_aec=False)
            streamer.running = False

            # Add data to buffer
            streamer.output_buffer.extend(b"\x00\x10" * BLOCKSIZE)

            outdata = np.ones((BLOCKSIZE, 1), dtype=np.int16) * 1000
            time_info = MagicMock()
            time_info.outputBufferDacTime = 0.0
            time_info.currentTime = 0.0

            streamer._output_callback(outdata, BLOCKSIZE, time_info, None)

            # Should fill with zeros when not running
            assert np.all(outdata == 0)

    def test_output_callback_partial_buffer(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _output_callback handles partial buffer."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            streamer = AudioStreamer(enable_aec=False)

            # Add partial data (less than BLOCKSIZE)
            partial_samples = 100
            streamer.output_buffer.extend(b"\x00\x10" * partial_samples)

            outdata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.outputBufferDacTime = 0.0
            time_info.currentTime = 0.0

            streamer._output_callback(outdata, BLOCKSIZE, time_info, None)

            # Buffer should be empty now
            assert len(streamer.output_buffer) == 0

    def test_output_callback_with_aec(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _output_callback with AEC enabled."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            streamer = AudioStreamer(enable_aec=True)

            # Add enough data
            streamer.output_buffer.extend(b"\x00\x10" * BLOCKSIZE)

            outdata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.outputBufferDacTime = 0.0
            time_info.currentTime = 0.0

            streamer._output_callback(outdata, BLOCKSIZE, time_info, None)

            # AEC reverse stream should have been processed
            assert streamer.audio_processor.process_reverse_stream.called


class TestMeterDisplay:
    """Test meter display functionality."""

    def test_meter_not_running(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test print_audio_meter does nothing when meter not running."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            streamer.meter_running = False

            with patch("sys.stdout") as mock_stdout:
                streamer.print_audio_meter()
                # Should not write to stdout when meter not running
                mock_stdout.write.assert_not_called()

    def test_simple_meter_with_mute(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _print_simple_meter shows muted state."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            streamer.is_muted = True
            streamer.micro_db = -30.0

            with patch("sys.stdout"):
                streamer._print_simple_meter()

    def test_simple_meter_with_participants(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _print_simple_meter shows participant meters."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            streamer.participants["test-123"] = {
                "name": "Test User",
                "db_level": -30.0,
                "last_update": time.time(),
            }

            with patch("sys.stdout"):
                streamer._print_simple_meter()

    def test_simple_meter_removes_stale_participants(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _print_simple_meter removes stale participants."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            # Add stale participant (last update > 5 seconds ago)
            streamer.participants["stale-123"] = {
                "name": "Stale User",
                "db_level": -30.0,
                "last_update": time.time() - 10.0,  # 10 seconds ago
            }

            with patch("sys.stdout"):
                streamer._print_simple_meter()

            # Stale participant should be removed
            assert "stale-123" not in streamer.participants


class TestStartAudioDevicesErrors:
    """Test error handling in start_audio_devices."""

    def test_handles_start_error(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test start_audio_devices handles errors."""
        if "main" in sys.modules:
            del sys.modules["main"]

        mock_sounddevice.InputStream.side_effect = RuntimeError("No input device")

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)

            with pytest.raises(RuntimeError):
                streamer.start_audio_devices()


class TestKeyboardHandler:
    """Test keyboard handler functionality."""

    def test_stop_keyboard_handler(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test stop_keyboard_handler."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            # Should not raise even if keyboard_thread is None
            streamer.stop_keyboard_handler()


class TestInputCallbackDebugLogging:
    """Test debug logging in input callback."""

    def test_input_callback_debug_logging_after_5_seconds(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test debug logging triggers after 5 seconds."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=False, loop=loop)

            # Set last_debug_time to more than 5 seconds ago
            streamer.last_debug_time = time.time() - 6.0

            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            streamer._input_callback(indata, BLOCKSIZE, time_info, None)

            # After callback, last_debug_time should be updated
            assert time.time() - streamer.last_debug_time < 1.0
            loop.close()


class TestInputDeviceChannelWarning:
    """Test input device channel warning."""

    def test_warns_on_insufficient_channels(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test warning when input device has fewer channels than needed."""
        if "main" in sys.modules:
            del sys.modules["main"]

        # Mock query_devices to return device with 0 input channels
        mock_sounddevice.query_devices = MagicMock(
            return_value={
                "name": "Low Channel Device",
                "max_input_channels": 0,  # Less than NUM_CHANNELS (1)
                "default_samplerate": 48000,
            }
        )

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            # This should log a warning but not crash
            streamer.start_audio_devices()


class TestQueueExceptionHandling:
    """Test exception handling when queueing audio frames."""

    def test_handles_queue_exception(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test handling of exception when queueing audio frame."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=False, loop=loop)

            # Make call_soon_threadsafe raise an exception
            loop.call_soon_threadsafe = MagicMock(side_effect=RuntimeError("Queue error"))

            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            # Should not raise
            streamer._input_callback(indata, BLOCKSIZE, time_info, None)
            loop.close()

    def test_handles_no_loop(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test handling when no event loop is available."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            # Create streamer without a loop
            streamer = AudioStreamer(enable_aec=False, loop=None)

            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            # Should not raise even without a loop
            streamer._input_callback(indata, BLOCKSIZE, time_info, None)


class TestAECReverseStreamException:
    """Test AEC reverse stream exception handling."""

    def test_handles_reverse_stream_exception(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test handling of exception in process_reverse_stream."""
        if "main" in sys.modules:
            del sys.modules["main"]

        # Make process_reverse_stream raise an exception
        mock_livekit_apm.AudioProcessingModule.return_value.process_reverse_stream.side_effect = RuntimeError(
            "Reverse stream error"
        )

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            streamer = AudioStreamer(enable_aec=True)

            # Add enough data to buffer
            streamer.output_buffer.extend(b"\x00\x10" * BLOCKSIZE)

            outdata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.outputBufferDacTime = 0.0
            time_info.currentTime = 0.0

            # Should not raise
            streamer._output_callback(outdata, BLOCKSIZE, time_info, None)


class TestSimpleMeterNotRunning:
    """Test _print_simple_meter when not running."""

    def test_simple_meter_returns_when_not_running(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test _print_simple_meter returns early when not running."""
        if "main" in sys.modules:
            del sys.modules["main"]

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer

            streamer = AudioStreamer(enable_aec=False)
            streamer.meter_running = False

            with patch("sys.stdout") as mock_stdout:
                # Call _print_simple_meter directly
                streamer._print_simple_meter()
                # Should not write anything to stdout
                mock_stdout.write.assert_not_called()


class TestAECProcessStreamException:
    """Test AEC process_stream exception handling."""

    def test_handles_process_stream_exception(
        self,
        mock_env_vars,
        mock_livekit_rtc,
        mock_sounddevice,
        mock_livekit_apm,
        mock_livekit_api,
        mock_auth,
        mock_list_devices,
    ):
        """Test handling of exception in process_stream."""
        if "main" in sys.modules:
            del sys.modules["main"]

        # Make process_stream raise an exception
        mock_livekit_apm.AudioProcessingModule.return_value.process_stream.side_effect = RuntimeError(
            "Process stream error"
        )

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(rtc=mock_livekit_rtc, api=mock_livekit_api),
                "livekit.rtc": mock_livekit_rtc,
                "livekit.rtc.apm": mock_livekit_apm,
                "livekit.api": mock_livekit_api,
                "dotenv": MagicMock(),
                "sounddevice": mock_sounddevice,
                "numpy": np,
                "auth": mock_auth,
                "list_devices": mock_list_devices,
            },
        ):
            from main import AudioStreamer, BLOCKSIZE

            loop = asyncio.new_event_loop()
            streamer = AudioStreamer(enable_aec=True, loop=loop)

            indata = np.zeros((BLOCKSIZE, 1), dtype=np.int16)
            time_info = MagicMock()
            time_info.currentTime = 0.0
            time_info.inputBufferAdcTime = 0.0

            # Should not raise
            streamer._input_callback(indata, BLOCKSIZE, time_info, None)
            loop.close()
