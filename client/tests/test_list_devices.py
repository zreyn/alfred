"""Unit tests for list_devices.py module."""

import os
import sys
from unittest.mock import MagicMock, patch
from io import StringIO

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestListAudioDevices:
    """Test list_audio_devices function."""

    def test_lists_devices(self):
        """Test list_audio_devices lists all devices."""
        if "list_devices" in sys.modules:
            del sys.modules["list_devices"]

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {
                "name": "Test Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "Test Speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
        ]
        mock_sd.default = MagicMock()
        mock_sd.default.device = (0, 1)

        with patch.dict(
            "sys.modules",
            {
                "sounddevice": mock_sd,
            },
        ):
            from list_devices import list_audio_devices

            # Capture stdout
            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                list_audio_devices()
                output = mock_stdout.getvalue()

            assert "AUDIO DEVICES DEBUG" in output
            assert "Test Microphone" in output
            assert "Test Speaker" in output
            assert "Total devices found: 2" in output

    def test_shows_default_devices(self):
        """Test list_audio_devices shows default input/output devices."""
        if "list_devices" in sys.modules:
            del sys.modules["list_devices"]

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {
                "name": "Default Input",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "Default Output",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
        ]
        mock_sd.default = MagicMock()
        mock_sd.default.device = (0, 1)

        # query_devices with index returns device dict
        def mock_query(index=None):
            if index is None:
                return mock_sd.query_devices.return_value
            return mock_sd.query_devices.return_value[index]

        mock_sd.query_devices.side_effect = mock_query

        with patch.dict(
            "sys.modules",
            {
                "sounddevice": mock_sd,
            },
        ):
            from list_devices import list_audio_devices

            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                list_audio_devices()
                output = mock_stdout.getvalue()

            assert "Default input device: 0" in output
            assert "Default output device: 1" in output
            assert "Default input info: Default Input" in output
            assert "Default output info: Default Output" in output

    def test_handles_no_default_input(self):
        """Test list_audio_devices handles no default input device."""
        if "list_devices" in sys.modules:
            del sys.modules["list_devices"]

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = []
        mock_sd.default = MagicMock()
        mock_sd.default.device = (None, 1)  # No default input

        with patch.dict(
            "sys.modules",
            {
                "sounddevice": mock_sd,
            },
        ):
            from list_devices import list_audio_devices

            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                list_audio_devices()
                output = mock_stdout.getvalue()

            assert "Default input device: None" in output

    def test_handles_no_default_output(self):
        """Test list_audio_devices handles no default output device."""
        if "list_devices" in sys.modules:
            del sys.modules["list_devices"]

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = []
        mock_sd.default = MagicMock()
        mock_sd.default.device = (0, None)  # No default output

        with patch.dict(
            "sys.modules",
            {
                "sounddevice": mock_sd,
            },
        ):
            from list_devices import list_audio_devices

            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                list_audio_devices()
                output = mock_stdout.getvalue()

            assert "Default output device: None" in output

    def test_handles_exception(self):
        """Test list_audio_devices handles exceptions gracefully."""
        if "list_devices" in sys.modules:
            del sys.modules["list_devices"]

        mock_sd = MagicMock()
        mock_sd.query_devices.side_effect = RuntimeError("Device error")

        with patch.dict(
            "sys.modules",
            {
                "sounddevice": mock_sd,
            },
        ):
            from list_devices import list_audio_devices

            with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
                # Should not raise
                list_audio_devices()
                output = mock_stdout.getvalue()

            assert "Error listing audio devices: Device error" in output
