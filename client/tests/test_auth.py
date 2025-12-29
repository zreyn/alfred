"""Unit tests for auth.py module."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestGenerateToken:
    """Test generate_token function."""

    def test_generates_token_with_identity(self):
        """Test generate_token returns a JWT token."""
        if "auth" in sys.modules:
            del sys.modules["auth"]

        mock_api = MagicMock()
        mock_access_token = MagicMock()
        mock_access_token.with_identity.return_value = mock_access_token
        mock_access_token.with_name.return_value = mock_access_token
        mock_access_token.with_grants.return_value = mock_access_token
        mock_access_token.to_jwt.return_value = "test-jwt-token"
        mock_api.AccessToken = MagicMock(return_value=mock_access_token)
        mock_api.VideoGrants = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(api=mock_api),
                "livekit.api": mock_api,
                "dotenv": MagicMock(),
            },
        ):
            with patch.dict(
                "os.environ",
                {
                    "LIVEKIT_API_KEY": "test-api-key",
                    "LIVEKIT_API_SECRET": "test-api-secret",
                },
            ):
                from auth import generate_token

                token = generate_token(
                    "test-room", identity="test-user", name="Test User"
                )

                assert token == "test-jwt-token"
                mock_access_token.with_identity.assert_called_once_with("test-user")
                mock_access_token.with_name.assert_called_once_with("Test User")

    def test_generates_default_identity(self):
        """Test generate_token uses default identity when not provided."""
        if "auth" in sys.modules:
            del sys.modules["auth"]

        mock_api = MagicMock()
        mock_access_token = MagicMock()
        mock_access_token.with_identity.return_value = mock_access_token
        mock_access_token.with_name.return_value = mock_access_token
        mock_access_token.with_grants.return_value = mock_access_token
        mock_access_token.to_jwt.return_value = "test-jwt-token"
        mock_api.AccessToken = MagicMock(return_value=mock_access_token)
        mock_api.VideoGrants = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(api=mock_api),
                "livekit.api": mock_api,
                "dotenv": MagicMock(),
            },
        ):
            with patch.dict(
                "os.environ",
                {
                    "LIVEKIT_API_KEY": "test-api-key",
                    "LIVEKIT_API_SECRET": "test-api-secret",
                },
            ):
                from auth import generate_token

                token = generate_token("my-room")

                mock_access_token.with_identity.assert_called_once_with(
                    "python-user-my-room"
                )
                mock_access_token.with_name.assert_called_once_with(
                    "python-user-my-room"
                )

    def test_raises_without_credentials(self):
        """Test generate_token raises when credentials are missing."""
        if "auth" in sys.modules:
            del sys.modules["auth"]

        mock_api = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "livekit": MagicMock(api=mock_api),
                "livekit.api": mock_api,
                "dotenv": MagicMock(),
            },
        ):
            with patch.dict("os.environ", {}, clear=True):
                from auth import generate_token

                with pytest.raises(
                    ValueError,
                    match="LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set",
                ):
                    generate_token("test-room")
