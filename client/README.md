```
cd client
uv sync --extra dev
export LIVEKIT_URL="ws://<server>:7880"
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secret
uv run python src/main.py -n "human-user"
```