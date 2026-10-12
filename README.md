# ECE209AS_F26_2601
UCLA ECE 209AS, Fall 2026 - Team ID 2601.

## Local typing and knock demo

The microphone demo displays **Typing**, **Knock**, or **Neutral**, with adjustable
thresholds and a return-to-neutral delay. Audio is processed locally and is not saved.

```bash
cd sound-demo
uv sync --locked
uv run --locked python server.py
```

Open http://127.0.0.1:8765 and start the microphone. YAMNet downloads on first launch
and is cached locally; model weights are not committed.

See [sound-demo/README.md](sound-demo/README.md) for setup, offline use, and testing.
