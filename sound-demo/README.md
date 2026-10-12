# Local typing and knock detector

A pretrained YAMNet baseline with exactly three displayed states: **Typing**, **Knock**,
and **Neutral**. No training data is required to run it. This is a separate uv project
from the course's desk-vibration app.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
cd sound-demo
uv sync --locked
uv run --locked python server.py
```

Open **http://127.0.0.1:8765** in Chrome or Safari. Click **Start microphone** and allow
microphone access. Use **Stop** to release it. Press Ctrl+C in the terminal to stop
the server. Python 3.11 is selected by `.python-version`; uv can download it if needed.

Tested on Apple Silicon macOS with Chrome. Other systems need compatible TensorFlow
wheels and a browser with microphone support; they have not been tested here.

## Model download and offline use

**Do not commit the model weights.** The first run downloads the versioned model
`https://tfhub.dev/google/yamnet/1` via TensorFlow Hub, then caches it in
`sound-demo/.cache/yamnet/`. Later runs reuse the local cache. The model is about
17 MB extracted; the TensorFlow Python environment is substantially larger.

You can prepare the cache before a demo:

```bash
uv run --locked python server.py --download-only
```

Once dependencies, Python, and the model are cached, inference can run offline.
For an offline machine, copy the cache folder from another machine, or point to
an already extracted YAMNet SavedModel:

```bash
uv run --locked python server.py --model-path /path/to/yamnet
```

Git tracks source files, `pyproject.toml`, and `uv.lock`; `.venv/`, `.cache/`, and audio
recordings are excluded. `uv.lock` pins dependencies, not the downloaded model's
bytes. The model handle is versioned; an archived copy with a checksum would be
appropriate if an experiment needs an immutable model artifact.

## Testing and thresholds

1. Remain quiet or speak: expect Neutral.
2. Type for 1–2 seconds, then pause for two seconds: expect Typing, then Neutral.
3. Knock on the table, then pause: expect Knock, then Neutral.
4. Repeat at different distances and test confusing background sounds.

The initial threshold for each target is **0.35**, an uncalibrated starting value,
not an accuracy guarantee. Raise it to reduce false triggers; lower it if events
are missed. If both targets pass their thresholds, the higher score wins.
The higher of the Typing and Computer keyboard scores triggers Typing.
Tap is shown for diagnosis only; Knock uses the Knock score. The return-to-neutral
delay defaults to 600 ms and can be adjusted from 0 to 1500 ms to bridge brief gaps
between detections.

The model processes overlapping one-second windows about four times per second,
subject to inference speed. Brief detections can remain visible until the event
leaves the window. History merges nearby detections heuristically and is not an
accurate event counter. YAMNet's general Knock class does not guarantee performance
on a particular table. Measure false triggers per hour and missed events on the
intended microphone before relying on this baseline.

Initial checks recognized keyboard-typing and door-knock fixtures while silence
stayed below both target thresholds. These checks do not establish real-world accuracy.

## Privacy and configuration

The browser sends raw microphone audio only to the Python server on **127.0.0.1**.
Audio is held in memory, not saved or sent to a remote inference service. The initial
dependency and model downloads contact external services. The page uses no CDN.

To choose another local port:

```bash
uv run --locked python server.py --port 8766
```

The server binds only to loopback. Keep it local; this development server is not
designed for public deployment.

## References

- [YAMNet example and input requirements](https://www.tensorflow.org/hub/tutorials/yamnet)
- [YAMNet model documentation and license](https://tfhub.dev/google/yamnet/1)
- [TensorFlow Hub caching](https://www.tensorflow.org/hub/caching)

The repository's source license does not replace the upstream model's license.
