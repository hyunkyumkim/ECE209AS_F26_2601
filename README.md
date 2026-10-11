# ECE209AS_F26_2601

UCLA ECE 209AS, Fall 2026 - Team ID 2601.

## Authors

- Hyun Kyum Kim
- Spike Rao
- Rahul Tripathi

## Desk Vibration Lab

A laptop-only vibration bakeoff project. Using only the laptop's built-in microphone, with the laptop resting on a desk, the app tells apart three classes:

- **Neutral**: nothing happening (background noise)
- **Knock**: a knuckle knock on the desk
- **Scratch**: a fingernail scratch on the desk

It computes an FFT of the microphone signal, classifies each frame with a linear SVM, and smooths the result so a single stray frame does not count as an interaction. It is built on the course vibration-interaction starter.

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (`winget install --id=astral-sh.uv -e` on Windows), then from this folder:

```bash
uv sync
uv run bakeoff_app.py
```

The environment uses Python 3.10 (see `.python-version`). Newer Python versions may not have wheels for `dearpygui` or `pyaudio`.

## Before you record

- In Windows Sound settings, turn off microphone enhancements, noise suppression, and auto-gain for the input you use.
- Pick the laptop microphone from the dropdown at the top of the app.
- Keep the laptop in the same spot on the desk for recording, training, and the demo.

## Workflow

The right-hand panel has three tabs.

### 1 Collect

1. Record **Neutral** first, for a few seconds, with the background you expect: typing, talking, chair movement. The quiet-frame gate for the other classes is calibrated from it.
2. Record **Knock**: press Record, knock, pause as long as you like, knock again, then press Stop.
3. Record **Scratch** the same way. Vary stroke length, speed, and pressure.

Notes:

- Neutral keeps every frame. Knock and Scratch keep only frames louder than the **quiet gate**, so pauses are not labeled as events. The yellow line on the loudness history plot shows the gate. The "gate" slider overrides it (0 = automatic).
- Each Record/Stop session adds to the class. Use **Clear** to discard a class's frames.
- Aim for the progress bar to fill (300 frames per class), with data from several people if possible.
- Recorded frames are saved to `recordings/<dataset name>/` (autosave when a recording stops, or press **Save**). The last dataset reopens at startup. Pick another dataset and press **Load**, optionally adding it to the current data.

### 2 Train

Choose a model and press **Train Model**. The default **LinearSVC** is the one that can be exported to C++ with `export_model_header.py`. **SVC poly degree 1** is available for comparison. The holdout accuracy shown is optimistic because neighboring frames are similar, so rely on the bakeoff trials instead.

**Save Model** and **Load Model** use the standard Windows dialogs and start in `models/`.

### 3 Bakeoff

Follows the class bakeoff procedure:

1. Press **space** to start a trial. The app records every smoothed label it sees.
2. The participant performs one interaction.
3. Press **space** again. The app prints the set of labels, for example `Neutral, Knock`.

Pass conditions, with Neutral always allowed in the set:

| Trial | Participant does | Passes if the set |
| --- | --- | --- |
| Positive Knock | a knock | contains Knock and not Scratch |
| Positive Scratch | a scratch | contains Scratch and not Knock |
| Negative Knock | anything but a knock | does not contain Knock |
| Negative Scratch | anything but a scratch | does not contain Scratch |

For self-testing, choose what the participant did, press **Score Trial**, and the tally keeps pass/fail counts per trial type.

## How it works

1. `src/stream_reader.py` captures mono audio and `src/stream_analyzer.py` computes the FFT.
2. `preprocess_data()` in `models.py` log-scales and L2-normalizes the spectrum, so hit strength matters less, and appends one loudness feature, so quiet background stays separable.
3. A scikit-learn linear SVM (balanced class weights) predicts a class for each fresh frame.
4. `PredictionSmoother` in `bakeoff.py` emits an interaction only when it fills at least 3 of the last 6 frames.

## Tuning

| Setting | File | Effect |
| --- | --- | --- |
| `SMOOTH_WINDOW`, `SMOOTH_MIN_VOTES` | `bakeoff.py` | Higher votes: fewer false positives, more missed short knocks |
| `GATE_MARGIN`, `GATE_FALLBACK_MARGIN` | `bakeoff_app.py` | How far above background a frame must be to be kept |
| `FRAMES_GOAL` | `bakeoff_app.py` | Target frames per class on the progress bar |
| `LOUDNESS_SCALE` | `models.py` | Scale of the loudness feature |

The defaults are starting points and have not been tuned on real recordings. Retrain after changing preprocessing, the microphone, or its position.

## Project layout

| Path | Purpose |
| --- | --- |
| `bakeoff_app.py` | The app (recommended entry point) |
| `bakeoff.py` | Prediction smoothing and trial scoring |
| `models.py` | Features, models, saving and loading, C++ export |
| `visualizer.py` | The original starter-style interface, kept for reference |
| `src/` | Audio capture and FFT code |
| `export_model_header.py` | Converts a trained LinearSVC `.model` file to a C++ header |
| `tests/` | Unit tests |
| `models/`, `recordings/`, `metadata/` | Saved models, recorded frames, app preferences (git-ignored) |

## Tests

```bash
uv run python -m unittest discover -s tests
```

The tests cover smoothing, trial scoring, and preprocessing. They do not use a microphone.

## Credits

Based on the course vibration-interaction starter. The audio streaming code derives from [Realtime_PyAudio_FFT](https://github.com/aiXander/Realtime_PyAudio_FFT) by Xander Steenbrugge (MIT). See `LICENSE`.
