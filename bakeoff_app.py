"""Desk Vibration Lab: live knock / scratch recognizer for the vibration bakeoff.

A restyled front end for the same pipeline as visualizer.py (models.py features,
bakeoff.py smoothing and trial scoring). Run with: uv run bakeoff_app.py

Workflow tabs on the right: 1 Collect -> 2 Train -> 3 Bakeoff.
Space bar starts/stops a bakeoff trial from any tab.
"""

import json
import os
import re
import threading
from collections import deque
from pathlib import Path

import numpy as np
import dearpygui.dearpygui as dpg
import pyaudio

from models import (
    DEFAULT_MODEL_TYPE,
    LOUDNESS_SCALE,
    MODEL_OPTIONS,
    get_audio_features,
    get_ear,
    load_model,
    predict,
    preprocess_data,
    save_model,
    train_model,
)
from bakeoff import (
    DEFAULT_CLASSES,
    NEUTRAL,
    TRIAL_KINDS,
    PredictionSmoother,
    evaluate_trial,
    format_results,
)

METADATA_PATH = Path("metadata/metadata.json")
RECORDINGS_DIR = Path("recordings")
MODELS_DIR = Path(__file__).resolve().parent / "models"  # default folder for the Save/Load dialogs
DEFAULT_DATASET = "session1"
FRAMES_GOAL = 300  # target number of recorded frames per class
HISTORY_LEN = 360  # loudness samples kept for the history plot
MIN_PLOT_HZ = 20.0
RIGHT_PANEL_W = 380

# ---- palette ----
BG = (13, 15, 21)
PANEL = (21, 25, 34)
PANEL_HI = (31, 37, 50)
TEXT = (226, 230, 238)
MUTED = (140, 148, 165)
ACCENT = (96, 165, 250)
GOOD = (74, 222, 128)
BAD = (248, 113, 113)
CLASS_COLORS = [
    (148, 163, 184),  # Neutral: slate
    (251, 146, 60),  # orange
    (45, 212, 191),  # teal
    (192, 132, 252),  # violet
    (250, 204, 21),  # amber
    (244, 114, 182),  # pink
]

# ---- state ----
metadata = {}
ear = None
device_names = []
device_index_by_name = {}

collected_data = {}  # {class_name: [spectrum, ...]}
record_cls = None  # class currently being recorded, or None
model = None
infer_classes = []
holdout_text = ""

smoother = PredictionSmoother()
trial_active = False
trial_results = set()
trial_tally = {kind: [0, 0] for kind in TRIAL_KINDS}
loudness_history = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
last_fft_count = -1
raw_label = "---"
skipped_quiet = 0  # frames dropped by the quiet gate in the current recording
GATE_MARGIN = 0.02  # added above the Neutral 95th percentile (loudness units)
GATE_FALLBACK_MARGIN = 0.05  # added above the recent quiet level when no Neutral data exists


# ---- helpers ----
def color_for(cls):
    if cls == NEUTRAL:
        return CLASS_COLORS[0]
    names = [c for c in collected_data if c != NEUTRAL] or [c for c in infer_classes if c != NEUTRAL]
    if cls in names:
        return CLASS_COLORS[1 + names.index(cls) % (len(CLASS_COLORS) - 1)]
    return ACCENT


def load_metadata():
    defaults = {
        "default_device": "",
        "data_classes": list(DEFAULT_CLASSES),
        "model_type": DEFAULT_MODEL_TYPE,
        "dataset_name": DEFAULT_DATASET,
    }
    try:
        with open(METADATA_PATH, "r") as f:
            return defaults | json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return defaults


def save_metadata():
    METADATA_PATH.parent.mkdir(exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)


def loudness_of(amps):
    return float(np.log10(1.0 + np.maximum(amps, 0.0).sum()) / LOUDNESS_SCALE)


def current_gate():
    """Loudness a frame must reach to be saved for a non-Neutral class.

    A manual slider value above 0 wins. Otherwise: just above the loudest typical
    Neutral frame if Neutral was recorded, else just above the recent quiet level.
    """
    manual = dpg.get_value("gate_slider")
    if manual > 0:
        return float(manual)
    neutral = collected_data.get(NEUTRAL, [])
    if len(neutral) >= 20:
        levels = [loudness_of(f) for f in neutral]
        return float(np.percentile(levels, 95)) + GATE_MARGIN
    recent = [v for v in loudness_history if v > 0]
    if recent:
        return float(np.percentile(recent, 25)) + GATE_FALLBACK_MARGIN
    return 0.0


def discover_devices():
    pa = pyaudio.PyAudio()
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            if info.get("maxInputChannels", 0) > 0 and info["name"] not in device_index_by_name:
                device_names.append(info["name"])
                device_index_by_name[info["name"]] = idx
    finally:
        pa.terminate()


def open_device(name):
    """(Re)open the analyzer on a device, closing the previous stream first."""
    global ear, last_fft_count
    if ear is not None:
        try:
            ear.stream_reader.terminate()
        except Exception as e:
            print(f"Could not close previous stream: {e}")
    ear = get_ear(device=device_index_by_name[name])
    last_fft_count = -1


# ---- class management ----
def add_class(name):
    name = (name or "").strip()
    dpg.set_value("class_input", "")
    if not name or name in collected_data:
        return
    collected_data[name] = []
    metadata["data_classes"] = list(collected_data)
    save_metadata()
    refresh_class_list()


def remove_class(name):
    global record_cls
    collected_data.pop(name, None)
    if record_cls == name:
        record_cls = None
    metadata["data_classes"] = list(collected_data)
    save_metadata()
    refresh_class_list()


def clear_class(name):
    collected_data[name] = []
    refresh_class_list()


def toggle_record(name):
    global record_cls, skipped_quiet
    was_recording = record_cls is not None
    record_cls = None if record_cls == name else name
    if record_cls is not None:
        skipped_quiet = 0
    refresh_class_list()
    if was_recording and record_cls is None and dpg.get_value("autosave_box"):
        save_dataset()


def refresh_class_list():
    dpg.delete_item("class_list", children_only=True)
    for name in collected_data:
        col = color_for(name)
        with dpg.group(parent="class_list"):
            with dpg.group(horizontal=True):
                dpg.add_text(name, color=col)
                dpg.add_text(f"{len(collected_data[name])} frames", tag=f"cnt_{name}", color=MUTED)
            dpg.add_progress_bar(
                tag=f"bar_{name}",
                default_value=min(1.0, len(collected_data[name]) / FRAMES_GOAL),
                width=-1,
                height=8,
            )
            with dpg.group(horizontal=True):
                recording = record_cls == name
                dpg.add_button(
                    label="Stop" if recording else "Record",
                    tag=f"rec_{name}",
                    width=110,
                    callback=lambda s, a, u: toggle_record(u),
                    user_data=name,
                )
                dpg.bind_item_theme(f"rec_{name}", "rec_theme" if recording else "btn_theme")
                dpg.add_button(label="Clear", callback=lambda s, a, u: clear_class(u), user_data=name)
                dpg.add_button(label="Remove", callback=lambda s, a, u: remove_class(u), user_data=name)
            dpg.add_spacer(height=6)
    if record_cls:
        dpg.set_value("record_hint", f"Recording '{record_cls}'. Do the interaction repeatedly, then press Stop.")
    else:
        dpg.set_value("record_hint", "Press Record on a class, repeat that interaction, then press Stop.")


# ---- saving and loading recorded frames ----
def safe_name(text):
    """Folder-safe dataset name: letters, digits, underscore, dash, space."""
    cleaned = re.sub(r"[^\w\- ]", "_", (text or "").strip()).strip()
    return cleaned or DEFAULT_DATASET


def list_datasets():
    if not RECORDINGS_DIR.exists():
        return []
    return sorted(p.name for p in RECORDINGS_DIR.iterdir() if (p / "meta.json").exists())


def set_data_status(text, color=MUTED):
    dpg.set_value("data_status", text)
    dpg.configure_item("data_status", color=color)


def refresh_dataset_list(select=None):
    names = list_datasets()
    dpg.configure_item("dataset_combo", items=names)
    if select in names:
        dpg.set_value("dataset_combo", select)


def current_feature_length():
    """Length of one spectrum from the live analyzer, or None if no device is open."""
    return len(ear.fft) if ear is not None else None


def save_dataset(name=None):
    """Write every class's frames to recordings/<name>/ (class_<i>.npy + meta.json)."""
    name = safe_name(name if name is not None else dpg.get_value("dataset_name"))
    folder = RECORDINGS_DIR / name
    try:
        folder.mkdir(parents=True, exist_ok=True)
        classes, n_features = [], None
        for i, (cls, frames) in enumerate(collected_data.items()):
            if not frames:
                classes.append({"name": cls, "file": None})
                continue
            arr = np.asarray(frames, dtype=np.float32)
            n_features = arr.shape[1]
            fname = f"class_{i}.npy"
            np.save(folder / fname, arr)
            classes.append({"name": cls, "file": fname})
        meta = {
            "classes": classes,
            "n_features": n_features,
            "rate": getattr(ear, "rate", None),
        }
        with open(folder / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)
        keep = {c["file"] for c in classes if c["file"]}
        for stale in folder.glob("class_*.npy"):
            if stale.name not in keep:
                stale.unlink()
    except OSError as e:
        set_data_status(f"Save failed: {e}", BAD)
        return False
    metadata["dataset_name"] = name
    save_metadata()
    total = sum(len(v) for v in collected_data.values())
    set_data_status(f"Saved {total} frames to recordings/{name}/", GOOD)
    refresh_dataset_list(select=name)
    return True


def load_dataset(name, append=False):
    """Load recordings/<name>/. Replaces current data unless append is True."""
    global record_cls
    name = safe_name(name)
    folder = RECORDINGS_DIR / name
    try:
        with open(folder / "meta.json", "r") as f:
            meta = json.load(f)
        loaded = {}
        for entry in meta["classes"]:
            loaded[entry["name"]] = (
                np.load(folder / entry["file"]).astype(np.float64) if entry["file"] else np.empty((0, 0))
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        set_data_status(f"Could not load '{name}': {e}", BAD)
        return False

    expected = current_feature_length()
    saved_len = meta.get("n_features")
    if expected is not None and saved_len and saved_len != expected:
        set_data_status(
            f"'{name}' was recorded with {saved_len} values per frame but this device gives {expected} "
            f"(different sample rate or device). Not loaded.",
            BAD,
        )
        return False

    record_cls = None
    if not append:
        collected_data.clear()
    for cls, arr in loaded.items():
        existing = collected_data.setdefault(cls, [])
        existing.extend(list(arr) if arr.size else [])
    metadata["data_classes"] = list(collected_data)
    metadata["dataset_name"] = name
    save_metadata()
    dpg.set_value("dataset_name", name)
    refresh_class_list()
    total = sum(len(v) for v in collected_data.values())
    set_data_status(f"{'Added' if append else 'Loaded'} '{name}': {total} frames in total", GOOD)
    return True


# ---- training ----
def set_train_status(text, color=MUTED):
    dpg.set_value("train_status", text)
    dpg.configure_item("train_status", color=color)


def train_callback():
    filled = [c for c, v in collected_data.items() if v]
    if len(filled) < 2:
        set_train_status("Record data for at least 2 classes first.", BAD)
        return
    dpg.configure_item("train_button", enabled=False)
    set_train_status("Training...", ACCENT)
    model_type = dpg.get_value("model_combo")
    snapshot = {k: list(v) for k, v in collected_data.items()}

    def work():
        global model, infer_classes, holdout_text
        try:
            xs, ys, names = [], [], []
            for i, (cls, frames) in enumerate(snapshot.items()):
                xs.extend(frames)
                ys.extend([i] * len(frames))
                names.append(cls)
            X = preprocess_data(np.array(xs))
            y = np.array(ys)

            holdout_text = ""
            counts = np.bincount(y, minlength=len(names))
            if all(c >= 10 for c in counts if c > 0):
                from sklearn.model_selection import train_test_split

                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
                probe, _ = train_model(Xtr, ytr, names, model_type)
                acc = float((predict(probe, Xte) == yte).mean())
                holdout_text = f"Holdout accuracy: {acc * 100:.1f}% (optimistic: neighboring frames look alike)"

            model, infer_classes = train_model(X, y, names, model_type)
            smoother.reset()
            set_train_status("Trained. Switch to the Bakeoff tab or just try it live.", GOOD)
            dpg.set_value("holdout_text", holdout_text)
        except Exception as e:
            set_train_status(f"Training failed: {e}", BAD)
        finally:
            dpg.configure_item("train_button", enabled=True)

    threading.Thread(target=work, daemon=True).start()


def pick_file(save, on_pick):
    """Show the native Windows Open/Save dialog off the UI thread, then call on_pick(path)."""

    def work():
        import tkinter as tk
        from tkinter import filedialog

        MODELS_DIR.mkdir(exist_ok=True)
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        types = [("Model files", "*.model"), ("All files", "*.*")]
        try:
            if save:
                path = filedialog.asksaveasfilename(
                    parent=root, title="Save model", defaultextension=".model",
                    initialdir=str(MODELS_DIR), initialfile="model.model", filetypes=types,
                )
            else:
                path = filedialog.askopenfilename(
                    parent=root, title="Load model", initialdir=str(MODELS_DIR), filetypes=types
                )
        finally:
            root.destroy()
        if path:
            on_pick(path)

    threading.Thread(target=work, daemon=True).start()


def save_model_to(path):
    if model is None:
        set_train_status("Nothing to save yet.", BAD)
        return
    try:
        save_model(model, infer_classes, Path(path))
        set_train_status(f"Saved {Path(path).name}", GOOD)
    except OSError as e:
        set_train_status(f"Save failed: {e}", BAD)


def load_model_from(path):
    global model, infer_classes
    try:
        model, infer_classes = load_model(Path(path))
        smoother.reset()
        set_train_status(f"Loaded {Path(path).name}", GOOD)
    except Exception as e:
        set_train_status(f"Load failed: {e}", BAD)


# ---- bakeoff trials ----
def update_tally():
    dpg.set_value(
        "tally_text",
        "\n".join(f"{kind}:  {p} / {n}" for kind, (p, n) in trial_tally.items()),
    )


def toggle_trial():
    global trial_active
    if model is None:
        dpg.set_value("trial_status", "Train or load a model first.")
        dpg.configure_item("trial_status", color=BAD)
        return
    if dpg.is_item_active("class_input") or dpg.is_item_focused("class_input"):
        return  # space belongs to the text box
    if not trial_active:
        trial_results.clear()
        smoother.reset()
        trial_active = True
        dpg.set_value("trial_status", "RECORDING - perform the interaction, then press space")
        dpg.configure_item("trial_status", color=GOOD)
        dpg.set_value("trial_result", "")
        return
    trial_active = False
    text = format_results(trial_results)
    dpg.set_value("trial_status", "Stopped. Press space for the next trial.")
    dpg.configure_item("trial_status", color=MUTED)
    dpg.set_value("trial_result", text)
    print(f"Trial results: {text}")


def score_trial():
    if trial_active or not trial_results:
        return
    kind = dpg.get_value("trial_kind")
    passed = evaluate_trial(kind, trial_results)
    trial_tally[kind][0] += int(passed)
    trial_tally[kind][1] += 1
    trial_results.clear()
    dpg.set_value("trial_result", f"{'PASS' if passed else 'FAIL'}: {kind}")
    dpg.configure_item("trial_result", color=GOOD if passed else BAD)
    update_tally()


def reset_tally():
    for counts in trial_tally.values():
        counts[0] = counts[1] = 0
    update_tally()


# ---- per-frame update ----
def update_frame():
    global last_fft_count, raw_label, skipped_quiet
    freqs, amps = get_audio_features(ear)
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)

    fresh = True
    if ear is not None:
        fresh = ear.num_ffts != last_fft_count
        last_fft_count = ear.num_ffts

    # Spectrum plot (frequencies below MIN_PLOT_HZ are hidden: log axis, DC rumble)
    keep = freqs >= MIN_PLOT_HZ
    if keep.any():
        xs, ys = freqs[keep].tolist(), amps[keep].tolist()
        dpg.set_value("spec_line", [xs, ys])
        dpg.configure_item("spec_fill", x=xs, y1=ys)
        peak = int(np.argmax(amps[keep]))
        dpg.set_value("peak_text", f"Peak {xs[peak]:.0f} Hz")
        dpg.set_axis_limits("x_axis", MIN_PLOT_HZ, xs[-1])
        dpg.set_axis_limits("y_axis", 0, dpg.get_value("y_max"))

    if not fresh:
        return

    level = loudness_of(amps)
    loudness_history.append(level)
    dpg.set_value("loud_bar", min(1.0, level))
    dpg.configure_item("loud_bar", overlay=f"level {level:.2f}")
    hist = list(loudness_history)
    dpg.set_value("hist_line", [list(range(len(hist))), hist])

    gate = current_gate()
    dpg.set_value("gate_line", [[0, HISTORY_LEN - 1], [gate, gate]])
    if record_cls is not None and record_cls in collected_data:
        # Neutral keeps every frame; event classes keep only frames above the gate,
        # so pauses between knocks/scratches are not mislabeled as the event.
        if record_cls == NEUTRAL or level >= gate:
            collected_data[record_cls].append(amps)
            n = len(collected_data[record_cls])
            dpg.set_value(f"cnt_{record_cls}", f"{n} frames")
            dpg.set_value(f"bar_{record_cls}", min(1.0, n / FRAMES_GOAL))
        else:
            skipped_quiet += 1
        if record_cls != NEUTRAL:
            dpg.set_value("gate_text", f"Quiet gate {gate:.2f} | skipped {skipped_quiet} quiet frames")
        else:
            dpg.set_value("gate_text", "Neutral: keeping every frame")

    if model is None:
        dpg.set_value("pred_big", "---")
        dpg.configure_item("pred_big", color=MUTED)
        dpg.set_value("pred_sub", "Record data and train a model to see live predictions.")
        return

    idx = int(predict(model, preprocess_data(amps[None, :]))[0])
    if not 0 <= idx < len(infer_classes):
        dpg.set_value("pred_sub", "Model/class mismatch: retrain.")
        return
    raw_label = infer_classes[idx]
    label = smoother.update(raw_label)
    dpg.set_value("pred_big", label)
    dpg.configure_item("pred_big", color=color_for(label))
    dpg.set_value("pred_sub", f"raw frame: {raw_label}" + ("   |   trial recording" if trial_active else ""))
    if trial_active:
        trial_results.add(label)
        dpg.set_value("trial_live", "Seen so far: " + format_results(trial_results))


# ---- UI ----
def make_themes():
    with dpg.theme(tag="global_theme"):
        with dpg.theme_component(dpg.mvAll):
            for target, col in [
                (dpg.mvThemeCol_WindowBg, BG),
                (dpg.mvThemeCol_ChildBg, PANEL),
                (dpg.mvThemeCol_PopupBg, PANEL_HI),
                (dpg.mvThemeCol_FrameBg, PANEL_HI),
                (dpg.mvThemeCol_FrameBgHovered, (42, 50, 68)),
                (dpg.mvThemeCol_FrameBgActive, (50, 60, 82)),
                (dpg.mvThemeCol_Text, TEXT),
                (dpg.mvThemeCol_Border, (40, 46, 62)),
                (dpg.mvThemeCol_Button, (45, 55, 78)),
                (dpg.mvThemeCol_ButtonHovered, (62, 76, 108)),
                (dpg.mvThemeCol_ButtonActive, (80, 98, 138)),
                (dpg.mvThemeCol_Tab, PANEL_HI),
                (dpg.mvThemeCol_TabHovered, (62, 76, 108)),
                (dpg.mvThemeCol_TabActive, (55, 90, 150)),
                (dpg.mvThemeCol_PlotHistogram, ACCENT),
                (dpg.mvThemeCol_SliderGrab, ACCENT),
                (dpg.mvThemeCol_Header, (45, 55, 78)),
            ]:
                dpg.add_theme_color(target, col)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 12)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 8)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 12, 8)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 16, 16)
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, (16, 19, 27), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg, PANEL, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder, (40, 46, 62), category=dpg.mvThemeCat_Plots)

    with dpg.theme(tag="btn_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (45, 55, 78))
    with dpg.theme(tag="rec_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (190, 60, 60))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (215, 80, 80))
    with dpg.theme(tag="accent_btn_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (55, 105, 190))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (75, 130, 220))
    with dpg.theme(tag="gate_theme"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (250, 204, 21), category=dpg.mvThemeCat_Plots)
    with dpg.theme(tag="line_theme"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, ACCENT, category=dpg.mvThemeCat_Plots)
        with dpg.theme_component(dpg.mvShadeSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (96, 165, 250, 70), category=dpg.mvThemeCat_Plots)


def make_fonts():
    fonts = {}
    regular, bold = r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"
    with dpg.font_registry():
        if os.path.exists(regular):
            fonts["body"] = dpg.add_font(regular, 17)
            dpg.bind_font(fonts["body"])
        if os.path.exists(bold):
            fonts["title"] = dpg.add_font(bold, 26)
            fonts["big"] = dpg.add_font(bold, 84)
            fonts["mid"] = dpg.add_font(bold, 20)
    return fonts


def build_ui():
    make_themes()
    fonts = make_fonts()

    with dpg.window(tag="root", no_title_bar=True, no_move=True, no_resize=True, no_scrollbar=True):
        with dpg.group(horizontal=True):
            # ---- left: live view ----
            with dpg.child_window(tag="left_panel", border=False, width=-RIGHT_PANEL_W - 14):
                with dpg.group(horizontal=True):
                    title = dpg.add_text("Desk Vibration Lab")
                    if "title" in fonts:
                        dpg.bind_item_font(title, fonts["title"])
                    dpg.add_spacer(width=20)
                    dpg.add_combo(
                        items=device_names,
                        default_value=metadata["default_device"] or (device_names[0] if device_names else ""),
                        tag="device_combo",
                        width=360,
                        callback=lambda s, a: (metadata.update(default_device=a), save_metadata(), open_device(a)),
                    )
                dpg.add_text("Knock vs. scratch, classified from desk vibration.", color=MUTED)

                with dpg.child_window(height=190, border=False):
                    dpg.add_text("LIVE PREDICTION", color=MUTED)
                    big = dpg.add_text("---", tag="pred_big", color=MUTED)
                    if "big" in fonts:
                        dpg.bind_item_font(big, fonts["big"])
                    dpg.add_text("", tag="pred_sub", color=MUTED)

                dpg.add_progress_bar(tag="loud_bar", width=-1, height=22, overlay="level 0.00")

                with dpg.group(horizontal=True):
                    dpg.add_text("Spectrum (log frequency)", color=MUTED)
                    dpg.add_text("", tag="peak_text", color=ACCENT)
                with dpg.plot(height=300, width=-1, no_title=True, no_mouse_pos=True):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="x_axis", label="Hz", scale=dpg.mvPlotScale_Log10)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="y_axis", label="amplitude")
                    dpg.add_shade_series([], [], tag="spec_fill", parent="y_axis")
                    dpg.add_line_series([], [], tag="spec_line", parent="y_axis")
                    dpg.bind_item_theme("spec_fill", "line_theme")
                    dpg.bind_item_theme("spec_line", "line_theme")
                dpg.add_slider_int(
                    tag="y_max", label="plot max", default_value=500_000, min_value=10_000, max_value=2_000_000, width=300
                )

                dpg.add_text("Loudness history (about 6 s)", color=MUTED)
                with dpg.plot(height=-1, width=-1, no_title=True, no_mouse_pos=True, no_menus=True):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="hx_axis", no_tick_labels=True, no_gridlines=True)
                    dpg.add_plot_axis(dpg.mvYAxis, tag="hy_axis", no_tick_labels=True)
                    dpg.add_line_series(list(range(HISTORY_LEN)), [0.0] * HISTORY_LEN, tag="hist_line", parent="hy_axis")
                    dpg.bind_item_theme("hist_line", "line_theme")
                    dpg.add_line_series([0, HISTORY_LEN - 1], [0.0, 0.0], tag="gate_line", parent="hy_axis")
                    dpg.bind_item_theme("gate_line", "gate_theme")
                    dpg.set_axis_limits("hx_axis", 0, HISTORY_LEN - 1)
                    dpg.set_axis_limits("hy_axis", 0, 1.2)

            # ---- right: workflow tabs ----
            with dpg.child_window(tag="right_panel", border=False, width=RIGHT_PANEL_W):
                with dpg.tab_bar():
                    with dpg.tab(label="1  Collect"):
                        dpg.add_text("Recorded data set", color=MUTED)
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag="dataset_name", default_value=metadata["dataset_name"], width=170)
                            dpg.add_button(label="Save", callback=lambda: save_dataset())
                        with dpg.group(horizontal=True):
                            dpg.add_combo(items=list_datasets(), tag="dataset_combo", width=170)
                            dpg.add_button(
                                label="Load",
                                callback=lambda: load_dataset(
                                    dpg.get_value("dataset_combo"), append=dpg.get_value("append_box")
                                ),
                            )
                        dpg.add_checkbox(label="Add to current data when loading", tag="append_box")
                        dpg.add_checkbox(label="Autosave when a recording stops", tag="autosave_box",
                                         default_value=True)
                        dpg.add_text("", tag="data_status", wrap=RIGHT_PANEL_W - 50, color=MUTED)
                        dpg.add_separator()
                        dpg.add_text("", tag="record_hint", wrap=RIGHT_PANEL_W - 50, color=MUTED)
                        dpg.add_text("Tip: record Neutral first so the quiet gate matches your room.",
                                     wrap=RIGHT_PANEL_W - 50, color=MUTED)
                        dpg.add_text("", tag="gate_text", wrap=RIGHT_PANEL_W - 50, color=ACCENT)
                        dpg.add_slider_float(tag="gate_slider", label="gate (0 = auto)", default_value=0.0,
                                             min_value=0.0, max_value=1.0, width=180)
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag="class_input", hint="new class name", width=200, on_enter=True,
                                               callback=lambda: add_class(dpg.get_value("class_input")))
                            dpg.add_button(label="Add", callback=lambda: add_class(dpg.get_value("class_input")))
                        dpg.add_child_window(tag="class_list", border=False, height=-1)

                    with dpg.tab(label="2  Train"):
                        dpg.add_text("Model", color=MUTED)
                        dpg.add_combo(
                            items=MODEL_OPTIONS,
                            default_value=metadata["model_type"],
                            tag="model_combo",
                            width=-1,
                            callback=lambda s, a: (metadata.update(model_type=a), save_metadata()),
                        )
                        dpg.add_button(label="Train Model", tag="train_button", width=-1, callback=train_callback)
                        dpg.bind_item_theme("train_button", "accent_btn_theme")
                        dpg.add_text("", tag="train_status", wrap=RIGHT_PANEL_W - 50, color=MUTED)
                        dpg.add_text("", tag="holdout_text", wrap=RIGHT_PANEL_W - 50, color=MUTED)
                        dpg.add_separator()
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Save Model", callback=lambda: pick_file(True, save_model_to))
                            dpg.add_button(label="Load Model", callback=lambda: pick_file(False, load_model_from))

                    with dpg.tab(label="3  Bakeoff"):
                        dpg.add_text("Space bar starts and stops a trial.", color=MUTED)
                        dpg.add_text("Train or load a model first.", tag="trial_status", wrap=RIGHT_PANEL_W - 50, color=MUTED)
                        dpg.add_text("", tag="trial_live", wrap=RIGHT_PANEL_W - 50, color=MUTED)
                        res = dpg.add_text("", tag="trial_result", wrap=RIGHT_PANEL_W - 50)
                        if "mid" in fonts:
                            dpg.bind_item_font(res, fonts["mid"])
                        dpg.add_separator()
                        dpg.add_text("Self-test: what did the participant do?", color=MUTED)
                        dpg.add_combo(items=TRIAL_KINDS, default_value=TRIAL_KINDS[0], tag="trial_kind", width=-1)
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Score Trial", callback=score_trial)
                            dpg.add_button(label="Reset Tally", callback=reset_tally)
                        dpg.add_text("", tag="tally_text")

    with dpg.handler_registry():
        dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=lambda: toggle_trial())

    dpg.bind_theme("global_theme")
    for cls in metadata["data_classes"]:
        collected_data.setdefault(cls, [])
    refresh_class_list()
    update_tally()
    refresh_dataset_list(select=metadata["dataset_name"])
    if metadata["dataset_name"] in list_datasets():
        load_dataset(metadata["dataset_name"])  # resume where the last session stopped


def fit_to_viewport(*_):
    dpg.set_item_width("root", dpg.get_viewport_client_width())
    dpg.set_item_height("root", dpg.get_viewport_client_height())
    dpg.configure_item("root", pos=(0, 0))


def main():
    global metadata
    metadata = load_metadata()
    if metadata["model_type"] not in MODEL_OPTIONS:
        metadata["model_type"] = DEFAULT_MODEL_TYPE

    discover_devices()
    if metadata["default_device"] not in device_names:
        metadata["default_device"] = device_names[0] if device_names else ""
    if metadata["default_device"]:
        open_device(metadata["default_device"])

    dpg.create_context()
    build_ui()
    dpg.create_viewport(title="Desk Vibration Lab", width=1320, height=800, min_width=1000, min_height=620)
    dpg.set_viewport_resize_callback(fit_to_viewport)
    dpg.setup_dearpygui()
    dpg.set_primary_window("root", True)
    dpg.show_viewport()
    fit_to_viewport()

    max_frames = int(os.environ.get("BAKEOFF_MAX_FRAMES", "0"))  # smoke-test hook
    frames = 0
    while dpg.is_dearpygui_running():
        update_frame()
        dpg.render_dearpygui_frame()
        frames += 1
        if max_frames and frames >= max_frames:
            break
    if ear is not None:
        ear.stream_reader.terminate()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
